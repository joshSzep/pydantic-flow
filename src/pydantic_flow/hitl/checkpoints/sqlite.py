"""SQLite checkpoint store implementation.

Stores checkpoints in a local SQLite database with WAL mode for performance.
"""

from __future__ import annotations

from pathlib import Path

import aiosqlite
from pydantic import BaseModel

from pydantic_flow.hitl.checkpoints.base import BaseCheckpointStore
from pydantic_flow.hitl.checkpoints.interface import CheckpointConflict
from pydantic_flow.hitl.checkpoints.interface import CheckpointEnvelope
from pydantic_flow.hitl.checkpoints.interface import CheckpointId
from pydantic_flow.hitl.checkpoints.interface import CheckpointQuery
from pydantic_flow.hitl.checkpoints.interface import RunId
from pydantic_flow.hitl.checkpoints.interface import SortOrder
from pydantic_flow.hitl.checkpoints.serde import deserialize_checkpoint
from pydantic_flow.hitl.checkpoints.serde import serialize_checkpoint


class SQLiteCheckpointStoreConfig(BaseModel):
    """Configuration for SQLite checkpoint store.

    Attributes:
        db_path: Path to SQLite database file.
        busy_timeout_ms: Timeout for locked database operations.

    """

    db_path: Path
    busy_timeout_ms: int = 30000


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS checkpoints (
    run_id TEXT NOT NULL,
    checkpoint_id TEXT PRIMARY KEY,
    node_id TEXT,
    created_at TIMESTAMP NOT NULL,
    schema_version INTEGER NOT NULL,
    envelope_json TEXT NOT NULL,
    content_hash TEXT
);

CREATE INDEX IF NOT EXISTS idx_run_created
    ON checkpoints(run_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_run_node_created
    ON checkpoints(run_id, node_id, created_at DESC);
"""


class SQLiteCheckpointStore(BaseCheckpointStore):
    """SQLite-based checkpoint store with WAL mode."""

    def __init__(self, config: SQLiteCheckpointStoreConfig) -> None:
        """Initialize the SQLite store.

        Args:
            config: Store configuration.

        """
        self.config = config
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Ensure database schema is initialized."""
        if self._initialized:
            return

        self.config.db_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiosqlite.connect(self.config.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute(f"PRAGMA busy_timeout={self.config.busy_timeout_ms}")
            await db.executescript(_SCHEMA_SQL)
            await db.commit()

        self._initialized = True

    async def _do_save(
        self, envelope: CheckpointEnvelope, overwrite: bool
    ) -> CheckpointEnvelope:
        """Save checkpoint to SQLite.

        Args:
            envelope: The prepared checkpoint envelope with computed hash.
            overwrite: If False, raise CheckpointConflict if ID exists.

        Returns:
            The saved envelope.

        Raises:
            CheckpointConflict: If checkpoint exists and overwrite=False.

        """
        await self._ensure_initialized()

        json_str = serialize_checkpoint(envelope)

        async with aiosqlite.connect(self.config.db_path) as db:
            await db.execute(f"PRAGMA busy_timeout={self.config.busy_timeout_ms}")

            if not overwrite:
                cursor = await db.execute(
                    "SELECT 1 FROM checkpoints WHERE checkpoint_id = ?",
                    (envelope.id,),
                )
                exists = await cursor.fetchone()
                if exists:
                    msg = (
                        f"Checkpoint {envelope.id} already exists "
                        f"for run {envelope.run_id}"
                    )
                    raise CheckpointConflict(msg)

            await db.execute(
                """
                INSERT INTO checkpoints (
                    run_id, checkpoint_id, node_id, created_at,
                    schema_version, envelope_json, content_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(checkpoint_id) DO UPDATE SET
                    envelope_json = excluded.envelope_json,
                    content_hash = excluded.content_hash,
                    created_at = excluded.created_at
                """,
                (
                    envelope.run_id,
                    envelope.id,
                    envelope.node_id,
                    envelope.created_at.isoformat(),
                    envelope.schema_version,
                    json_str,
                    envelope.content_hash,
                ),
            )
            await db.commit()

        return envelope

    async def _do_latest(
        self, run_id: RunId, node_id: str | None = None
    ) -> CheckpointEnvelope | None:
        """Get the most recent checkpoint from SQLite.

        Args:
            run_id: The run to query.
            node_id: Optional node filter.

        Returns:
            The latest checkpoint envelope, or None if not found.

        """
        await self._ensure_initialized()

        async with aiosqlite.connect(self.config.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute(f"PRAGMA busy_timeout={self.config.busy_timeout_ms}")

            if node_id is not None:
                cursor = await db.execute(
                    """
                    SELECT envelope_json FROM checkpoints
                    WHERE run_id = ? AND node_id = ?
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    (run_id, node_id),
                )
            else:
                cursor = await db.execute(
                    """
                    SELECT envelope_json FROM checkpoints
                    WHERE run_id = ?
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    (run_id,),
                )

            row = await cursor.fetchone()
            if row is None:
                return None

            return deserialize_checkpoint(row["envelope_json"])

    async def _do_get(
        self, run_id: RunId, checkpoint_id: CheckpointId
    ) -> CheckpointEnvelope | None:
        """Get a specific checkpoint from SQLite.

        Args:
            run_id: The run identifier.
            checkpoint_id: The checkpoint identifier.

        Returns:
            The checkpoint envelope, or None if not found.

        """
        await self._ensure_initialized()

        async with aiosqlite.connect(self.config.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute(f"PRAGMA busy_timeout={self.config.busy_timeout_ms}")

            cursor = await db.execute(
                """
                SELECT envelope_json FROM checkpoints
                WHERE run_id = ? AND checkpoint_id = ?
                """,
                (run_id, checkpoint_id),
            )

            row = await cursor.fetchone()
            if row is None:
                return None

            return deserialize_checkpoint(row["envelope_json"])

    async def _do_list(
        self, query: CheckpointQuery
    ) -> tuple[list[CheckpointEnvelope], str | None]:
        """List checkpoints from SQLite.

        Args:
            query: Query parameters for filtering and pagination.

        Returns:
            Tuple of (list of checkpoint envelopes, next cursor or None).

        """
        await self._ensure_initialized()

        async with aiosqlite.connect(self.config.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute(f"PRAGMA busy_timeout={self.config.busy_timeout_ms}")

            conditions = []
            params: list[str | int] = []

            if query.run_id is not None:
                conditions.append("run_id = ?")
                params.append(query.run_id)

            if query.node_id is not None:
                conditions.append("node_id = ?")
                params.append(query.node_id)

            if query.since is not None:
                conditions.append("created_at >= ?")
                params.append(query.since.isoformat())

            if query.until is not None:
                conditions.append("created_at <= ?")
                params.append(query.until.isoformat())

            where_clause = " AND ".join(conditions) if conditions else "1=1"

            order = "DESC" if query.sort_order == SortOrder.DESC else "ASC"

            cursor_offset = 0
            if query.cursor is not None:
                try:
                    cursor_offset = int(query.cursor)
                except ValueError:
                    cursor_offset = 0

            sql = f"""
                SELECT envelope_json FROM checkpoints
                WHERE {where_clause}
                ORDER BY created_at {order}
                LIMIT ? OFFSET ?
            """
            params.extend([query.limit + 1, cursor_offset])

            cursor = await db.execute(sql, params)
            rows = await cursor.fetchall()

            envelopes = [deserialize_checkpoint(row["envelope_json"]) for row in rows]

            next_cursor = None
            if len(envelopes) > query.limit:
                envelopes = envelopes[: query.limit]
                next_cursor = str(cursor_offset + query.limit)

            return envelopes, next_cursor

    async def _do_delete(self, run_id: RunId, checkpoint_id: CheckpointId) -> bool:
        """Delete a checkpoint from SQLite.

        Args:
            run_id: The run identifier.
            checkpoint_id: The checkpoint identifier.

        Returns:
            True if deleted, False if didn't exist.

        """
        await self._ensure_initialized()

        async with aiosqlite.connect(self.config.db_path) as db:
            await db.execute(f"PRAGMA busy_timeout={self.config.busy_timeout_ms}")

            cursor = await db.execute(
                """
                DELETE FROM checkpoints
                WHERE run_id = ? AND checkpoint_id = ?
                """,
                (run_id, checkpoint_id),
            )
            await db.commit()

            return cursor.rowcount > 0

    async def _do_purge(self, run_id: RunId) -> int:
        """Delete all checkpoints for a run from SQLite.

        Args:
            run_id: The run identifier.

        Returns:
            Number of checkpoints deleted.

        """
        await self._ensure_initialized()

        async with aiosqlite.connect(self.config.db_path) as db:
            await db.execute(f"PRAGMA busy_timeout={self.config.busy_timeout_ms}")

            cursor = await db.execute(
                "DELETE FROM checkpoints WHERE run_id = ?",
                (run_id,),
            )
            await db.commit()

            return cursor.rowcount

    async def _do_healthcheck(self) -> bool:
        """Verify SQLite database health.

        Returns:
            True if healthy.

        """
        await self._ensure_initialized()

        async with aiosqlite.connect(self.config.db_path) as db:
            await db.execute("SELECT 1")
        return True

    def __repr__(self) -> str:
        """Return a string representation of the store."""
        return f"SQLiteCheckpointStore(db_path={self.config.db_path})"
