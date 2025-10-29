"""SQLite checkpoint store implementation.

Stores checkpoints in a local SQLite database with WAL mode for performance.
"""

from __future__ import annotations

from pathlib import Path

import aiosqlite
from pydantic import BaseModel

from pydantic_flow.checkpoints.interface import CheckpointBackendError
from pydantic_flow.checkpoints.interface import CheckpointConflict
from pydantic_flow.checkpoints.interface import CheckpointEnvelope
from pydantic_flow.checkpoints.interface import CheckpointId
from pydantic_flow.checkpoints.interface import CheckpointQuery
from pydantic_flow.checkpoints.interface import RunId
from pydantic_flow.checkpoints.interface import SortOrder
from pydantic_flow.checkpoints.serde import compute_content_hash
from pydantic_flow.checkpoints.serde import deserialize_checkpoint
from pydantic_flow.checkpoints.serde import serialize_checkpoint


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


class SQLiteCheckpointStore:
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

    async def save(
        self, envelope: CheckpointEnvelope, *, overwrite: bool = False
    ) -> CheckpointEnvelope:
        """Save a checkpoint to SQLite.

        Args:
            envelope: The checkpoint envelope to save.
            overwrite: If False, raise CheckpointConflict if ID exists.

        Returns:
            The saved envelope with computed content hash.

        Raises:
            CheckpointConflict: If checkpoint ID exists and overwrite=False.
            CheckpointBackendError: If database operation fails.

        """
        await self._ensure_initialized()

        try:
            envelope_copy = envelope.model_copy(deep=True)
            if envelope_copy.content_hash is None:
                envelope_copy.content_hash = compute_content_hash(envelope_copy)

            json_str = serialize_checkpoint(envelope_copy)

            async with aiosqlite.connect(self.config.db_path) as db:
                await db.execute(f"PRAGMA busy_timeout={self.config.busy_timeout_ms}")

                if not overwrite:
                    cursor = await db.execute(
                        "SELECT 1 FROM checkpoints WHERE checkpoint_id = ?",
                        (envelope_copy.id,),
                    )
                    exists = await cursor.fetchone()
                    if exists:
                        msg = (
                            f"Checkpoint {envelope_copy.id} already exists "
                            f"for run {envelope_copy.run_id}"
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
                        envelope_copy.run_id,
                        envelope_copy.id,
                        envelope_copy.node_id,
                        envelope_copy.created_at.isoformat(),
                        envelope_copy.schema_version,
                        json_str,
                        envelope_copy.content_hash,
                    ),
                )
                await db.commit()

            return envelope_copy

        except CheckpointConflict:
            raise
        except Exception as e:
            msg = f"Failed to save checkpoint: {e}"
            raise CheckpointBackendError(msg, cause=e) from e

    async def latest(
        self, run_id: RunId, node_id: str | None = None
    ) -> CheckpointEnvelope | None:
        """Get the most recent checkpoint for a run.

        Args:
            run_id: The run to query.
            node_id: Optional node filter.

        Returns:
            The latest checkpoint envelope, or None if not found.

        Raises:
            CheckpointBackendError: If database operation fails.

        """
        await self._ensure_initialized()

        try:
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

        except Exception as e:
            msg = f"Failed to get latest checkpoint: {e}"
            raise CheckpointBackendError(msg, cause=e) from e

    async def get(
        self, run_id: RunId, checkpoint_id: CheckpointId
    ) -> CheckpointEnvelope | None:
        """Get a specific checkpoint by ID.

        Args:
            run_id: The run identifier.
            checkpoint_id: The checkpoint identifier.

        Returns:
            The checkpoint envelope, or None if not found.

        Raises:
            CheckpointBackendError: If database operation fails.

        """
        await self._ensure_initialized()

        try:
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

        except Exception as e:
            msg = f"Failed to get checkpoint: {e}"
            raise CheckpointBackendError(msg, cause=e) from e

    async def list(
        self, query: CheckpointQuery
    ) -> tuple[list[CheckpointEnvelope], str | None]:
        """List checkpoints matching query criteria.

        Args:
            query: Query parameters for filtering and pagination.

        Returns:
            Tuple of (list of checkpoint envelopes, next cursor or None).

        Raises:
            CheckpointBackendError: If database operation fails.

        """
        await self._ensure_initialized()

        try:
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

                envelopes = [
                    deserialize_checkpoint(row["envelope_json"]) for row in rows
                ]

                next_cursor = None
                if len(envelopes) > query.limit:
                    envelopes = envelopes[: query.limit]
                    next_cursor = str(cursor_offset + query.limit)

                return envelopes, next_cursor

        except Exception as e:
            msg = f"Failed to list checkpoints: {e}"
            raise CheckpointBackendError(msg, cause=e) from e

    async def delete(self, run_id: RunId, checkpoint_id: CheckpointId) -> bool:
        """Delete a specific checkpoint.

        Args:
            run_id: The run identifier.
            checkpoint_id: The checkpoint identifier.

        Returns:
            True if checkpoint was deleted, False if it didn't exist.

        Raises:
            CheckpointBackendError: If database operation fails.

        """
        await self._ensure_initialized()

        try:
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

        except Exception as e:
            msg = f"Failed to delete checkpoint: {e}"
            raise CheckpointBackendError(msg, cause=e) from e

    async def purge(self, run_id: RunId) -> int:
        """Delete all checkpoints for a run.

        Args:
            run_id: The run identifier.

        Returns:
            Number of checkpoints deleted.

        Raises:
            CheckpointBackendError: If database operation fails.

        """
        await self._ensure_initialized()

        try:
            async with aiosqlite.connect(self.config.db_path) as db:
                await db.execute(f"PRAGMA busy_timeout={self.config.busy_timeout_ms}")

                cursor = await db.execute(
                    "DELETE FROM checkpoints WHERE run_id = ?",
                    (run_id,),
                )
                await db.commit()

                return cursor.rowcount

        except Exception as e:
            msg = f"Failed to purge checkpoints: {e}"
            raise CheckpointBackendError(msg, cause=e) from e

    async def healthcheck(self) -> bool:
        """Verify database connectivity and permissions.

        Raises:
            CheckpointBackendError: If database is unhealthy.

        """
        try:
            await self._ensure_initialized()

            async with aiosqlite.connect(self.config.db_path) as db:
                await db.execute("SELECT 1")
            return True

        except Exception as e:
            msg = f"Healthcheck failed: {e}"
            raise CheckpointBackendError(msg, cause=e) from e

    def __repr__(self) -> str:
        """Return a string representation of the store."""
        return f"SQLiteCheckpointStore(db_path={self.config.db_path})"
