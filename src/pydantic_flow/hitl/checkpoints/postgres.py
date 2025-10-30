"""Postgres checkpoint store implementation.

Stores checkpoints in PostgreSQL with JSONB for efficient querying.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    pass


class PostgresCheckpointStoreConfig(BaseModel):
    """Configuration for Postgres checkpoint store.

    Attributes:
        dsn: PostgreSQL connection string.
        schema_name: Database schema name.

    """

    dsn: str
    schema_name: str = "public"


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS {schema}.checkpoints (
    run_id TEXT NOT NULL,
    checkpoint_id TEXT PRIMARY KEY,
    node_id TEXT,
    created_at TIMESTAMPTZ NOT NULL,
    schema_version INTEGER NOT NULL,
    envelope JSONB NOT NULL,
    content_hash TEXT
);

CREATE INDEX IF NOT EXISTS idx_run_created
    ON {schema}.checkpoints(run_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_run_node_created
    ON {schema}.checkpoints(run_id, node_id, created_at DESC);
"""


class PostgresCheckpointStore(BaseCheckpointStore):
    """Postgres-based checkpoint store with JSONB storage."""

    def __init__(self, config: PostgresCheckpointStoreConfig) -> None:
        """Initialize the Postgres store.

        Args:
            config: Store configuration.

        """
        self.config = config
        self._pool = None
        self._initialized = False

    async def _get_pool(self):
        """Get or create connection pool."""
        if self._pool is None:
            import asyncpg  # noqa: PLC0415

            self._pool = await asyncpg.create_pool(self.config.dsn)
        return self._pool

    async def _ensure_initialized(self) -> None:
        """Ensure database schema is initialized."""
        if self._initialized:
            return

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(_SCHEMA_SQL.format(schema=self.config.schema_name))

        self._initialized = True

    async def _do_save(
        self, envelope: CheckpointEnvelope, overwrite: bool
    ) -> CheckpointEnvelope:
        """Save checkpoint to Postgres.

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

        pool = await self._get_pool()
        async with pool.acquire() as conn, conn.transaction():
            if not overwrite:
                exists = await conn.fetchval(
                    f"SELECT 1 FROM {self.config.schema_name}.checkpoints "
                    f"WHERE checkpoint_id = $1",
                    envelope.id,
                )
                if exists:
                    msg = (
                        f"Checkpoint {envelope.id} already exists "
                        f"for run {envelope.run_id}"
                    )
                    raise CheckpointConflict(msg)

            await conn.execute(
                f"""
                    INSERT INTO {self.config.schema_name}.checkpoints (
                        run_id, checkpoint_id, node_id, created_at,
                        schema_version, envelope, content_hash
                    ) VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7)
                    ON CONFLICT(checkpoint_id) DO UPDATE SET
                        envelope = EXCLUDED.envelope,
                        content_hash = EXCLUDED.content_hash,
                        created_at = EXCLUDED.created_at
                    """,
                envelope.run_id,
                envelope.id,
                envelope.node_id,
                envelope.created_at,
                envelope.schema_version,
                json_str,
                envelope.content_hash,
            )

        return envelope

    async def _do_latest(
        self, run_id: RunId, node_id: str | None = None
    ) -> CheckpointEnvelope | None:
        """Get the most recent checkpoint from Postgres.

        Args:
            run_id: The run to query.
            node_id: Optional node filter.

        Returns:
            The latest checkpoint envelope, or None if not found.

        """
        await self._ensure_initialized()

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            if node_id is not None:
                row = await conn.fetchrow(
                    f"""
                    SELECT envelope::text FROM {self.config.schema_name}.checkpoints
                    WHERE run_id = $1 AND node_id = $2
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    run_id,
                    node_id,
                )
            else:
                row = await conn.fetchrow(
                    f"""
                    SELECT envelope::text FROM {self.config.schema_name}.checkpoints
                    WHERE run_id = $1
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    run_id,
                )

            if row is None:
                return None

            return deserialize_checkpoint(row["envelope"])

    async def _do_get(
        self, run_id: RunId, checkpoint_id: CheckpointId
    ) -> CheckpointEnvelope | None:
        """Get a specific checkpoint from Postgres.

        Args:
            run_id: The run identifier.
            checkpoint_id: The checkpoint identifier.

        Returns:
            The checkpoint envelope, or None if not found.

        """
        await self._ensure_initialized()

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT envelope::text FROM {self.config.schema_name}.checkpoints
                WHERE run_id = $1 AND checkpoint_id = $2
                """,
                run_id,
                checkpoint_id,
            )

            if row is None:
                return None

            return deserialize_checkpoint(row["envelope"])

    async def _do_list(
        self, query: CheckpointQuery
    ) -> tuple[list[CheckpointEnvelope], str | None]:
        """List checkpoints from Postgres.

        Args:
            query: Query parameters for filtering and pagination.

        Returns:
            Tuple of (list of checkpoint envelopes, next cursor or None).

        """
        await self._ensure_initialized()

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            conditions = []
            params = []
            param_idx = 1

            if query.run_id is not None:
                conditions.append(f"run_id = ${param_idx}")
                params.append(query.run_id)
                param_idx += 1

            if query.node_id is not None:
                conditions.append(f"node_id = ${param_idx}")
                params.append(query.node_id)
                param_idx += 1

            if query.since is not None:
                conditions.append(f"created_at >= ${param_idx}")
                params.append(query.since)
                param_idx += 1

            if query.until is not None:
                conditions.append(f"created_at <= ${param_idx}")
                params.append(query.until)
                param_idx += 1

            where_clause = " AND ".join(conditions) if conditions else "TRUE"
            order = "DESC" if query.sort_order == SortOrder.DESC else "ASC"

            cursor_offset = 0
            if query.cursor is not None:
                try:
                    cursor_offset = int(query.cursor)
                except ValueError:
                    cursor_offset = 0

            sql = f"""
                SELECT envelope::text FROM {self.config.schema_name}.checkpoints
                WHERE {where_clause}
                ORDER BY created_at {order}
                LIMIT ${param_idx} OFFSET ${param_idx + 1}
            """
            params.extend([query.limit + 1, cursor_offset])

            rows = await conn.fetch(sql, *params)

            envelopes = [deserialize_checkpoint(row["envelope"]) for row in rows]

            next_cursor = None
            if len(envelopes) > query.limit:
                envelopes = envelopes[: query.limit]
                next_cursor = str(cursor_offset + query.limit)

            return envelopes, next_cursor

    async def _do_delete(self, run_id: RunId, checkpoint_id: CheckpointId) -> bool:
        """Delete a checkpoint from Postgres.

        Args:
            run_id: The run identifier.
            checkpoint_id: The checkpoint identifier.

        Returns:
            True if deleted, False if didn't exist.

        """
        await self._ensure_initialized()

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                f"""
                DELETE FROM {self.config.schema_name}.checkpoints
                WHERE run_id = $1 AND checkpoint_id = $2
                """,
                run_id,
                checkpoint_id,
            )

            return result.split()[-1] != "0"

    async def _do_purge(self, run_id: RunId) -> int:
        """Delete all checkpoints for a run from Postgres.

        Args:
            run_id: The run identifier.

        Returns:
            Number of checkpoints deleted.

        """
        await self._ensure_initialized()

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                f"DELETE FROM {self.config.schema_name}.checkpoints WHERE run_id = $1",
                run_id,
            )

            return int(result.split()[-1])

    async def _do_healthcheck(self) -> bool:
        """Verify Postgres database health.

        Returns:
            True if healthy.

        """
        await self._ensure_initialized()

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return True

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def __repr__(self) -> str:
        """Return a string representation of the store."""
        return (
            f"PostgresCheckpointStore(dsn=<redacted>, schema={self.config.schema_name})"
        )
