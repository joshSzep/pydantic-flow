"""Postgres checkpoint store implementation.

Stores checkpoints in PostgreSQL with JSONB for efficient querying.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from pydantic_flow.hitl.checkpoints.interface import CheckpointBackendError
from pydantic_flow.hitl.checkpoints.interface import CheckpointConflict
from pydantic_flow.hitl.checkpoints.interface import CheckpointEnvelope
from pydantic_flow.hitl.checkpoints.interface import CheckpointId
from pydantic_flow.hitl.checkpoints.interface import CheckpointQuery
from pydantic_flow.hitl.checkpoints.interface import RunId
from pydantic_flow.hitl.checkpoints.interface import SortOrder
from pydantic_flow.hitl.checkpoints.serde import compute_content_hash
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


class PostgresCheckpointStore:
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

    async def save(
        self, envelope: CheckpointEnvelope, *, overwrite: bool = False
    ) -> CheckpointEnvelope:
        """Save a checkpoint to Postgres.

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

            pool = await self._get_pool()
            async with pool.acquire() as conn, conn.transaction():
                if not overwrite:
                    exists = await conn.fetchval(
                        f"SELECT 1 FROM {self.config.schema_name}.checkpoints "
                        f"WHERE checkpoint_id = $1",
                        envelope_copy.id,
                    )
                    if exists:
                        msg = (
                            f"Checkpoint {envelope_copy.id} already exists "
                            f"for run {envelope_copy.run_id}"
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
                    envelope_copy.run_id,
                    envelope_copy.id,
                    envelope_copy.node_id,
                    envelope_copy.created_at,
                    envelope_copy.schema_version,
                    json_str,
                    envelope_copy.content_hash,
                )

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
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                result = await conn.execute(
                    f"DELETE FROM {self.config.schema_name}.checkpoints "
                    "WHERE run_id = $1",
                    run_id,
                )

                return int(result.split()[-1])

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

            pool = await self._get_pool()
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True

        except Exception as e:
            msg = f"Healthcheck failed: {e}"
            raise CheckpointBackendError(msg, cause=e) from e

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
