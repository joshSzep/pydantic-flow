"""PostgreSQL storage backend for checkpoint v2.

This module provides a PostgreSQL-based implementation of the checkpoint
storage backend, optimized for production use with MVCC concurrency control
and connection pooling.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from typing import Literal
import uuid

from pydantic import BaseModel
from pydantic import Field

from pydantic_flow.checkpoints.types import ExecutionTrace
from pydantic_flow.checkpoints.types import NodeExecutionTrace
from pydantic_flow.checkpoints.types import RunId
from pydantic_flow.checkpoints.types import RunMetadata
from pydantic_flow.checkpoints.types import StateSnapshot


class PostgresCheckpointConfig(BaseModel):
    """Configuration for PostgreSQL checkpoint backend.

    Attributes:
        connection_string: PostgreSQL connection URL.
        min_pool_size: Minimum connection pool size.
        max_pool_size: Maximum connection pool size.
        create_schema: Whether to create schema on initialization.
        timeout: Query timeout in seconds.
        compress_level: Compression level for state data (1-9).

    """

    connection_string: str
    min_pool_size: int = Field(default=2, ge=1)
    max_pool_size: int = Field(default=10, ge=1)
    create_schema: bool = True
    timeout: float = 10.0
    compress_level: int = Field(default=6, ge=1, le=9)


class PostgresCheckpointBackend:
    """PostgreSQL backend for checkpoint storage.

    Optimized for production use with multi-process concurrency support via
    MVCC (Multi-Version Concurrency Control). Provides connection pooling
    and prepared statements for optimal performance.

    Concurrency Model:
        - Multiple concurrent readers without blocking
        - Multiple concurrent writers via MVCC
        - Row-level locking for updates
        - No lock contention on reads

    Production Features:
        - Connection pooling for efficient resource usage
        - Prepared statements for query optimization
        - Foreign key constraints for data integrity
        - Cascading deletes for cleanup
        - Indexes for query performance

    Args:
        config: PostgreSQL backend configuration.

    """

    def __init__(self, config: PostgresCheckpointConfig):
        """Initialize PostgreSQL backend.

        Args:
            config: Backend configuration.

        """
        self.config = config
        self.pool: Any | None = None

    async def initialize(self) -> None:
        """Initialize connection pool and schema."""
        try:
            import asyncpg
        except ImportError as e:
            msg = (
                "asyncpg is required for PostgreSQL backend. "
                "Install with: pip install asyncpg"
            )
            raise ImportError(msg) from e

        self.pool = await asyncpg.create_pool(
            self.config.connection_string,
            min_size=self.config.min_pool_size,
            max_size=self.config.max_pool_size,
            command_timeout=self.config.timeout,
        )

        if self.config.create_schema:
            await self._create_schema()

    async def close(self) -> None:
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None

    async def healthcheck(self) -> bool:
        """Check database health.

        Returns:
            True if database is accessible and operational.

        """
        if not self.pool:
            return False

        try:
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception:
            return False

    async def _create_schema(self) -> None:
        """Create database schema with indexes and constraints."""
        if not self.pool:
            msg = "Database not initialized"
            raise RuntimeError(msg)

        schema = """
        -- State snapshots (Track 1: Resume capability)
        CREATE TABLE IF NOT EXISTS state_snapshots (
            snapshot_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            wave_number INTEGER NOT NULL,
            data_compressed BYTEA NOT NULL,
            state_hash TEXT NOT NULL,
            trace_id TEXT,
            created_at TIMESTAMP NOT NULL DEFAULT NOW(),
            UNIQUE(run_id, wave_number)
        );
        CREATE INDEX IF NOT EXISTS idx_snapshots_run_wave
            ON state_snapshots(run_id, wave_number);
        CREATE INDEX IF NOT EXISTS idx_snapshots_created
            ON state_snapshots(created_at);

        -- Execution traces (Track 2: Debugging)
        CREATE TABLE IF NOT EXISTS execution_traces (
            trace_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            wave_number INTEGER NOT NULL,
            checkpoint_snapshot_id TEXT NOT NULL,
            data_compressed BYTEA NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT NOW(),
            FOREIGN KEY (checkpoint_snapshot_id)
                REFERENCES state_snapshots(snapshot_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_traces_run
            ON execution_traces(run_id, wave_number);
        CREATE INDEX IF NOT EXISTS idx_traces_created
            ON execution_traces(created_at);
        CREATE INDEX IF NOT EXISTS idx_traces_checkpoint
            ON execution_traces(checkpoint_snapshot_id);

        -- Node execution traces
        CREATE TABLE IF NOT EXISTS node_traces (
            log_id TEXT PRIMARY KEY,
            trace_id TEXT NOT NULL,
            node_id TEXT NOT NULL,
            wave_number INTEGER NOT NULL,
            data_compressed BYTEA NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT NOW(),
            FOREIGN KEY (trace_id)
                REFERENCES execution_traces(trace_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_node_traces_trace
            ON node_traces(trace_id);

        -- Run metadata
        CREATE TABLE IF NOT EXISTS run_metadata (
            run_id TEXT PRIMARY KEY,
            flow_id TEXT NOT NULL,
            started_at TIMESTAMP NOT NULL,
            completed_at TIMESTAMP,
            status TEXT NOT NULL,
            total_waves INTEGER NOT NULL,
            error_json TEXT,
            created_at TIMESTAMP NOT NULL DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_runs_started
            ON run_metadata(started_at DESC);
        CREATE INDEX IF NOT EXISTS idx_runs_status
            ON run_metadata(status);
        """

        async with self.pool.acquire() as conn:
            await conn.execute(schema)

    async def save_run_metadata(self, metadata: RunMetadata) -> None:
        """Save run metadata."""
        if not self.pool:
            msg = "Database not initialized"
            raise RuntimeError(msg)

        error_json = metadata.error.model_dump_json() if metadata.error else None

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO run_metadata
                    (run_id, flow_id, started_at, completed_at, status,
                     total_waves, error_json)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT(run_id) DO UPDATE SET
                    completed_at = EXCLUDED.completed_at,
                    status = EXCLUDED.status,
                    total_waves = EXCLUDED.total_waves,
                    error_json = EXCLUDED.error_json
                """,
                metadata.run_id,
                metadata.flow_id,
                metadata.started_at,
                metadata.completed_at,
                metadata.status.value,
                metadata.total_waves,
                error_json,
            )

    async def get_run_metadata(self, run_id: RunId) -> RunMetadata | None:
        """Retrieve run metadata."""
        if not self.pool:
            msg = "Database not initialized"
            raise RuntimeError(msg)

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM run_metadata WHERE run_id = $1", run_id
            )

        if not row:
            return None

        return RunMetadata.model_validate(dict(row))

    async def save_state_snapshot(self, snapshot: StateSnapshot) -> None:
        """Save state snapshot."""
        if not self.pool:
            msg = "Database not initialized"
            raise RuntimeError(msg)

        data_compressed = snapshot.serialize()

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO state_snapshots
                    (snapshot_id, run_id, wave_number, data_compressed,
                     state_hash, trace_id, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT(snapshot_id) DO UPDATE SET
                    trace_id = EXCLUDED.trace_id
                """,
                snapshot.snapshot_id,
                snapshot.run_id,
                snapshot.wave_number,
                data_compressed,
                snapshot.state_hash,
                snapshot.trace_id,
                snapshot.created_at,
            )

    async def get_state_snapshot(
        self, run_id: RunId, wave_number: int
    ) -> StateSnapshot | None:
        """Retrieve state snapshot."""
        if not self.pool:
            msg = "Database not initialized"
            raise RuntimeError(msg)

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT data_compressed
                FROM state_snapshots
                WHERE run_id = $1 AND wave_number = $2
                """,
                run_id,
                wave_number,
            )

        if not row:
            return None

        return StateSnapshot.deserialize(row["data_compressed"])

    async def update_state_snapshot(self, snapshot: StateSnapshot) -> None:
        """Update existing state snapshot."""
        if not self.pool:
            msg = "Database not initialized"
            raise RuntimeError(msg)

        data_compressed = snapshot.serialize()

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE state_snapshots
                SET data_compressed = $1, trace_id = $2
                WHERE snapshot_id = $3
                """,
                data_compressed,
                snapshot.trace_id,
                snapshot.snapshot_id,
            )

    async def get_snapshots_range(
        self,
        run_id: RunId,
        start_wave: int,
        end_wave: int,
        order: Literal["ASC", "DESC"] = "ASC",
    ) -> list[StateSnapshot]:
        """Retrieve range of snapshots for state reconstruction."""
        if not self.pool:
            msg = "Database not initialized"
            raise RuntimeError(msg)

        if order not in ("ASC", "DESC"):
            msg = f"Invalid order: {order}"
            raise ValueError(msg)

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT data_compressed
                FROM state_snapshots
                WHERE run_id = $1 AND wave_number >= $2 AND wave_number <= $3
                ORDER BY wave_number {order}
                """,
                run_id,
                start_wave,
                end_wave,
            )

        return [StateSnapshot.deserialize(row["data_compressed"]) for row in rows]

    async def save_trace(self, trace: ExecutionTrace) -> None:
        """Save execution trace with checkpoint validation."""
        if not self.pool:
            msg = "Database not initialized"
            raise RuntimeError(msg)

        from pydantic_flow.checkpoints.serialization import TypedSerializer
        from pydantic_flow.checkpoints.serialization import compress

        async with self.pool.acquire() as conn:
            # Validate checkpoint reference exists
            exists = await conn.fetchval(
                "SELECT 1 FROM state_snapshots WHERE snapshot_id = $1",
                trace.checkpoint_snapshot_id,
            )
            if not exists:
                msg = f"Invalid checkpoint reference: {trace.checkpoint_snapshot_id}"
                raise ValueError(msg)

            # Serialize trace (includes all node traces)
            data = TypedSerializer.serialize(trace)
            data_compressed = compress(data, level=self.config.compress_level)

            await conn.execute(
                """
                INSERT INTO execution_traces
                    (trace_id, run_id, wave_number, checkpoint_snapshot_id,
                     data_compressed, created_at)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                trace.trace_id,
                trace.run_id,
                trace.wave_number,
                trace.checkpoint_snapshot_id,
                data_compressed,
                trace.started_at,
            )

            # Save individual node traces
            for node_trace in trace.node_traces:
                node_data = TypedSerializer.serialize(node_trace)
                node_compressed = compress(node_data, level=self.config.compress_level)

                await conn.execute(
                    """
                    INSERT INTO node_traces
                        (log_id, trace_id, node_id, wave_number,
                         data_compressed, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    node_trace.log_id,
                    trace.trace_id,
                    node_trace.node_id,
                    node_trace.wave_number,
                    node_compressed,
                    node_trace.started_at,
                )

    async def get_trace(self, run_id: RunId, wave_number: int) -> ExecutionTrace | None:
        """Retrieve execution trace."""
        if not self.pool:
            msg = "Database not initialized"
            raise RuntimeError(msg)

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT data_compressed
                FROM execution_traces
                WHERE run_id = $1 AND wave_number = $2
                """,
                run_id,
                wave_number,
            )

        if not row:
            return None

        from pydantic_flow.checkpoints.serialization import TypedSerializer
        from pydantic_flow.checkpoints.serialization import decompress

        decompressed = decompress(row["data_compressed"])
        return TypedSerializer.deserialize(decompressed)

    async def delete_trace(self, run_id: RunId, wave_number: int) -> bool:
        """Delete execution trace."""
        if not self.pool:
            msg = "Database not initialized"
            raise RuntimeError(msg)

        async with self.pool.acquire() as conn:
            result = await conn.execute(
                """
                DELETE FROM execution_traces
                WHERE run_id = $1 AND wave_number = $2
                """,
                run_id,
                wave_number,
            )

        return result != "DELETE 0"

    async def save_node_trace(self, node_trace: NodeExecutionTrace) -> None:
        """Save node execution trace."""
        if not self.pool:
            msg = "Database not initialized"
            raise RuntimeError(msg)

        from pydantic_flow.checkpoints.serialization import TypedSerializer
        from pydantic_flow.checkpoints.serialization import compress

        data = TypedSerializer.serialize(node_trace)
        data_compressed = compress(data, level=self.config.compress_level)

        trace_id = str(uuid.uuid4())

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO node_traces
                    (log_id, trace_id, node_id, wave_number,
                     data_compressed, created_at)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                node_trace.log_id,
                trace_id,
                node_trace.node_id,
                node_trace.wave_number,
                data_compressed,
                node_trace.started_at,
            )

    async def get_node_trace(self, log_id: str) -> NodeExecutionTrace | None:
        """Retrieve node execution trace."""
        if not self.pool:
            msg = "Database not initialized"
            raise RuntimeError(msg)

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data_compressed FROM node_traces WHERE log_id = $1",
                log_id,
            )

        if not row:
            return None

        from pydantic_flow.checkpoints.serialization import TypedSerializer
        from pydantic_flow.checkpoints.serialization import decompress

        decompressed = decompress(row["data_compressed"])
        return TypedSerializer.deserialize(decompressed)

    async def list_runs(
        self,
        *,
        before: datetime | None = None,
        after: datetime | None = None,
        limit: int | None = None,
    ) -> list[RunMetadata]:
        """List runs with optional filtering."""
        if not self.pool:
            msg = "Database not initialized"
            raise RuntimeError(msg)

        query = "SELECT * FROM run_metadata WHERE 1=1"
        params: list[datetime | int] = []
        param_idx = 1

        if before:
            query += f" AND started_at < ${param_idx}"
            params.append(before)
            param_idx += 1

        if after:
            query += f" AND started_at > ${param_idx}"
            params.append(after)
            param_idx += 1

        query += " ORDER BY started_at DESC"

        if limit:
            query += f" LIMIT ${param_idx}"
            params.append(limit)

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        return [RunMetadata.model_validate(dict(row)) for row in rows]

    async def delete_run(
        self, run_id: RunId, *, keep_checkpoints: bool = False
    ) -> None:
        """Delete all data for a run."""
        if not self.pool:
            msg = "Database not initialized"
            raise RuntimeError(msg)

        async with self.pool.acquire() as conn:
            # Delete traces (cascades to node traces)
            await conn.execute("DELETE FROM execution_traces WHERE run_id = $1", run_id)

            # Delete metadata
            await conn.execute("DELETE FROM run_metadata WHERE run_id = $1", run_id)

            # Optionally delete checkpoints
            if not keep_checkpoints:
                await conn.execute(
                    "DELETE FROM state_snapshots WHERE run_id = $1", run_id
                )

    async def append_events_batch(
        self,
        log_id: str,
        events: list[Any],
        start_sequence: int,
    ) -> None:
        """Append batch of events to event log.

        Note: This is a placeholder implementation for Phase 3.
        Full event storage will be implemented when needed.

        Args:
            log_id: Event log identifier.
            events: List of progress items to append.
            start_sequence: Starting sequence number for this batch.

        """
        pass

    async def stream_events(
        self,
        log_id: str,
        start_offset: int = 0,
        end_offset: int | None = None,
    ) -> list[Any]:
        """Stream events from event log.

        Note: This is a placeholder implementation for Phase 3.
        Full event retrieval will be implemented when needed.

        Args:
            log_id: Event log identifier.
            start_offset: Starting offset for event stream.
            end_offset: Optional ending offset.

        Returns:
            Empty list (placeholder).

        """
        return []
