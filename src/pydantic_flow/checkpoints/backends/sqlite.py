"""SQLite storage backend for checkpoint v2.

This module provides a SQLite-based implementation of the checkpoint storage
backend, optimized for local development and single-process applications.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Literal
import uuid

import aiosqlite
from pydantic import BaseModel
from pydantic import Field

from pydantic_flow.checkpoints.types import ExecutionTrace
from pydantic_flow.checkpoints.types import NodeExecutionTrace
from pydantic_flow.checkpoints.types import RunId
from pydantic_flow.checkpoints.types import RunMetadata
from pydantic_flow.checkpoints.types import StateSnapshot


class SQLiteCheckpointConfig(BaseModel):
    """Configuration for SQLite checkpoint backend.

    Attributes:
        db_path: Path to SQLite database file.
        create_tables: Whether to create schema on initialization.
        wal_mode: Enable WAL mode for better concurrency.
        timeout: Database timeout in seconds.
        compress_level: Compression level for state data (1-9).

    """

    db_path: Path
    create_tables: bool = True
    wal_mode: bool = True
    timeout: float = 5.0
    compress_level: int = Field(default=6, ge=1, le=9)


class SQLiteCheckpointBackend:
    """SQLite backend for checkpoint storage.

    Optimized for local development and single-process applications. Uses
    Write-Ahead Logging (WAL) mode for improved read concurrency.

    Concurrency Model:
        - Unlimited concurrent reads
        - Single writer (queued via SQLite)
        - WAL mode allows reads during writes

    Limitations:
        - Single-process only (no multi-process support)
        - Single writer can be bottleneck under high load

    Args:
        config: SQLite backend configuration.

    """

    def __init__(self, config: SQLiteCheckpointConfig):
        """Initialize SQLite backend.

        Args:
            config: Backend configuration.

        """
        self.config = config
        self.db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Initialize database connection and schema."""
        self.config.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.db = await aiosqlite.connect(
            str(self.config.db_path), timeout=self.config.timeout
        )
        self.db.row_factory = aiosqlite.Row

        if self.config.wal_mode:
            await self.db.execute("PRAGMA journal_mode=WAL")
            await self.db.execute("PRAGMA busy_timeout=5000")

        if self.config.create_tables:
            await self._create_schema()

    async def close(self) -> None:
        """Close database connection."""
        if self.db:
            await self.db.close()
            self.db = None

    async def healthcheck(self) -> bool:
        """Check database health.

        Returns:
            True if database is accessible and operational.

        """
        if not self.db:
            return False

        try:
            cursor = await self.db.execute("SELECT 1")
            await cursor.fetchone()
            return True
        except Exception:
            return False

    async def _create_schema(self) -> None:
        """Create database schema with indexes."""
        if not self.db:
            msg = "Database not initialized"
            raise RuntimeError(msg)

        schema = """
        -- State snapshots (Track 1: Resume capability)
        CREATE TABLE IF NOT EXISTS state_snapshots (
            snapshot_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            wave_number INTEGER NOT NULL,
            data_compressed BLOB NOT NULL,
            state_hash TEXT NOT NULL,
            trace_id TEXT,
            created_at TIMESTAMP NOT NULL,
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
            data_compressed BLOB NOT NULL,
            created_at TIMESTAMP NOT NULL,
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
            data_compressed BLOB NOT NULL,
            created_at TIMESTAMP NOT NULL,
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
            created_at TIMESTAMP NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_runs_started
            ON run_metadata(started_at DESC);
        CREATE INDEX IF NOT EXISTS idx_runs_status
            ON run_metadata(status);
        """

        await self.db.executescript(schema)
        await self.db.commit()

    async def save_run_metadata(self, metadata: RunMetadata) -> None:
        """Save run metadata."""
        if not self.db:
            msg = "Database not initialized"
            raise RuntimeError(msg)

        error_json = metadata.error.model_dump_json() if metadata.error else None

        await self.db.execute(
            """
            INSERT INTO run_metadata
                (run_id, flow_id, started_at, completed_at, status,
                 total_waves, error_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(run_id) DO UPDATE SET
                completed_at = excluded.completed_at,
                status = excluded.status,
                total_waves = excluded.total_waves,
                error_json = excluded.error_json
            """,
            (
                metadata.run_id,
                metadata.flow_id,
                metadata.started_at.isoformat(),
                metadata.completed_at.isoformat() if metadata.completed_at else None,
                metadata.status.value,
                metadata.total_waves,
                error_json,
            ),
        )
        await self.db.commit()

    async def get_run_metadata(self, run_id: RunId) -> RunMetadata | None:
        """Retrieve run metadata."""
        if not self.db:
            msg = "Database not initialized"
            raise RuntimeError(msg)

        cursor = await self.db.execute(
            "SELECT * FROM run_metadata WHERE run_id = ?", (run_id,)
        )
        row = await cursor.fetchone()

        if not row:
            return None

        return RunMetadata.model_validate(dict(row))

    async def save_state_snapshot(self, snapshot: StateSnapshot) -> None:
        """Save state snapshot."""
        if not self.db:
            msg = "Database not initialized"
            raise RuntimeError(msg)

        data_compressed = snapshot.serialize()

        await self.db.execute(
            """
            INSERT INTO state_snapshots
                (snapshot_id, run_id, wave_number, data_compressed,
                 state_hash, trace_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(snapshot_id) DO UPDATE SET
                trace_id = excluded.trace_id
            """,
            (
                snapshot.snapshot_id,
                snapshot.run_id,
                snapshot.wave_number,
                data_compressed,
                snapshot.state_hash,
                snapshot.trace_id,
                snapshot.created_at.isoformat(),
            ),
        )
        await self.db.commit()

    async def get_state_snapshot(
        self, run_id: RunId, wave_number: int
    ) -> StateSnapshot | None:
        """Retrieve state snapshot."""
        if not self.db:
            msg = "Database not initialized"
            raise RuntimeError(msg)

        cursor = await self.db.execute(
            """
            SELECT data_compressed
            FROM state_snapshots
            WHERE run_id = ? AND wave_number = ?
            """,
            (run_id, wave_number),
        )
        row = await cursor.fetchone()

        if not row:
            return None

        return StateSnapshot.deserialize(row["data_compressed"])

    async def update_state_snapshot(self, snapshot: StateSnapshot) -> None:
        """Update existing state snapshot."""
        if not self.db:
            msg = "Database not initialized"
            raise RuntimeError(msg)

        data_compressed = snapshot.serialize()

        await self.db.execute(
            """
            UPDATE state_snapshots
            SET data_compressed = ?, trace_id = ?
            WHERE snapshot_id = ?
            """,
            (data_compressed, snapshot.trace_id, snapshot.snapshot_id),
        )
        await self.db.commit()

    async def get_snapshots_range(
        self,
        run_id: RunId,
        start_wave: int,
        end_wave: int,
        order: Literal["ASC", "DESC"] = "ASC",
    ) -> list[StateSnapshot]:
        """Retrieve range of snapshots for state reconstruction."""
        if not self.db:
            msg = "Database not initialized"
            raise RuntimeError(msg)

        if order not in ("ASC", "DESC"):
            msg = f"Invalid order: {order}"
            raise ValueError(msg)

        cursor = await self.db.execute(
            f"""
            SELECT data_compressed
            FROM state_snapshots
            WHERE run_id = ? AND wave_number >= ? AND wave_number <= ?
            ORDER BY wave_number {order}
            """,
            (run_id, start_wave, end_wave),
        )
        rows = await cursor.fetchall()

        return [StateSnapshot.deserialize(row["data_compressed"]) for row in rows]

    async def save_trace(self, trace: ExecutionTrace) -> None:
        """Save execution trace with checkpoint validation."""
        if not self.db:
            msg = "Database not initialized"
            raise RuntimeError(msg)

        # Validate checkpoint reference exists
        cursor = await self.db.execute(
            "SELECT 1 FROM state_snapshots WHERE snapshot_id = ?",
            (trace.checkpoint_snapshot_id,),
        )
        if not await cursor.fetchone():
            msg = f"Invalid checkpoint reference: {trace.checkpoint_snapshot_id}"
            raise ValueError(msg)

        # Serialize trace (includes all node traces)
        from pydantic_flow.checkpoints.serialization import TypedSerializer
        from pydantic_flow.checkpoints.serialization import compress

        data = TypedSerializer.serialize(trace)
        data_compressed = compress(data, level=self.config.compress_level)

        await self.db.execute(
            """
            INSERT INTO execution_traces
                (trace_id, run_id, wave_number, checkpoint_snapshot_id,
                 data_compressed, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                trace.trace_id,
                trace.run_id,
                trace.wave_number,
                trace.checkpoint_snapshot_id,
                data_compressed,
                trace.started_at.isoformat(),
            ),
        )

        # Save individual node traces
        for node_trace in trace.node_traces:
            node_data = TypedSerializer.serialize(node_trace)
            node_compressed = compress(node_data, level=self.config.compress_level)

            await self.db.execute(
                """
                INSERT INTO node_traces
                    (log_id, trace_id, node_id, wave_number,
                     data_compressed, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    node_trace.log_id,
                    trace.trace_id,
                    node_trace.node_id,
                    node_trace.wave_number,
                    node_compressed,
                    node_trace.started_at.isoformat(),
                ),
            )

        await self.db.commit()

    async def get_trace(self, run_id: RunId, wave_number: int) -> ExecutionTrace | None:
        """Retrieve execution trace."""
        if not self.db:
            msg = "Database not initialized"
            raise RuntimeError(msg)

        cursor = await self.db.execute(
            """
            SELECT data_compressed
            FROM execution_traces
            WHERE run_id = ? AND wave_number = ?
            """,
            (run_id, wave_number),
        )
        row = await cursor.fetchone()

        if not row:
            return None

        from pydantic_flow.checkpoints.serialization import TypedSerializer
        from pydantic_flow.checkpoints.serialization import decompress

        decompressed = decompress(row["data_compressed"])
        return TypedSerializer.deserialize(decompressed)

    async def delete_trace(self, run_id: RunId, wave_number: int) -> bool:
        """Delete execution trace."""
        if not self.db:
            msg = "Database not initialized"
            raise RuntimeError(msg)

        cursor = await self.db.execute(
            "DELETE FROM execution_traces WHERE run_id = ? AND wave_number = ?",
            (run_id, wave_number),
        )
        await self.db.commit()

        return cursor.rowcount > 0

    async def save_node_trace(self, node_trace: NodeExecutionTrace) -> None:
        """Save node execution trace."""
        if not self.db:
            msg = "Database not initialized"
            raise RuntimeError(msg)

        from pydantic_flow.checkpoints.serialization import TypedSerializer
        from pydantic_flow.checkpoints.serialization import compress

        data = TypedSerializer.serialize(node_trace)
        data_compressed = compress(data, level=self.config.compress_level)

        trace_id = str(uuid.uuid4())

        await self.db.execute(
            """
            INSERT INTO node_traces
                (log_id, trace_id, node_id, wave_number,
                 data_compressed, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                node_trace.log_id,
                trace_id,
                node_trace.node_id,
                node_trace.wave_number,
                data_compressed,
                node_trace.started_at.isoformat(),
            ),
        )
        await self.db.commit()

    async def get_node_trace(self, log_id: str) -> NodeExecutionTrace | None:
        """Retrieve node execution trace."""
        if not self.db:
            msg = "Database not initialized"
            raise RuntimeError(msg)

        cursor = await self.db.execute(
            "SELECT data_compressed FROM node_traces WHERE log_id = ?",
            (log_id,),
        )
        row = await cursor.fetchone()

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
        if not self.db:
            msg = "Database not initialized"
            raise RuntimeError(msg)

        query = "SELECT * FROM run_metadata WHERE 1=1"
        params: list[str | int] = []

        if before:
            query += " AND started_at < ?"
            params.append(before.isoformat())

        if after:
            query += " AND started_at > ?"
            params.append(after.isoformat())

        query += " ORDER BY started_at DESC"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor = await self.db.execute(query, params)
        rows = await cursor.fetchall()

        return [RunMetadata.model_validate(dict(row)) for row in rows]

    async def delete_run(
        self, run_id: RunId, *, keep_checkpoints: bool = False
    ) -> None:
        """Delete all data for a run."""
        if not self.db:
            msg = "Database not initialized"
            raise RuntimeError(msg)

        # Delete traces (cascades to node traces)
        await self.db.execute(
            "DELETE FROM execution_traces WHERE run_id = ?", (run_id,)
        )

        # Delete metadata
        await self.db.execute("DELETE FROM run_metadata WHERE run_id = ?", (run_id,))

        # Optionally delete checkpoints
        if not keep_checkpoints:
            await self.db.execute(
                "DELETE FROM state_snapshots WHERE run_id = ?", (run_id,)
            )

        await self.db.commit()

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
        # Placeholder - will be implemented when event streaming is needed
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
        # Placeholder - will be implemented when event streaming is needed
        return []
