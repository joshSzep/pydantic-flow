"""Composable checkpoint backends for v2.

This module provides composable backend patterns that enable:
- MultiConsumerStorage: Write to primary + replica backends
- TieredStorage: Hot/cold storage with automatic fallback
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from typing import Literal

from pydantic import BaseModel

from pydantic_flow.checkpoints.types import ExecutionTrace
from pydantic_flow.checkpoints.types import NodeExecutionTrace
from pydantic_flow.checkpoints.types import RunId
from pydantic_flow.checkpoints.types import RunMetadata
from pydantic_flow.checkpoints.types import StateSnapshot


class MultiConsumerConfig(BaseModel):
    """Configuration for multi-consumer storage.

    Attributes:
        primary: Primary backend (required for all operations).
        replicas: List of replica backends (write-only, best-effort).
        fail_on_replica_error: If True, fail on replica write errors.

    """

    fail_on_replica_error: bool = False


class MultiConsumerStorage:
    """Multi-consumer checkpoint storage.

    Writes to primary backend and replicates to multiple replicas.
    Reads only from primary. Useful for backup and replication.

    Pattern:
        primary = SQLiteCheckpointBackend(...)
        replica1 = S3CheckpointBackend(...)
        replica2 = PostgresCheckpointBackend(...)

        storage = MultiConsumerStorage(
            primary=primary,
            replicas=[replica1, replica2],
            config=MultiConsumerConfig(fail_on_replica_error=False)
        )

    Args:
        primary: Primary backend for all operations.
        replicas: Replica backends for replication (best-effort).
        config: Multi-consumer configuration.

    """

    def __init__(
        self,
        primary: Any,
        replicas: list[Any],
        config: MultiConsumerConfig | None = None,
    ):
        """Initialize multi-consumer storage.

        Args:
            primary: Primary backend.
            replicas: List of replica backends.
            config: Optional configuration.

        """
        self.primary = primary
        self.replicas = replicas
        self.config = config or MultiConsumerConfig()

    async def initialize(self) -> None:
        """Initialize all backends."""
        await self.primary.initialize()
        for replica in self.replicas:
            try:
                await replica.initialize()
            except Exception:
                if self.config.fail_on_replica_error:
                    raise

    async def close(self) -> None:
        """Close all backends."""
        import contextlib

        await self.primary.close()
        for replica in self.replicas:
            with contextlib.suppress(Exception):
                await replica.close()

    async def healthcheck(self) -> bool:
        """Check primary backend health."""
        return await self.primary.healthcheck()

    async def save_run_metadata(self, metadata: RunMetadata) -> None:
        """Save run metadata to primary and replicas."""
        await self.primary.save_run_metadata(metadata)

        for replica in self.replicas:
            try:
                await replica.save_run_metadata(metadata)
            except Exception:
                if self.config.fail_on_replica_error:
                    raise

    async def get_run_metadata(self, run_id: RunId) -> RunMetadata | None:
        """Retrieve run metadata from primary."""
        return await self.primary.get_run_metadata(run_id)

    async def save_state_snapshot(self, snapshot: StateSnapshot) -> None:
        """Save state snapshot to primary and replicas."""
        await self.primary.save_state_snapshot(snapshot)

        for replica in self.replicas:
            try:
                await replica.save_state_snapshot(snapshot)
            except Exception:
                if self.config.fail_on_replica_error:
                    raise

    async def get_state_snapshot(
        self, run_id: RunId, wave_number: int
    ) -> StateSnapshot | None:
        """Retrieve state snapshot from primary."""
        return await self.primary.get_state_snapshot(run_id, wave_number)

    async def update_state_snapshot(self, snapshot: StateSnapshot) -> None:
        """Update state snapshot in primary and replicas."""
        await self.primary.update_state_snapshot(snapshot)

        for replica in self.replicas:
            try:
                await replica.update_state_snapshot(snapshot)
            except Exception:
                if self.config.fail_on_replica_error:
                    raise

    async def get_snapshots_range(
        self,
        run_id: RunId,
        start_wave: int,
        end_wave: int,
        order: Literal["ASC", "DESC"] = "ASC",
    ) -> list[StateSnapshot]:
        """Retrieve range of snapshots from primary."""
        return await self.primary.get_snapshots_range(
            run_id, start_wave, end_wave, order
        )

    async def save_trace(self, trace: ExecutionTrace) -> None:
        """Save execution trace to primary and replicas."""
        await self.primary.save_trace(trace)

        for replica in self.replicas:
            try:
                await replica.save_trace(trace)
            except Exception:
                if self.config.fail_on_replica_error:
                    raise

    async def get_trace(self, run_id: RunId, wave_number: int) -> ExecutionTrace | None:
        """Retrieve execution trace from primary."""
        return await self.primary.get_trace(run_id, wave_number)

    async def delete_trace(self, run_id: RunId, wave_number: int) -> bool:
        """Delete execution trace from primary and replicas."""
        result = await self.primary.delete_trace(run_id, wave_number)

        for replica in self.replicas:
            try:
                await replica.delete_trace(run_id, wave_number)
            except Exception:
                if self.config.fail_on_replica_error:
                    raise

        return result

    async def save_node_trace(self, node_trace: NodeExecutionTrace) -> None:
        """Save node execution trace to primary and replicas."""
        await self.primary.save_node_trace(node_trace)

        for replica in self.replicas:
            try:
                await replica.save_node_trace(node_trace)
            except Exception:
                if self.config.fail_on_replica_error:
                    raise

    async def get_node_trace(self, log_id: str) -> NodeExecutionTrace | None:
        """Retrieve node execution trace from primary."""
        return await self.primary.get_node_trace(log_id)

    async def list_runs(
        self,
        *,
        before: datetime | None = None,
        after: datetime | None = None,
        limit: int | None = None,
    ) -> list[RunMetadata]:
        """List runs from primary."""
        return await self.primary.list_runs(before=before, after=after, limit=limit)

    async def delete_run(
        self, run_id: RunId, *, keep_checkpoints: bool = False
    ) -> None:
        """Delete run from primary and replicas."""
        await self.primary.delete_run(run_id, keep_checkpoints=keep_checkpoints)

        for replica in self.replicas:
            try:
                await replica.delete_run(run_id, keep_checkpoints=keep_checkpoints)
            except Exception:
                if self.config.fail_on_replica_error:
                    raise

    async def append_events_batch(
        self,
        log_id: str,
        events: list[Any],
        start_sequence: int,
    ) -> None:
        """Append events batch to primary and replicas."""
        await self.primary.append_events_batch(log_id, events, start_sequence)

        for replica in self.replicas:
            try:
                await replica.append_events_batch(log_id, events, start_sequence)
            except Exception:
                if self.config.fail_on_replica_error:
                    raise

    async def stream_events(
        self,
        log_id: str,
        start_offset: int = 0,
        end_offset: int | None = None,
    ) -> list[Any]:
        """Stream events from primary."""
        return await self.primary.stream_events(log_id, start_offset, end_offset)


class TieredStorageConfig(BaseModel):
    """Configuration for tiered storage.

    Attributes:
        prefer_hot: Prefer hot storage for writes when both available.
        fallback_on_cold_miss: Fallback to cold storage on hot miss.

    """

    prefer_hot: bool = True
    fallback_on_cold_miss: bool = True


class TieredStorage:
    """Tiered checkpoint storage with hot and cold backends.

    Reads from hot storage first, falls back to cold storage.
    Writes can go to hot, cold, or both depending on configuration.

    Pattern:
        hot = PostgresCheckpointBackend(...)  # Fast, expensive
        cold = S3CheckpointBackend(...)       # Slow, cheap

        storage = TieredStorage(
            hot=hot,
            cold=cold,
            config=TieredStorageConfig(prefer_hot=True)
        )

    Args:
        hot: Hot storage backend (fast, expensive).
        cold: Cold storage backend (slow, cheap).
        config: Tiered storage configuration.

    """

    def __init__(
        self,
        hot: Any,
        cold: Any,
        config: TieredStorageConfig | None = None,
    ):
        """Initialize tiered storage.

        Args:
            hot: Hot storage backend.
            cold: Cold storage backend.
            config: Optional configuration.

        """
        self.hot = hot
        self.cold = cold
        self.config = config or TieredStorageConfig()

    async def initialize(self) -> None:
        """Initialize both backends."""
        await self.hot.initialize()
        await self.cold.initialize()

    async def close(self) -> None:
        """Close both backends."""
        await self.hot.close()
        await self.cold.close()

    async def healthcheck(self) -> bool:
        """Check hot backend health."""
        return await self.hot.healthcheck()

    async def save_run_metadata(self, metadata: RunMetadata) -> None:
        """Save run metadata to hot storage."""
        await self.hot.save_run_metadata(metadata)

    async def get_run_metadata(self, run_id: RunId) -> RunMetadata | None:
        """Retrieve run metadata from hot, fallback to cold."""
        metadata = await self.hot.get_run_metadata(run_id)

        if metadata is None and self.config.fallback_on_cold_miss:
            metadata = await self.cold.get_run_metadata(run_id)

        return metadata

    async def save_state_snapshot(self, snapshot: StateSnapshot) -> None:
        """Save state snapshot to hot storage."""
        await self.hot.save_state_snapshot(snapshot)

    async def get_state_snapshot(
        self, run_id: RunId, wave_number: int
    ) -> StateSnapshot | None:
        """Retrieve state snapshot from hot, fallback to cold."""
        snapshot = await self.hot.get_state_snapshot(run_id, wave_number)

        if snapshot is None and self.config.fallback_on_cold_miss:
            snapshot = await self.cold.get_state_snapshot(run_id, wave_number)

        return snapshot

    async def update_state_snapshot(self, snapshot: StateSnapshot) -> None:
        """Update state snapshot in hot storage."""
        await self.hot.update_state_snapshot(snapshot)

    async def get_snapshots_range(
        self,
        run_id: RunId,
        start_wave: int,
        end_wave: int,
        order: Literal["ASC", "DESC"] = "ASC",
    ) -> list[StateSnapshot]:
        """Retrieve range of snapshots from hot, fallback to cold."""
        snapshots = await self.hot.get_snapshots_range(
            run_id, start_wave, end_wave, order
        )

        if not snapshots and self.config.fallback_on_cold_miss:
            snapshots = await self.cold.get_snapshots_range(
                run_id, start_wave, end_wave, order
            )

        return snapshots

    async def save_trace(self, trace: ExecutionTrace) -> None:
        """Save execution trace to hot storage."""
        await self.hot.save_trace(trace)

    async def get_trace(self, run_id: RunId, wave_number: int) -> ExecutionTrace | None:
        """Retrieve execution trace from hot, fallback to cold."""
        trace = await self.hot.get_trace(run_id, wave_number)

        if trace is None and self.config.fallback_on_cold_miss:
            trace = await self.cold.get_trace(run_id, wave_number)

        return trace

    async def delete_trace(self, run_id: RunId, wave_number: int) -> bool:
        """Delete execution trace from hot storage."""
        return await self.hot.delete_trace(run_id, wave_number)

    async def save_node_trace(self, node_trace: NodeExecutionTrace) -> None:
        """Save node execution trace to hot storage."""
        await self.hot.save_node_trace(node_trace)

    async def get_node_trace(self, log_id: str) -> NodeExecutionTrace | None:
        """Retrieve node execution trace from hot, fallback to cold."""
        node_trace = await self.hot.get_node_trace(log_id)

        if node_trace is None and self.config.fallback_on_cold_miss:
            node_trace = await self.cold.get_node_trace(log_id)

        return node_trace

    async def list_runs(
        self,
        *,
        before: datetime | None = None,
        after: datetime | None = None,
        limit: int | None = None,
    ) -> list[RunMetadata]:
        """List runs from hot storage."""
        return await self.hot.list_runs(before=before, after=after, limit=limit)

    async def delete_run(
        self, run_id: RunId, *, keep_checkpoints: bool = False
    ) -> None:
        """Delete run from hot storage."""
        await self.hot.delete_run(run_id, keep_checkpoints=keep_checkpoints)

    async def append_events_batch(
        self,
        log_id: str,
        events: list[Any],
        start_sequence: int,
    ) -> None:
        """Append events batch to hot storage."""
        await self.hot.append_events_batch(log_id, events, start_sequence)

    async def stream_events(
        self,
        log_id: str,
        start_offset: int = 0,
        end_offset: int | None = None,
    ) -> list[Any]:
        """Stream events from hot storage."""
        return await self.hot.stream_events(log_id, start_offset, end_offset)

    async def move_to_cold(self, run_id: RunId) -> None:
        """Move run data from hot to cold storage.

        This is a utility method for implementing lifecycle policies.

        Args:
            run_id: Run to move to cold storage.

        """
        metadata = await self.hot.get_run_metadata(run_id)
        if metadata:
            await self.cold.save_run_metadata(metadata)

        runs = await self.hot.list_runs()
        for run in runs:
            if run.run_id == run_id:
                for wave in range(run.total_waves):
                    snapshot = await self.hot.get_state_snapshot(run_id, wave)
                    if snapshot:
                        await self.cold.save_state_snapshot(snapshot)

                    trace = await self.hot.get_trace(run_id, wave)
                    if trace:
                        await self.cold.save_trace(trace)

        await self.hot.delete_run(run_id, keep_checkpoints=False)
