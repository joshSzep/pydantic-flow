"""Storage backend interface for checkpoint v2.

This module defines the protocol that all checkpoint storage backends must
implement, enabling pluggable storage solutions.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from typing import Literal
from typing import Protocol

from pydantic_flow.checkpoints.types import ExecutionTrace
from pydantic_flow.checkpoints.types import NodeExecutionTrace
from pydantic_flow.checkpoints.types import RunId
from pydantic_flow.checkpoints.types import RunMetadata
from pydantic_flow.checkpoints.types import StateSnapshot


class CheckpointStorageBackend(Protocol):
    """Protocol for checkpoint storage backends.

    All storage backends must implement this interface to provide consistent
    checkpoint persistence across different storage solutions (SQLite, Postgres,
    S3, etc.).

    All methods are async to support various storage backends. Implementations
    must be safe for concurrent access according to their backend's capabilities.
    """

    async def save_run_metadata(self, metadata: RunMetadata) -> None:
        """Save run metadata.

        Args:
            metadata: Run metadata to save.

        Raises:
            Exception: If save operation fails.

        """
        ...

    async def get_run_metadata(self, run_id: RunId) -> RunMetadata | None:
        """Retrieve run metadata.

        Args:
            run_id: Run identifier.

        Returns:
            Run metadata if found, None otherwise.

        Raises:
            Exception: If retrieval operation fails.

        """
        ...

    async def save_state_snapshot(self, snapshot: StateSnapshot) -> None:
        """Save state snapshot.

        Snapshots are the primary mechanism for resuming execution. This method
        must be atomic and idempotent.

        Args:
            snapshot: State snapshot to save.

        Raises:
            Exception: If save operation fails.

        """
        ...

    async def get_state_snapshot(
        self, run_id: RunId, wave_number: int
    ) -> StateSnapshot | None:
        """Retrieve state snapshot for a specific wave.

        Args:
            run_id: Run identifier.
            wave_number: Wave/step number.

        Returns:
            State snapshot if found, None otherwise.

        Raises:
            Exception: If retrieval operation fails.

        """
        ...

    async def update_state_snapshot(self, snapshot: StateSnapshot) -> None:
        """Update existing state snapshot.

        Used to update checkpoint-trace bidirectional references after trace
        is saved.

        Args:
            snapshot: Updated state snapshot.

        Raises:
            Exception: If update operation fails.

        """
        ...

    async def get_snapshots_range(
        self,
        run_id: RunId,
        start_wave: int,
        end_wave: int,
        order: Literal["ASC", "DESC"] = "ASC",
    ) -> list[StateSnapshot]:
        """Retrieve range of snapshots for state reconstruction.

        This method enables efficient batch fetching for state reconstruction,
        supporting both forward (ASC) and backward (DESC) time travel.

        Args:
            run_id: Run identifier.
            start_wave: Starting wave number (inclusive).
            end_wave: Ending wave number (inclusive).
            order: Sort order ("ASC" for forward, "DESC" for backward).

        Returns:
            List of state snapshots in requested order.

        Raises:
            Exception: If retrieval operation fails.

        """
        ...

    async def save_trace(self, trace: ExecutionTrace) -> None:
        """Save execution trace.

        Traces provide detailed execution history for debugging. The trace's
        checkpoint_snapshot_id field must reference a valid checkpoint.

        Args:
            trace: Execution trace to save.

        Raises:
            ValueError: If checkpoint_snapshot_id references invalid checkpoint.
            Exception: If save operation fails.

        """
        ...

    async def get_trace(self, run_id: RunId, wave_number: int) -> ExecutionTrace | None:
        """Retrieve execution trace for a specific wave.

        Args:
            run_id: Run identifier.
            wave_number: Wave/step number.

        Returns:
            Execution trace if found, None otherwise.

        Raises:
            Exception: If retrieval operation fails.

        """
        ...

    async def delete_trace(self, run_id: RunId, wave_number: int) -> bool:
        """Delete execution trace.

        Part of vacuum operations to reclaim storage space.

        Args:
            run_id: Run identifier.
            wave_number: Wave/step number.

        Returns:
            True if trace was deleted, False if not found.

        Raises:
            Exception: If deletion operation fails.

        """
        ...

    async def save_node_trace(self, node_trace: NodeExecutionTrace) -> None:
        """Save node execution trace.

        Args:
            node_trace: Node execution trace to save.

        Raises:
            Exception: If save operation fails.

        """
        ...

    async def get_node_trace(self, log_id: str) -> NodeExecutionTrace | None:
        """Retrieve node execution trace.

        Args:
            log_id: Node trace log identifier.

        Returns:
            Node execution trace if found, None otherwise.

        Raises:
            Exception: If retrieval operation fails.

        """
        ...

    async def append_events_batch(
        self,
        log_id: str,
        events: list[Any],
        start_sequence: int,
    ) -> None:
        """Append batch of events to event log.

        Args:
            log_id: Event log identifier.
            events: List of progress items to append.
            start_sequence: Starting sequence number for this batch.

        Raises:
            Exception: If append operation fails.

        """
        ...

    async def stream_events(
        self,
        log_id: str,
        start_offset: int = 0,
        end_offset: int | None = None,
    ) -> list[Any]:
        """Stream events from event log.

        Args:
            log_id: Event log identifier.
            start_offset: Starting offset for event stream.
            end_offset: Optional ending offset.

        Returns:
            List of progress items.

        Raises:
            Exception: If streaming operation fails.

        """
        ...

    async def list_runs(
        self,
        *,
        before: datetime | None = None,
        after: datetime | None = None,
        limit: int | None = None,
    ) -> list[RunMetadata]:
        """List runs with optional filtering.

        Used by CLI and vacuum operations to query run history.

        Args:
            before: Only return runs started before this time.
            after: Only return runs started after this time.
            limit: Maximum number of runs to return.

        Returns:
            List of run metadata, newest first.

        Raises:
            Exception: If query operation fails.

        """
        ...

    async def delete_run(
        self, run_id: RunId, *, keep_checkpoints: bool = False
    ) -> None:
        """Delete all data for a run.

        Part of vacuum operations. Can optionally preserve checkpoints while
        deleting traces and events.

        Args:
            run_id: Run identifier.
            keep_checkpoints: If True, keep state snapshots but delete traces.

        Raises:
            Exception: If deletion operation fails.

        """
        ...

    async def initialize(self) -> None:
        """Initialize backend resources.

        Called once before any operations. Should create schemas, connections,
        pools, etc.

        Raises:
            Exception: If initialization fails.

        """
        ...

    async def close(self) -> None:
        """Close backend resources.

        Called during shutdown. Should close connections, pools, etc.

        Raises:
            Exception: If cleanup fails.

        """
        ...

    async def healthcheck(self) -> bool:
        """Check backend health.

        Used for monitoring and startup validation.

        Returns:
            True if backend is healthy and operational.

        """
        ...
