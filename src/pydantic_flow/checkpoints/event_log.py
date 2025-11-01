"""Event streaming and logging for checkpoint v2.

This module provides the StreamingEventLog class for capturing and persisting
execution events with memory safety and backpressure handling.
"""

from __future__ import annotations

import asyncio
from datetime import UTC
from datetime import datetime
import secrets
from typing import Any

from pydantic_flow.cache.events import CacheHit
from pydantic_flow.checkpoints.interface import CheckpointStorageBackend
from pydantic_flow.checkpoints.serialization import TypedSerializer
from pydantic_flow.checkpoints.types import EventSummary
from pydantic_flow.checkpoints.types import NodeExecutionTrace
from pydantic_flow.checkpoints.types import RunId
from pydantic_flow.checkpoints.types import SnapshotId
from pydantic_flow.checkpoints.types import StateRef
from pydantic_flow.streaming import ProgressItem
from pydantic_flow.streaming import TokenChunk
from pydantic_flow.streaming import ToolCall


def generate_log_id() -> str:
    """Generate a unique log ID.

    Returns:
        New log ID.

    """
    return secrets.token_urlsafe(16)


class StreamingEventLog:
    """Captures execution events and persists them to storage.

    This class handles event buffering with dual limits (count + bytes) to prevent
    memory exhaustion. It supports async flushing with backpressure and circuit
    breaker patterns for resilience.

    Attributes:
        store: Backend storage for event persistence.
        run_id: Flow execution run identifier.
        node_id: Node identifier.
        wave_number: Wave/step number.
        log_id: Unique identifier for this event log.
        snapshot_id: Associated snapshot ID.
        buffer_size: Maximum number of events before flush.
        buffer_max_bytes: Maximum buffer size in bytes before flush.

    """

    def __init__(  # noqa: PLR0913
        self,
        store: CheckpointStorageBackend,
        run_id: RunId,
        node_id: str,
        wave_number: int,
        snapshot_id: SnapshotId,
        *,
        buffer_size: int = 100,
        buffer_max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    ) -> None:
        """Initialize streaming event log.

        Args:
            store: Backend storage for events.
            run_id: Flow execution run identifier.
            node_id: Node identifier.
            wave_number: Wave/step number.
            snapshot_id: Associated snapshot ID.
            buffer_size: Maximum events in buffer before flush.
            buffer_max_bytes: Maximum buffer bytes before flush.

        """
        self.store = store
        self.run_id = run_id
        self.node_id = node_id
        self.wave_number = wave_number
        self.snapshot_id = snapshot_id
        self.log_id = generate_log_id()

        # Event buffering with dual limits
        self.event_buffer: list[ProgressItem] = []
        self.buffer_size = buffer_size
        self.buffer_max_bytes = buffer_max_bytes
        self.buffer_current_bytes = 0
        self.buffer_lock = asyncio.Lock()

        # Tracking
        self.flush_task: asyncio.Task[None] | None = None
        self.flush_failures = 0
        self.max_flush_failures = 10  # Circuit breaker

        # Metadata
        self.total_events = 0
        self.token_count = 0
        self.tool_calls: list[str] = []
        self.cache_hits = 0
        self.started_at = datetime.now(UTC)
        self.completed_at: datetime | None = None

    async def append(self, event: ProgressItem) -> None:
        """Append event with memory safety.

        Args:
            event: Progress item to append.

        Raises:
            RuntimeError: If circuit breaker is triggered.

        """
        if self.flush_failures >= self.max_flush_failures:
            msg = "Event flush circuit breaker triggered"
            raise RuntimeError(msg)

        # Update metadata
        self._update_metadata(event)

        # Calculate event size (exclude callback for serialization)
        # Create a serializable copy without callback
        event_dict = event.model_dump(mode="python", exclude={"interrupt_callback"})
        event_bytes = len(TypedSerializer.serialize(event_dict))

        async with self.buffer_lock:
            self.event_buffer.append(event)
            self.buffer_current_bytes += event_bytes
            self.total_events += 1

        # Flush when EITHER limit reached
        if (
            len(self.event_buffer) >= self.buffer_size
            or self.buffer_current_bytes >= self.buffer_max_bytes
        ):
            await self._maybe_start_flush()

            # Backpressure if buffer still large
            if (
                len(self.event_buffer) > self.buffer_size * 2
                and self.flush_task
                and not self.flush_task.done()
            ):
                try:
                    await asyncio.wait_for(self.flush_task, timeout=10.0)
                except TimeoutError:
                    msg = "Flush timeout during backpressure"
                    raise RuntimeError(msg) from None

    async def _maybe_start_flush(self) -> None:
        """Start flush if not already running."""
        if self.flush_task and not self.flush_task.done():
            return
        self.flush_task = asyncio.create_task(self._flush())

    async def _flush(self) -> None:
        """Flush buffer to storage."""
        async with self.buffer_lock:
            if not self.event_buffer:
                return
            batch = self.event_buffer.copy()
            self.event_buffer.clear()
            self.buffer_current_bytes = 0
            start_seq = self.total_events - len(batch)

        try:
            await self.store.append_events_batch(
                log_id=self.log_id,
                events=batch,
                start_sequence=start_seq,
            )
            self.flush_failures = 0
        except Exception as e:
            self.flush_failures += 1
            import logging

            logger = logging.getLogger(__name__)
            logger.error(
                f"Flush failed ({self.flush_failures}/{self.max_flush_failures}): {e}"
            )

            if self.flush_failures >= self.max_flush_failures:
                logger.critical("Event flush circuit breaker triggered")
                raise

    def _update_metadata(self, event: ProgressItem) -> None:
        """Update metadata based on event type.

        Args:
            event: Progress item to analyze.

        """
        if isinstance(event, TokenChunk):
            self.token_count += len(event.text)
        elif isinstance(event, ToolCall):
            self.tool_calls.append(event.tool_name)
        elif isinstance(event, CacheHit):
            self.cache_hits += 1

    async def finalize(  # noqa: PLR0913
        self,
        input_ref: StateRef,
        output_ref: StateRef | None = None,
        next_nodes: list[str] | None = None,
        route_decision: Any | None = None,
        cache_hit: bool = False,
        cache_key: str | None = None,
        error: dict[str, Any] | None = None,
    ) -> NodeExecutionTrace:
        """Finalize the event log and create trace.

        Args:
            input_ref: Reference to input state.
            output_ref: Reference to output state (if successful).
            next_nodes: Next nodes to execute.
            route_decision: Routing decision made (if any).
            cache_hit: Whether execution used cache.
            cache_key: Cache key used (if any).
            error: Error details (if failed).

        Returns:
            NodeExecutionTrace with complete execution information.

        """
        # Final flush
        if self.event_buffer:
            await self._flush()

        # Wait for flush to complete
        if self.flush_task and not self.flush_task.done():
            await self.flush_task

        self.completed_at = datetime.now(UTC)

        # Build event summary
        event_summary = EventSummary(
            total_events=self.total_events,
            token_count=self.token_count,
            tool_call_count=len(self.tool_calls),
            cache_hits=self.cache_hits,
            tool_calls=self.tool_calls,
        )

        from pydantic_flow.checkpoints.types import ExecutionError

        # Create trace
        return NodeExecutionTrace(
            log_id=self.log_id,
            node_id=self.node_id,
            wave_number=self.wave_number,
            snapshot_id=self.snapshot_id,
            input_ref=input_ref,
            output_ref=output_ref,
            event_log_id=self.log_id,
            total_events=self.total_events,
            event_summary=event_summary,
            started_at=self.started_at,
            completed_at=self.completed_at,
            next_nodes=next_nodes or [],
            route_decision=route_decision,
            cache_hit=cache_hit,
            cache_key=cache_key,
            error=ExecutionError(**error) if error else None,
        )


def create_minimal_trace(  # noqa: PLR0913
    node_id: str,
    wave_number: int,
    snapshot_id: SnapshotId,
    input_ref: StateRef,
    output_ref: StateRef | None = None,
    next_nodes: list[str] | None = None,
    route_decision: Any | None = None,
    cache_hit: bool = False,
    cache_key: str | None = None,
    error: dict[str, Any] | None = None,
) -> NodeExecutionTrace:
    """Create a minimal trace when sampling is disabled.

    Args:
        node_id: Node identifier.
        wave_number: Wave/step number.
        snapshot_id: Associated snapshot ID.
        input_ref: Reference to input state.
        output_ref: Reference to output state (if successful).
        next_nodes: Next nodes to execute.
        route_decision: Routing decision made (if any).
        cache_hit: Whether execution used cache.
        cache_key: Cache key used (if any).
        error: Error details (if failed).

    Returns:
        NodeExecutionTrace with minimal information.

    """
    from pydantic_flow.checkpoints.types import ExecutionError

    now = datetime.now(UTC)
    return NodeExecutionTrace(
        log_id=generate_log_id(),
        node_id=node_id,
        wave_number=wave_number,
        snapshot_id=snapshot_id,
        input_ref=input_ref,
        output_ref=output_ref,
        event_log_id="",  # No events captured
        total_events=0,
        event_summary=EventSummary(
            total_events=0,
            token_count=0,
            tool_call_count=0,
            cache_hits=0,
            tool_calls=[],
        ),
        started_at=now,
        completed_at=now,
        next_nodes=next_nodes or [],
        route_decision=route_decision,
        cache_hit=cache_hit,
        cache_key=cache_key,
        error=ExecutionError(**error) if error else None,
    )
