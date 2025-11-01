"""Core types for checkpoint v2.

This module defines the fundamental data models for state snapshots,
execution traces, and related types.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC
from datetime import datetime
from enum import StrEnum
import hashlib
import json
import secrets
from typing import Any

from pydantic import BaseModel
from pydantic import Field
from pydantic_core import core_schema

from pydantic_flow.checkpoints.serialization import TypedSerializer
from pydantic_flow.checkpoints.serialization import compress
from pydantic_flow.checkpoints.serialization import decompress


class CheckpointId(str):
    """Unique identifier for a checkpoint."""

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        """Tell Pydantic to treat CheckpointId as a string."""
        return core_schema.no_info_after_validator_function(
            cls, core_schema.str_schema()
        )


class RunId(str):
    """Unique identifier for a flow execution run."""

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        """Tell Pydantic to treat RunId as a string."""
        return core_schema.no_info_after_validator_function(
            cls, core_schema.str_schema()
        )


class SnapshotId(str):
    """Unique identifier for a state snapshot."""

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        """Tell Pydantic to treat SnapshotId as a string."""
        return core_schema.no_info_after_validator_function(
            cls, core_schema.str_schema()
        )


def generate_checkpoint_id() -> CheckpointId:
    """Generate a unique checkpoint ID.

    Returns:
        New checkpoint ID.

    """
    return CheckpointId(secrets.token_urlsafe(16))


def generate_run_id() -> RunId:
    """Generate a unique run ID.

    Returns:
        New run ID.

    """
    return RunId(secrets.token_urlsafe(16))


def generate_snapshot_id() -> SnapshotId:
    """Generate a unique snapshot ID.

    Returns:
        New snapshot ID.

    """
    return SnapshotId(secrets.token_urlsafe(16))


def generate_trace_id() -> str:
    """Generate a unique trace ID.

    Returns:
        New trace ID.

    """
    return secrets.token_urlsafe(16)


class DeletedKey:
    """Sentinel type for deleted keys in reverse deltas."""

    def __repr__(self) -> str:
        """Represent the deleted key sentinel."""
        return "<DeletedKey>"

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        """Tell Pydantic how to handle DeletedKey in schemas."""
        from pydantic_core import core_schema

        return core_schema.is_instance_schema(cls)


DELETED_KEY = DeletedKey()


class StateSnapshot(BaseModel):
    """State snapshot for resume with time-travel support.

    Attributes:
        version: Version for forward/backward compatibility.
        snapshot_id: Unique identifier for this snapshot.
        run_id: Flow execution run identifier.
        wave_number: Wave/step number in execution.
        forward_delta: State changes to apply forward.
        reverse_delta: State changes to apply backward.
        full_state: Complete state (stored every Nth wave).
        state_hash: SHA-256 hash of complete state.
        next_frontier: Nodes to execute next.
        routing_ended: Whether routing ended with Route.END.
        trace_id: Reference to associated execution trace.
        created_at: Timestamp of snapshot creation.

    """

    version: int = 2
    snapshot_id: SnapshotId
    run_id: RunId
    wave_number: int
    forward_delta: dict[str, BaseModel] | None = None
    reverse_delta: dict[str, BaseModel | DeletedKey] | None = None
    full_state: dict[str, BaseModel] | None = None
    state_hash: str
    next_frontier: list[str]
    routing_ended: bool
    trace_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def serialize(self) -> bytes:
        """Serialize with type preservation and compression.

        Returns:
            Compressed serialized bytes.

        """
        data = TypedSerializer.serialize(self)
        return compress(data, level=6)

    @classmethod
    def deserialize(cls, data: bytes) -> StateSnapshot:
        """Deserialize with type reconstruction.

        Args:
            data: Compressed serialized bytes.

        Returns:
            Reconstructed StateSnapshot.

        """
        decompressed = decompress(data)
        return TypedSerializer.deserialize(decompressed)

    def compute_state_hash(self, complete_state: Mapping[str, BaseModel]) -> str:
        """Compute SHA-256 hash of complete state.

        Args:
            complete_state: Complete state dictionary to hash.

        Returns:
            Hex-encoded SHA-256 hash.

        """
        canonical_json = json.dumps(
            {k: v.model_dump(mode="json") for k, v in complete_state.items()},
            sort_keys=True,
        )
        return hashlib.sha256(canonical_json.encode()).hexdigest()


class StateRef(BaseModel):
    """Reference to state in a checkpoint.

    Attributes:
        snapshot_id: ID of snapshot containing the state.
        state_key: Key within the snapshot.

    """

    snapshot_id: SnapshotId
    state_key: str


class ExecutionError(BaseModel):
    """Execution error details.

    Attributes:
        error_type: Type of error (class name).
        error_message: Error message.
        traceback: Optional traceback string.

    """

    error_type: str
    error_message: str
    traceback: str | None = None


class EventSummary(BaseModel):
    """Aggregate statistics from event stream.

    Attributes:
        total_events: Total number of events.
        token_count: Total tokens emitted.
        tool_call_count: Number of tool calls.
        cache_hits: Number of cache hits.
        tool_calls: List of tool names called.

    """

    total_events: int
    token_count: int
    tool_call_count: int
    cache_hits: int
    tool_calls: list[str]


class EventRef(BaseModel):
    """Reference to events in storage.

    Attributes:
        log_id: Unique identifier for the event log.
        start_offset: Starting offset in the event stream.
        end_offset: Ending offset in the event stream.

    """

    log_id: str
    start_offset: int
    end_offset: int


class NodeExecutionTrace(BaseModel):
    """Complete trace of a node's execution.

    Attributes:
        log_id: Unique identifier for this trace.
        node_id: Node identifier.
        wave_number: Wave/step number.
        snapshot_id: Associated snapshot ID.
        input_ref: Reference to input state.
        output_ref: Reference to output state (if successful).
        event_log_id: ID of the event log.
        total_events: Total number of events.
        event_summary: Aggregate event statistics.
        started_at: Start timestamp.
        completed_at: Completion timestamp.
        next_nodes: Next nodes to execute.
        route_decision: Routing decision made (if any).
        cache_hit: Whether execution used cache.
        cache_key: Cache key used (if any).
        error: Error details (if failed).

    """

    log_id: str
    node_id: str
    wave_number: int
    snapshot_id: SnapshotId
    input_ref: StateRef
    output_ref: StateRef | None
    event_log_id: str
    total_events: int
    event_summary: EventSummary
    started_at: datetime
    completed_at: datetime
    next_nodes: list[str]
    route_decision: Any | None = None
    cache_hit: bool = False
    cache_key: str | None = None
    error: ExecutionError | None = None


class ExecutionTrace(BaseModel):
    """Complete execution trace for debugging.

    Attributes:
        trace_id: Unique identifier for this trace.
        run_id: Flow execution run identifier.
        wave_number: Wave/step number.
        checkpoint_snapshot_id: Required reference to checkpoint snapshot.
        node_traces: List of node execution traces.
        parallel_batch_id: ID for parallel execution batch.
        started_at: Start timestamp.
        completed_at: Completion timestamp.

    """

    trace_id: str
    run_id: RunId
    wave_number: int
    checkpoint_snapshot_id: SnapshotId
    node_traces: list[NodeExecutionTrace]
    parallel_batch_id: str
    started_at: datetime
    completed_at: datetime


class RunMetadata(BaseModel):
    """Metadata about a flow execution run.

    Attributes:
        run_id: Unique identifier for the run.
        flow_id: Flow identifier.
        started_at: Run start timestamp.
        completed_at: Run completion timestamp (if finished).
        status: Current run status.
        total_waves: Total number of waves executed.
        error: Error details (if failed).

    """

    class Status(StrEnum):
        """Run status enumeration."""

        RUNNING = "running"
        COMPLETED = "completed"
        FAILED = "failed"

    run_id: RunId
    flow_id: str
    started_at: datetime
    completed_at: datetime | None = None
    status: Status = Status.RUNNING
    total_waves: int = 0
    error: ExecutionError | None = None
