"""Run configuration for flow execution.

This module provides configuration options for flow execution including
recursion limits, timeouts, observability settings, and checkpoint persistence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from pydantic import BaseModel
from pydantic import Field

from pydantic_flow.core.durability import DurabilityMode

if TYPE_CHECKING:
    pass


class RunConfig(BaseModel):
    """Configuration for flow execution.

    Attributes:
        max_steps: Maximum number of execution steps before raising
            RecursionLimitError. Prevents infinite loops. Default is 25.
        timeout_seconds: Optional timeout in seconds. If exceeded, raises
            TimeoutError. None means no timeout.
        trace_iterations: Whether to emit structured iteration events for
            observability. Default is True.
        recent_events_count: Number of recent iterations to include in
            RecursionLimitError messages for debugging. Default is 3.
        durability_mode: Controls checkpoint frequency. Options:
            - ASYNC: Background checkpoint after each node (default, recommended)
            - SYNC: Synchronous checkpoint before next node starts
            - EXIT: Checkpoint only on flow completion or error

        Note: HITL (Human-in-the-Loop) interruptions create their own checkpoints
        independently of this setting. This mode controls automatic crash recovery.

        checkpoint_store: Optional checkpoint store for persistence.
            If provided, checkpoints are saved according to durability_mode.
        run_id: Optional run identifier for checkpoint correlation.
            If not provided, a new UUID will be generated.
        max_checkpoint_size_mb: Maximum checkpoint size in MB. Larger checkpoints
            emit warnings but don't fail. Default is 100 MB.
        checkpoint_compression: Enable gzip compression for node_states to reduce
            storage size. Default is True.

    """

    model_config = {"arbitrary_types_allowed": True}

    max_steps: int = Field(default=25, ge=1)
    timeout_seconds: int | None = Field(default=None, ge=1)
    trace_iterations: bool = Field(default=True)
    recent_events_count: int = Field(
        default=3,
        ge=1,
        le=100,
        description=(
            "Number of recent iterations to include in RecursionLimitError messages"
        ),
    )
    durability_mode: DurabilityMode = Field(
        default=DurabilityMode.ASYNC,
        description="Checkpoint frequency mode for crash recovery",
    )
    checkpoint_store: Any | None = Field(
        default=None,
        description="Optional checkpoint store for persistence",
    )
    run_id: str | None = Field(
        default=None,
        description="Optional run identifier for checkpoint correlation",
    )
    max_checkpoint_size_mb: int = Field(
        default=100,
        ge=1,
        description=(
            "Maximum checkpoint size in MB. "
            "Larger checkpoints will emit warning but not fail."
        ),
    )
    checkpoint_compression: bool = Field(
        default=True,
        description="Enable gzip compression for checkpoint node_states",
    )
