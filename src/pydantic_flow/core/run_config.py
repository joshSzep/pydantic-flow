"""Run configuration for flow execution.

This module provides configuration options for flow execution including
recursion limits, timeouts, observability settings, and checkpoint persistence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from pydantic import BaseModel
from pydantic import Field

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
        checkpoint_store: Optional checkpoint store for persistence.
            If provided, checkpoints are saved on interruption.
        run_id: Optional run identifier for checkpoint correlation.
            If not provided, a new UUID will be generated.

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
    checkpoint_store: Any | None = Field(
        default=None,
        description="Optional checkpoint store for persistence",
    )
    run_id: str | None = Field(
        default=None,
        description="Optional run identifier for checkpoint correlation",
    )
