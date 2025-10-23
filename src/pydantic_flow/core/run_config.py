"""Run configuration for flow execution.

This module provides configuration options for flow execution including
recursion limits, timeouts, and observability settings.
"""

from pydantic import BaseModel
from pydantic import Field


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

    """

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
