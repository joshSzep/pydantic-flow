"""Custom exceptions for pydantic-flow.

This module provides specialized exception types for flow execution errors.

BREAKING CHANGE: Added InterruptionRequested exception and FlowCheckpoint model
for HITL (Human-in-the-Loop) support.
"""

from __future__ import annotations

from collections.abc import Awaitable
from collections.abc import Callable
from enum import IntEnum
from typing import Any

from pydantic import BaseModel
from pydantic import Field


class HandlerPriority(IntEnum):
    """Priority levels for interrupt handlers.

    Lower values execute first. Critical handlers (0-25) always execute,
    even during high-throughput scenarios.

    Attributes:
        CRITICAL: Critical handlers that must always run (0-25).
        HIGH: High-priority handlers (26-50).
        NORMAL: Normal-priority handlers (51-75).
        LOW: Low-priority handlers (76-100).

    """

    CRITICAL = 0
    HIGH = 26
    NORMAL = 51
    LOW = 76


class InterruptHandlerRegistration(BaseModel):
    """Registration record for an interrupt callback handler.

    Attributes:
        callback: Async function that determines if execution should interrupt.
        priority: Priority level determining execution order and criticality.
        metadata: Additional context about the handler.

    """

    model_config = {"arbitrary_types_allowed": True}

    callback: Callable[[Any], Awaitable[Any]]
    priority: int = HandlerPriority.NORMAL
    metadata: dict[str, Any] = Field(default_factory=dict)


class FlowCheckpoint(BaseModel):
    """Serializable checkpoint for resuming interrupted flows.

    Attributes:
        flow_id: Unique identifier for the flow instance.
        run_id: Unique identifier for this execution run.
        interrupted_node_id: ID of the node where interruption occurred.
        node_states: Captured state of all nodes at interruption time.
        edge_history: Sequence of edges traversed before interruption.
        metadata: Additional context about the checkpoint.

    """

    flow_id: str
    run_id: str
    interrupted_node_id: str
    node_states: dict[str, Any]
    edge_history: list[tuple[str, str]]
    metadata: dict[str, Any] = Field(default_factory=dict)


class FlowError(Exception):
    """Base exception for flow-related errors."""


class RecursionLimitError(FlowError):
    """Raised when the maximum number of execution steps is exceeded.

    This prevents infinite loops in flows with conditional routing.
    The error message includes the step count and recent trace information.
    """


class RoutingError(FlowError):
    """Raised when a routing function returns an invalid target.

    This occurs when:
    - A router returns a node name that doesn't exist in the flow
    - A router returns an empty list of targets without ending
    - A router's output cannot be mapped via the provided mapping dict
    """


class FlowTimeoutError(FlowError):
    """Raised when flow execution exceeds the configured timeout.

    This prevents flows from running indefinitely when a time limit is set.
    """


class InterruptionRequested(FlowError):
    """Raised when a HITL interrupt callback requests execution halt.

    This exception carries the checkpoint and decision information needed
    to resume execution after human intervention.

    Attributes:
        checkpoint: Serializable state for resuming the flow.
        decision: The interrupt decision that triggered this exception.

    """

    def __init__(self, checkpoint: FlowCheckpoint, decision: Any) -> None:
        """Initialize the interruption exception.

        Args:
            checkpoint: Serializable state for resuming.
            decision: InterruptDecision that triggered the interrupt.

        """
        self.checkpoint = checkpoint
        self.decision = decision
        super().__init__(
            f"Execution interrupted at node {checkpoint.interrupted_node_id}"
        )
