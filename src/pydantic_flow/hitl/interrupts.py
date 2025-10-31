"""HITL interruption exceptions and handler registration.

This module provides the core interrupt mechanism for Human-in-the-Loop flows,
including exception types, priority levels, and handler registration.
"""

from __future__ import annotations

from collections.abc import Awaitable
from collections.abc import Callable
from enum import IntEnum
from typing import TYPE_CHECKING
from typing import Any

from pydantic import BaseModel
from pydantic import Field

if TYPE_CHECKING:
    pass


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
        conversation_memory: Optional serialized conversation memory state.
        execution_progress: Map of node_id to execution status
            (pending/running/completed/failed).
        checkpoint_reason: Reason for checkpoint creation
            (node_completion/interruption/flow_end/error).
        checkpoint_node_id: ID of the node that just completed
            (if reason is node_completion).
        metadata: Additional context about the checkpoint.

    """

    flow_id: str
    run_id: str
    interrupted_node_id: str
    node_states: dict[str, Any]
    edge_history: list[tuple[str, str]]
    conversation_memory: Any = None
    execution_progress: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Map of node_id to execution status: pending, running, completed, failed"
        ),
    )
    checkpoint_reason: str = Field(
        default="interruption",
        description=(
            "Reason for checkpoint: node_completion, interruption, flow_end, error"
        ),
    )
    checkpoint_node_id: str | None = Field(
        default=None,
        description=(
            "ID of node that just completed (for node_completion checkpoints)"
        ),
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class InterruptionRequested(Exception):
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
