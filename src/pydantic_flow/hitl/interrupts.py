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

from pydantic_flow.checkpoints.types import RunId
from pydantic_flow.checkpoints.types import SnapshotId
from pydantic_flow.checkpoints.types import StateSnapshot

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


class InterruptionRequested(Exception):
    """Raised when a HITL interrupt occurs using unified checkpoint system.

    This exception uses the unified checkpoint system, enabling time-travel,
    forking, and universal resume for HITL interrupts.

    Attributes:
        snapshot: The complete StateSnapshot that was saved at interrupt point.
        decision: The interrupt decision that triggered this exception.
        metadata: Additional context about the interrupt.

    """

    def __init__(
        self,
        snapshot: StateSnapshot,
        decision: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the interruption exception.

        Args:
            snapshot: Complete StateSnapshot at interrupt point
                (includes run_id, snapshot_id, wave_number).
            decision: InterruptDecision that triggered the interrupt.
            metadata: Additional interrupt context.

        """
        self.snapshot = snapshot
        self.decision = decision
        self.metadata = metadata or {}
        super().__init__(
            f"Execution interrupted at node {snapshot.interrupted_node_id} "
            f"(run: {snapshot.run_id}, wave: {snapshot.wave_number}, "
            f"snapshot: {snapshot.snapshot_id})"
        )

    @property
    def snapshot_id(self) -> SnapshotId:
        """Snapshot ID for convenient access."""
        return self.snapshot.snapshot_id

    @property
    def run_id(self) -> RunId:
        """Run ID for convenient access."""
        return self.snapshot.run_id

    @property
    def interrupted_node_id(self) -> str:
        """Interrupted node ID for convenient access."""
        return self.snapshot.interrupted_node_id or "unknown"
