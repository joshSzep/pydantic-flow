"""HITL decision types for interrupt handling.

This module provides the decision types that interrupt callbacks return
to control flow execution.
"""

from __future__ import annotations

from collections.abc import Awaitable
from collections.abc import Callable
from typing import TYPE_CHECKING
from typing import Any

from pydantic import BaseModel
from pydantic import Field

if TYPE_CHECKING:
    pass


class InterruptDecision(BaseModel):
    """Decision returned by interrupt callback handlers.

    Attributes:
        should_interrupt: Whether to interrupt execution.
        reason: Human-readable explanation for the interruption decision.
        replacement_value: Optional replacement value to inject into the stream.
        metadata: Additional context about the decision.

    """

    should_interrupt: bool
    reason: str | None = None
    replacement_value: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def proceed(reason: str | None = None) -> InterruptDecision:
        """Create a decision to continue execution.

        Args:
            reason: Optional explanation for continuing.

        Returns:
            InterruptDecision with should_interrupt=False.

        """
        return InterruptDecision(should_interrupt=False, reason=reason)

    @staticmethod
    def interrupt(
        reason: str,
        replacement_value: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> InterruptDecision:
        """Create a decision to interrupt execution.

        Args:
            reason: Explanation for interruption.
            replacement_value: Optional value to inject into the stream.
            metadata: Additional context about the interruption.

        Returns:
            InterruptDecision with should_interrupt=True.

        """
        return InterruptDecision(
            should_interrupt=True,
            reason=reason,
            replacement_value=replacement_value,
            metadata=metadata or {},
        )


InterruptCallback = Callable[[Any], Awaitable[InterruptDecision]]
