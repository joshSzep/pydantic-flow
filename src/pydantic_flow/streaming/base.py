"""Base classes for streaming progress events.

This module defines the foundational types and base class for all
streaming progress items.
"""

from __future__ import annotations

from collections.abc import Awaitable
from collections.abc import Callable
from datetime import UTC
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING
from typing import Any

from pydantic import BaseModel
from pydantic import Field

if TYPE_CHECKING:
    from pydantic_flow.hitl.decisions import InterruptDecision


class ProgressType(StrEnum):
    """Type discriminator for progress items.

    Attributes:
        START: Stream begins.
        TOKEN: Text token from LLM.
        PARTIAL_FIELDS: Incremental structured field updates.
        TOOL_CALL: Tool invocation intent declared.
        TOOL_ARG_PROGRESS: Tool argument formation in progress.
        TOOL_RESULT: Tool execution result.
        RETRIEVAL: Retrieved item from search/db.
        METRIC: Performance or quality metric.
        ERROR: Non-fatal error or warning.
        END: Stream completes successfully.
        HEARTBEAT: Liveness signal during long operation.
        CACHE_HIT: Cache hit occurred.
        CACHE_MISS: Cache miss occurred.
        CACHE_WRITE: Cache write completed.
        CACHE_ERROR: Cache operation error.
        MEMORY_COMPRESSION_PENDING: Memory compression about to begin.
        MEMORY_COMPRESSION_COMPLETE: Memory compression completed.
        CHECKPOINT_SAVED: Checkpoint persisted to storage.

    """

    START = "start"
    TOKEN = "token"
    PARTIAL_FIELDS = "partial_fields"
    TOOL_CALL = "tool_call"
    TOOL_ARG_PROGRESS = "tool_arg_progress"
    TOOL_RESULT = "tool_result"
    RETRIEVAL = "retrieval"
    METRIC = "metric"
    ERROR = "error"
    END = "end"
    HEARTBEAT = "heartbeat"
    MEMORY_COMPRESSION_PENDING = "memory_compression_pending"
    MEMORY_COMPRESSION_COMPLETE = "memory_compression_complete"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    CACHE_WRITE = "cache_write"
    CACHE_ERROR = "cache_error"
    CHECKPOINT_SAVED = "checkpoint_saved"


class ProgressItem(BaseModel):
    """Base class for all streaming progress events.

    Attributes:
        type: Discriminator for the progress item type.
        timestamp: When the event occurred.
        run_id: Unique identifier for this execution run.
        node_id: Identifier of the node emitting this event.
        interrupt_callback: Optional callback to determine if execution should be
            interrupted.

    """

    model_config = {"frozen": False, "arbitrary_types_allowed": True}

    type: ProgressType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    run_id: str = ""
    node_id: str = ""
    interrupt_callback: Callable[[ProgressItem], Awaitable[Any]] | None = None

    def set_interrupt_callback(
        self, callback: Callable[[ProgressItem], Awaitable[Any]]
    ) -> None:
        """Set the interrupt callback for this progress item.

        Args:
            callback: Async function that determines if execution should interrupt.

        """
        self.interrupt_callback = callback

    async def check_interrupt(self) -> Any:
        """Check if this progress item should trigger an interrupt.

        Returns:
            InterruptDecision indicating whether to interrupt.

        """
        if self.interrupt_callback is None:
            from pydantic_flow.hitl.decisions import InterruptDecision  # noqa: PLC0415

            return InterruptDecision.proceed()
        return await self.interrupt_callback(self)


if TYPE_CHECKING:
    InterruptCallback = Callable[[ProgressItem], Awaitable[InterruptDecision]]
else:
    InterruptCallback = Callable
