"""Memory-related streaming events.

This module defines events related to memory operations,
particularly memory compression.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from pydantic_flow.streaming.base import ProgressItem
from pydantic_flow.streaming.base import ProgressType


class MemoryCompressionPending(ProgressItem):
    """Memory compression is about to begin.

    This event is emitted before conversation memory compression occurs,
    allowing interrupt handlers to approve, reject, or configure the compression.

    The interrupt callback can:
    - Proceed: Allow compression to continue with the configured compressor
    - Interrupt: Cancel compression for this operation
    - Replace: Provide a different compressor via replacement_value

    Attributes:
        estimated_tokens: Estimated token count in current conversation history.
        message_count: Number of messages in the conversation history.
        compressor_name: Name of the compressor strategy being used.
        compression_reason: Human-readable explanation for why compression is needed.
        metadata: Additional context about the compression trigger.

    Example:
        ```python
        async def approve_compression(item: ProgressItem) -> InterruptDecision:
            if isinstance(item, MemoryCompressionPending):
                if item.message_count < 20:
                    return InterruptDecision.interrupt("Too few messages")
            return InterruptDecision.proceed()

        event.set_interrupt_callback(approve_compression)
        decision = await event.check_interrupt()
        ```

    """

    type: ProgressType = ProgressType.MEMORY_COMPRESSION_PENDING
    estimated_tokens: int
    message_count: int
    compressor_name: str
    compression_reason: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryCompressionComplete(ProgressItem):
    """Memory compression has completed.

    This event is emitted after conversation memory compression completes,
    providing metrics and allowing interrupt handlers to accept, reject,
    or modify the compressed result.

    The interrupt callback can:
    - Proceed: Accept the compressed messages
    - Interrupt: Reject compression and restore original messages
    - Replace: Provide alternative compressed messages via replacement_value

    Attributes:
        metrics: Detailed compression metrics including token savings.
        compressed_messages_preview: Preview of compressed message structure
            with role and truncated content.

    Example:
        ```python
        async def review_compression(item: ProgressItem) -> InterruptDecision:
            if isinstance(item, MemoryCompressionComplete):
                if item.metrics.compression_ratio < 0.3:
                    return InterruptDecision.interrupt(
                        "Insufficient compression",
                        metadata={"required_ratio": 0.3}
                    )
            return InterruptDecision.proceed()

        event.set_interrupt_callback(review_compression)
        decision = await event.check_interrupt()
        ```

    """

    type: ProgressType = ProgressType.MEMORY_COMPRESSION_COMPLETE
    metrics: Any
    compressed_messages_preview: list[dict[str, str]] = Field(default_factory=list)
