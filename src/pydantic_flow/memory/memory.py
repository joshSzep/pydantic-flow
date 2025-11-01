"""Conversation memory management for pydantic-flow.

This module provides ConversationMemory for managing message history
within flows and agents, compatible with pydantic-ai's message format.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
from contextvars import ContextVar
from typing import TYPE_CHECKING
from typing import Protocol
from typing import runtime_checkable

from pydantic_flow.checkpoints.types import SnapshotReason
from pydantic_flow.checkpoints.types import StateSnapshot
from pydantic_flow.checkpoints.types import generate_run_id
from pydantic_flow.checkpoints.types import generate_snapshot_id
from pydantic_flow.hitl.interrupts import InterruptionRequested
from pydantic_flow.memory.events import MemoryCompressionComplete
from pydantic_flow.memory.events import MemoryCompressionPending

if TYPE_CHECKING:
    from pydantic_ai import ModelMessage

    from pydantic_flow.memory.compression import CompressionMetrics
    from pydantic_flow.memory.compression import MemoryCompressor
    from pydantic_flow.streaming.base import ProgressItem


_memory_event_emitter: ContextVar[Callable[[ProgressItem], None] | None] = ContextVar(
    "_memory_event_emitter"
)


@runtime_checkable
class MemoryProtocol(Protocol):
    """Protocol for memory objects that can be used in context.

    Both ConversationMemory and ReadOnlyConversationMemory implement this.
    """

    def append(self, message: ModelMessage) -> None:
        """Append a message to the conversation history."""
        ...

    def extend(self, messages: Sequence[ModelMessage]) -> None:
        """Append multiple messages to the conversation history."""
        ...

    def get(self) -> list[ModelMessage]:
        """Get the full conversation history."""
        ...

    def clear(self) -> None:
        """Clear all messages from the conversation history."""
        ...

    def copy(self) -> ConversationMemory:
        """Create a deep copy of this conversation memory."""
        ...

    def __len__(self) -> int:
        """Return the number of messages in the conversation history."""
        ...


_active_flow_memory: ContextVar[MemoryProtocol | None] = ContextVar(  # type: ignore[assignment]
    "_active_flow_memory", default=None
)


class ReadOnlyMemoryError(Exception):
    """Raised when attempting to modify read-only conversation memory."""

    pass


class ConversationMemory:
    """Thread-safe conversation memory for managing message history.

    This class wraps pydantic-ai's ModelMessage format and provides methods
    for appending, retrieving, and managing conversation history.

    The memory is compatible with pydantic-ai's message_history parameter
    and can be used across nodes within a flow.

    Supports optional compression via MemoryCompressor protocol to manage
    context when approaching token limits.
    """

    def __init__(
        self,
        initial_messages: Sequence[ModelMessage] | None = None,
        compressor: MemoryCompressor | None = None,
    ) -> None:
        """Initialize conversation memory.

        Args:
            initial_messages: Optional sequence of initial messages in
                            pydantic-ai ModelMessage format.
            compressor: Optional MemoryCompressor for automatic context management.

        """
        self._messages: list[ModelMessage] = (
            list(initial_messages) if initial_messages else []
        )
        self._compressor: MemoryCompressor | None = compressor
        self._compression_history: list[CompressionMetrics] = []
        self._last_compression_rejected: bool = False

    def append(self, message: ModelMessage) -> None:
        """Append a message to the conversation history.

        Args:
            message: A pydantic-ai ModelMessage to append.

        """
        self._messages.append(message)

    def extend(self, messages: Sequence[ModelMessage]) -> None:
        """Append multiple messages to the conversation history.

        Args:
            messages: Sequence of pydantic-ai ModelMessages.

        """
        self._messages.extend(messages)

    def get(self) -> list[ModelMessage]:
        """Get the full conversation history.

        Returns:
            List of pydantic-ai ModelMessages.

        """
        return self._messages.copy()

    def clear(self) -> None:
        """Clear all messages from the conversation history."""
        self._messages.clear()

    def copy(self) -> ConversationMemory:
        """Create a deep copy of this conversation memory.

        Returns:
            New ConversationMemory instance with copied message history.

        """
        return ConversationMemory(initial_messages=self._messages)

    def __len__(self) -> int:
        """Return the number of messages in the conversation history.

        Returns:
            Number of messages.

        """
        return len(self._messages)

    async def maybe_compress(self) -> CompressionMetrics | None:
        """Check if compression is needed and perform it if so.

        This method implements the full compression flow with HITL interruption:
        1. Check if compression is needed via compressor.should_compress()
        2. Emit MemoryCompressionPending event (interruptible)
        3. Perform compression via compressor.compress()
        4. Emit MemoryCompressionComplete event (interruptible)
        5. Apply or rollback based on interrupt decisions

        Returns:
            CompressionMetrics if compression was performed and accepted,
            None otherwise.

        Raises:
            Various exceptions from compressor or interrupt handlers.

        """
        # No compression if no compressor configured
        if self._compressor is None:
            return None

        # Check if compression is needed
        # For BaseMemoryCompressor, we need to call the protocol's should_compress
        # which requires estimated_tokens. We'll use a helper for estimate.
        if hasattr(self._compressor, "_estimate_tokens"):
            estimated_tokens = self._compressor._estimate_tokens(self._messages)  # type: ignore[attr-defined]
        else:
            # Fallback: simple heuristic
            estimated_tokens = sum(len(str(m)) // 4 for m in self._messages)

        if not await self._compressor.should_compress(self._messages, estimated_tokens):
            return None

        # Get event emitter from context
        event_emitter = _memory_event_emitter.get(None)
        if event_emitter is None:
            # No emitter context - skip HITL flow, just compress directly
            compressed, metrics = await self._compressor.compress(self._messages)
            self._messages = compressed
            self._compression_history.append(metrics)
            self._last_compression_rejected = False
            return metrics

        # Create and emit MemoryCompressionPending event
        pending_event = MemoryCompressionPending(
            estimated_tokens=estimated_tokens,
            message_count=len(self._messages),
            compressor_name=self._compressor.name,
            compression_reason=(
                f"Message count ({len(self._messages)}) approaching limits"
            ),
            metadata={
                "current_token_estimate": estimated_tokens,
                "compressor_type": type(self._compressor).__name__,
            },
        )

        # Emit pending event
        event_emitter(pending_event)

        # Check if compression was rejected at pending stage
        if pending_event.interrupt_callback is not None:
            decision = await pending_event.check_interrupt()
            if decision.should_interrupt:
                self._last_compression_rejected = True
                # Check for replacement compressor
                if decision.replacement_value is not None:
                    # User provided alternative compressor - use it instead
                    self._compressor = decision.replacement_value
                    # Recursively try with new compressor
                    return await self.maybe_compress()
                # No replacement - create checkpoint and abort
                checkpoint = StateSnapshot(
                    snapshot_id=generate_snapshot_id(),
                    run_id=generate_run_id(),
                    wave_number=0,
                    state_hash="",
                    next_frontier=[],
                    routing_ended=False,
                    reason=SnapshotReason.HITL_INTERRUPT,
                    interrupted_node_id="memory_compression",
                    metadata={"compression_rejected": True},
                )
                raise InterruptionRequested(checkpoint, decision)

        # Store original messages for potential rollback
        original_messages = self._messages.copy()

        # Perform compression
        compressed, metrics = await self._compressor.compress(self._messages)

        # Create preview of compressed messages
        preview_max_len = 100
        preview = []
        for msg in compressed[:5]:  # First 5 messages
            content = self._extract_content(msg)
            truncated_content = (
                content[:preview_max_len] + "..."
                if len(content) > preview_max_len
                else content
            )
            preview.append({
                "role": self._extract_role(msg),
                "content": truncated_content,
            })

        # Create and emit MemoryCompressionComplete event
        complete_event = MemoryCompressionComplete(
            metrics=metrics,
            compressed_messages_preview=preview,
        )

        # Emit complete event
        event_emitter(complete_event)

        # Check if compression result was rejected
        if complete_event.interrupt_callback is not None:
            decision = await complete_event.check_interrupt()
            if decision.should_interrupt:
                self._last_compression_rejected = True
                # Check for replacement messages
                if decision.replacement_value is not None:
                    # User provided custom compressed messages
                    self._messages = decision.replacement_value
                    self._compression_history.append(metrics)
                    return metrics
                # No replacement - rollback to original
                self._messages = original_messages
                checkpoint = StateSnapshot(
                    snapshot_id=generate_snapshot_id(),
                    run_id=generate_run_id(),
                    wave_number=0,
                    state_hash="",
                    next_frontier=[],
                    routing_ended=False,
                    reason=SnapshotReason.HITL_INTERRUPT,
                    interrupted_node_id="memory_compression",
                    metadata={"compression_rejected": True, "rollback": True},
                )
                raise InterruptionRequested(checkpoint, decision)

        # Compression accepted - apply it
        self._messages = compressed
        self._compression_history.append(metrics)
        self._last_compression_rejected = False
        return metrics

    def _extract_role(self, message: ModelMessage) -> str:
        """Extract role from a message.

        Args:
            message: ModelMessage instance.

        Returns:
            Role string (user, assistant, system, etc.).

        """
        # Handle both dict and object messages
        if isinstance(message, dict):
            return message.get("role", "unknown")
        return getattr(message, "role", "unknown")

    def _extract_content(self, message: ModelMessage) -> str:
        """Extract content from a message as string.

        Args:
            message: ModelMessage instance.

        Returns:
            Content as string.

        """
        # Handle both dict and object messages
        if isinstance(message, dict):
            content = message.get("content")
        else:
            content = getattr(message, "content", None)

        # Handle different content types
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # Extract text from content blocks
            texts = []
            for item in content:
                if isinstance(item, dict):
                    texts.append(str(item.get("text", "")))
                else:
                    texts.append(str(item))
            return " ".join(texts)
        return str(content) if content else ""

    def __repr__(self) -> str:
        """Return string representation of the memory.

        Returns:
            String showing message count.

        """
        return f"ConversationMemory({len(self._messages)} messages)"

    @property
    def compression_history(self) -> list[CompressionMetrics]:
        """Get the history of compression operations.

        Returns:
            List of CompressionMetrics from past compressions.

        """
        return self._compression_history.copy()

    @property
    def last_compression_rejected(self) -> bool:
        """Check if the last compression attempt was rejected.

        Returns:
            True if last compression was rejected via interrupt callback.

        """
        return self._last_compression_rejected


class ReadOnlyConversationMemory:
    """Read-only wrapper for ConversationMemory.

    Provides read access to conversation history but prevents modifications.
    Useful for sub-flows that need context but shouldn't modify parent memory.
    """

    def __init__(self, memory: ConversationMemory) -> None:
        """Initialize read-only wrapper.

        Args:
            memory: The ConversationMemory instance to wrap.

        """
        self._memory = memory

    def append(self, message: ModelMessage) -> None:
        """Raise error on modification attempt.

        Args:
            message: Message that would be appended.

        Raises:
            ReadOnlyMemoryError: Always, as memory is read-only.

        """
        msg = "Cannot append to read-only conversation memory"
        raise ReadOnlyMemoryError(msg)

    def extend(self, messages: Sequence[ModelMessage]) -> None:
        """Raise error on modification attempt.

        Args:
            messages: Messages that would be extended.

        Raises:
            ReadOnlyMemoryError: Always, as memory is read-only.

        """
        msg = "Cannot extend read-only conversation memory"
        raise ReadOnlyMemoryError(msg)

    def get(self) -> list[ModelMessage]:
        """Get the full conversation history (read-only).

        Returns:
            Copy of conversation messages.

        """
        return self._memory.get()

    def clear(self) -> None:
        """Raise error on modification attempt.

        Raises:
            ReadOnlyMemoryError: Always, as memory is read-only.

        """
        msg = "Cannot clear read-only conversation memory"
        raise ReadOnlyMemoryError(msg)

    def copy(self) -> ConversationMemory:
        """Create a deep copy of the underlying memory.

        Returns:
            New ConversationMemory instance with copied message history.

        """
        return self._memory.copy()

    def __len__(self) -> int:
        """Return the number of messages in the conversation history.

        Returns:
            Number of messages.

        """
        return len(self._memory)

    def __repr__(self) -> str:
        """Return string representation of the read-only memory.

        Returns:
            String showing it's read-only and message count.

        """
        return f"ReadOnlyConversationMemory({len(self._memory)} messages)"
