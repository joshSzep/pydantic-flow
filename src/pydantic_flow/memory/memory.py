"""Conversation memory management for pydantic-flow.

This module provides ConversationMemory for managing message history
within flows and agents, compatible with pydantic-ai's message format.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextvars import ContextVar
from typing import TYPE_CHECKING
from typing import Protocol
from typing import runtime_checkable

if TYPE_CHECKING:
    from pydantic_ai import ModelMessage


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
    """

    def __init__(self, initial_messages: Sequence[ModelMessage] | None = None) -> None:
        """Initialize conversation memory.

        Args:
            initial_messages: Optional sequence of initial messages in
                            pydantic-ai ModelMessage format.

        """
        self._messages: list[ModelMessage] = (
            list(initial_messages) if initial_messages else []
        )

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

    def __repr__(self) -> str:
        """Return string representation of the memory.

        Returns:
            String showing message count.

        """
        return f"ConversationMemory({len(self._messages)} messages)"


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
