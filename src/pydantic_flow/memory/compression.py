"""Memory compression protocols and base classes for pydantic-flow.

This module provides the core abstractions for pluggable memory compression
strategies that manage conversation context when approaching LLM token limits.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING
from typing import Any
from typing import Protocol
from typing import runtime_checkable

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator

if TYPE_CHECKING:
    from pydantic_ai import ModelMessage


class CompressionMetrics(BaseModel):
    """Metrics from a compression operation.

    This model captures detailed information about a compression operation,
    enabling monitoring, analysis, and quality validation.

    Attributes:
        messages_before: Number of messages before compression.
        messages_after: Number of messages after compression.
        estimated_tokens_before: Estimated token count before compression.
        estimated_tokens_after: Estimated token count after compression.
        tokens_saved: Estimated tokens saved by compression (computed).
        compression_ratio: Ratio of after/before tokens (< 1.0 means compression).
        compression_strategy: Name of the compression strategy used.
        compression_time_ms: Time taken to compress in milliseconds.
        metadata: Strategy-specific metadata for additional context.

    Example:
        ```python
        metrics = CompressionMetrics(
            messages_before=50,
            messages_after=10,
            estimated_tokens_before=5000,
            estimated_tokens_after=1000,
            tokens_saved=4000,
            compression_ratio=0.2,
            compression_strategy="sliding_window",
            compression_time_ms=0.5,
            metadata={"dropped_messages": 40},
        )
        ```

    """

    messages_before: int = Field(ge=0)
    messages_after: int = Field(ge=0)
    estimated_tokens_before: int = Field(ge=0)
    estimated_tokens_after: int = Field(ge=0)
    tokens_saved: int = Field(ge=0)
    compression_ratio: float = Field(ge=0.0, le=1.0)
    compression_strategy: str
    compression_time_ms: float = Field(ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("compression_ratio")
    @classmethod
    def validate_compression_ratio(cls, v: float) -> float:
        """Validate compression ratio is between 0.0 and 1.0.

        Args:
            v: The compression ratio value.

        Returns:
            The validated compression ratio.

        """
        if not 0.0 <= v <= 1.0:
            msg = f"Compression ratio must be between 0.0 and 1.0, got {v}"
            raise ValueError(msg)
        return v

    @property
    def percentage_reduction(self) -> float:
        """Calculate the percentage reduction in tokens.

        Returns:
            Percentage of tokens saved (0-100).

        Example:
            ```python
            metrics = CompressionMetrics(...)
            print(f"Saved {metrics.percentage_reduction:.1f}% of tokens")
            ```

        """
        if self.estimated_tokens_before == 0:
            return 0.0
        return (self.tokens_saved / self.estimated_tokens_before) * 100.0

    @property
    def messages_removed(self) -> int:
        """Calculate the number of messages removed.

        Returns:
            Number of messages removed by compression.

        """
        return max(0, self.messages_before - self.messages_after)


@runtime_checkable
class MemoryCompressor(Protocol):
    """Protocol for memory compression strategies.

    All memory compressors must implement this protocol to be pluggable
    within the pydantic-flow memory management system.

    The protocol defines the contract for:
    1. Determining when compression should occur
    2. Performing the compression operation
    3. Providing a strategy name for identification

    Example:
        ```python
        class MyCompressor:
            @property
            def name(self) -> str:
                return "my_strategy"

            async def should_compress(
                self,
                messages: Sequence[ModelMessage],
                estimated_tokens: int,
            ) -> bool:
                return estimated_tokens > 8000

            async def compress(
                self,
                messages: Sequence[ModelMessage],
            ) -> tuple[list[ModelMessage], CompressionMetrics]:
                # Implement compression logic
                compressed = messages[-10:]  # Keep last 10
                metrics = CompressionMetrics(...)
                return compressed, metrics
        ```

    """

    async def should_compress(
        self,
        messages: Sequence[ModelMessage],
        estimated_tokens: int,
    ) -> bool:
        """Determine if compression should be triggered.

        This method is called before compression to decide whether
        the current message history needs to be compressed.

        Args:
            messages: Current message history.
            estimated_tokens: Estimated token count for the messages.

        Returns:
            True if compression should occur, False otherwise.

        """
        ...

    async def compress(
        self,
        messages: Sequence[ModelMessage],
    ) -> tuple[list[ModelMessage], CompressionMetrics]:
        """Compress the message history.

        This method performs the actual compression operation and
        returns both the compressed messages and detailed metrics.

        Args:
            messages: Messages to compress.

        Returns:
            Tuple of (compressed_messages, compression_metrics).

        Raises:
            Exception: If compression fails for any reason.

        """
        ...

    @property
    def name(self) -> str:
        """Name of this compression strategy.

        Returns:
            Human-readable strategy name (e.g., "sliding_window").

        """
        ...


class BaseMemoryCompressor(ABC):
    """Abstract base class for memory compressors with common utilities.

    This class provides a foundation for implementing compression strategies
    with shared functionality for token estimation, message partitioning,
    and threshold checking.

    Subclasses must implement:
    - `compress()`: The compression algorithm
    - `name` property: Strategy identifier

    Attributes:
        max_tokens: Maximum token limit before compression triggers.
        preserve_system_messages: Whether to always keep system messages.
        preserve_recent_messages: Number of recent messages to preserve.

    Example:
        ```python
        class SimpleCompressor(BaseMemoryCompressor):
            @property
            def name(self) -> str:
                return "simple"

            async def compress(self, messages):
                system, compressible, recent = self._partition_messages(messages)
                return system + recent, metrics
        ```

    """

    def __init__(
        self,
        *,
        max_tokens: int = 8000,
        preserve_system_messages: bool = True,
        preserve_recent_messages: int = 5,
    ) -> None:
        """Initialize base memory compressor.

        Args:
            max_tokens: Maximum token limit before compression. Default 8000.
            preserve_system_messages: Always keep system messages. Default True.
            preserve_recent_messages: Number of recent messages to preserve.
                Default 5.

        """
        self.max_tokens = max_tokens
        self.preserve_system_messages = preserve_system_messages
        self.preserve_recent_messages = preserve_recent_messages

    async def should_compress(
        self,
        messages: Sequence[ModelMessage],
        estimated_tokens: int,
    ) -> bool:
        """Determine if compression should be triggered based on token count.

        Default implementation compresses when exceeding max_tokens.
        Subclasses can override this method for custom compression triggers.

        Args:
            messages: Current message history.
            estimated_tokens: Estimated token count.

        Returns:
            True if estimated_tokens exceeds max_tokens.

        """
        return estimated_tokens > self.max_tokens

    @abstractmethod
    async def compress(
        self,
        messages: Sequence[ModelMessage],
    ) -> tuple[list[ModelMessage], CompressionMetrics]:
        """Compress the message history.

        Subclasses must implement their specific compression logic.

        Args:
            messages: Messages to compress.

        Returns:
            Tuple of (compressed_messages, compression_metrics).

        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this compression strategy.

        Subclasses must provide a unique strategy name.

        Returns:
            Strategy name string.

        """
        ...

    def _estimate_tokens(self, messages: Sequence[ModelMessage]) -> int:
        """Estimate token count for messages using simple heuristic.

        Uses approximation of ~4 characters per token. Subclasses can
        override with model-specific tokenizers for accuracy.

        Args:
            messages: Messages to estimate tokens for.

        Returns:
            Estimated token count.

        Example:
            ```python
            tokens = compressor._estimate_tokens(messages)
            print(f"Estimated {tokens} tokens")
            ```

        """
        total_chars = 0

        for msg in messages:
            # pydantic_ai messages have a 'parts' attribute
            parts = getattr(msg, "parts", None)
            if parts is not None:
                for part in parts:
                    # Extract content from different part types
                    content = getattr(part, "content", None)
                    if content is not None and isinstance(content, str):
                        total_chars += len(content)
                    # Also check for 'text' attribute (TextPart)
                    text = getattr(part, "text", None)
                    if text is not None and isinstance(text, str):
                        total_chars += len(text)
            else:
                # Fallback: try direct content attribute (for other message types)
                content = getattr(msg, "content", None)
                if content is not None:
                    if isinstance(content, str):
                        total_chars += len(content)
                    elif isinstance(content, list):
                        for part in content:
                            text = getattr(part, "text", None)
                            if text is not None:
                                total_chars += len(text)

        # Rough estimate: 4 characters per token
        return total_chars // 4

    def _partition_messages(
        self,
        messages: Sequence[ModelMessage],
    ) -> tuple[list[ModelMessage], list[ModelMessage], list[ModelMessage]]:
        """Partition messages into system, compressible, and recent.

        This utility method divides the message history into three categories:
        1. System messages (if preservation is enabled)
        2. Compressible messages (middle messages that can be compressed)
        3. Recent messages (last N messages to preserve)

        Args:
            messages: Messages to partition.

        Returns:
            Tuple of (system_messages, compressible_messages, recent_messages).

        Example:
            ```python
            system, compressible, recent = self._partition_messages(messages)
            print(f"System: {len(system)}, Compressible: {len(compressible)}, "
                  f"Recent: {len(recent)}")
            ```

        """
        system_msgs: list[ModelMessage] = []
        compressible_msgs: list[ModelMessage] = []
        recent_msgs: list[ModelMessage] = []

        for i, msg in enumerate(messages):
            # Last N messages are "recent"
            if i >= len(messages) - self.preserve_recent_messages:
                recent_msgs.append(msg)
            # System messages if preservation enabled
            elif self.preserve_system_messages and self._is_system_message(msg):
                system_msgs.append(msg)
            else:
                compressible_msgs.append(msg)

        return system_msgs, compressible_msgs, recent_msgs

    def _is_system_message(self, msg: ModelMessage) -> bool:
        """Check if message is a system message.

        Handles different ModelMessage structures from pydantic-ai.

        Args:
            msg: Message to check.

        Returns:
            True if message is a system message.

        Example:
            ```python
            if self._is_system_message(msg):
                print("This is a system message")
            ```

        """
        # pydantic-ai ModelMessage structure varies by type
        role = getattr(msg, "role", None)
        if role is not None:
            return role == "system"
        kind = getattr(msg, "kind", None)
        if kind is not None:
            return kind == "system"
        return False
