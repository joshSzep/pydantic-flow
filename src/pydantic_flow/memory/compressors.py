"""Built-in memory compression strategies for conversation history management.

This module provides three concrete compression strategies:
- SlidingWindowCompressor: Fast, simple, keeps recent messages
- SummarizationCompressor: LLM-based summarization for quality
- HybridCompressor: Adaptive strategy selection based on context
"""

from collections.abc import Sequence
from time import perf_counter
from typing import Annotated
from typing import Any
from typing import ClassVar

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator

from pydantic_flow.memory.compression import BaseMemoryCompressor
from pydantic_flow.memory.compression import CompressionMetrics

# Type alias for the agent - allows any object with a run() method in tests
AgentType = Annotated[Any, "Agent[None, str]"]


class SlidingWindowCompressor(BaseModel, BaseMemoryCompressor):
    """Keep system messages and N most recent messages, drop the middle.

    Fast, simple compression that preserves conversation context without LLM calls.
    Ideal for real-time applications where speed matters more than semantic quality.

    The compressor partitions messages into three groups:
    1. System messages (always preserved if preserve_system_messages=True)
    2. Compressible middle messages (dropped during compression)
    3. Recent messages (always preserved, controlled by window_size)

    Example:
        ```python
        compressor = SlidingWindowCompressor(
            max_tokens=4000,
            window_size=10,
            preserve_system_messages=True
        )

        # Original: [system, msg1, msg2, ..., msg50]
        # After: [system, msg41, msg42, ..., msg50]
        compressed, metrics = await compressor.compress(messages)
        ```

    Attributes:
        window_size: Number of recent messages to preserve (default: 10)
        max_tokens: Token limit triggering compression (default: 4000)
        preserve_system_messages: Whether to keep system messages (default: True)
        preserve_recent_messages: Window size for recent preservation (default: 5)

    """

    window_size: int = Field(default=10, ge=1)
    max_tokens: int = Field(default=4000, ge=1)
    preserve_system_messages: bool = True
    preserve_recent_messages: int = Field(default=5, ge=0)

    @field_validator("window_size")
    @classmethod
    def _validate_window_size(cls, v: int) -> int:
        """Ensure window_size is positive."""
        if v < 1:
            msg = "window_size must be at least 1"
            raise ValueError(msg)
        return v

    @property
    def name(self) -> str:
        """Return the name of this compression strategy."""
        return f"sliding_window_{self.window_size}"

    async def compress(
        self, messages: Sequence[Any]
    ) -> tuple[list[Any], CompressionMetrics]:
        """Compress messages by keeping system and recent messages only.

        Args:
            messages: Full conversation history to compress

        Returns:
            Tuple of (compressed_messages, compression_metrics)

        Example:
            ```python
            compressed, metrics = await compressor.compress(messages)
            print(f"Reduced from {metrics.messages_before} to {metrics.messages_after}")
            print(f"Saved ~{metrics.tokens_saved} tokens")
            ```

        """
        start_time = perf_counter()

        # Partition using window_size as the recent threshold
        system_msgs: list[Any] = []
        recent_msgs: list[Any] = []

        for i, msg in enumerate(messages):
            # Last window_size messages are "recent"
            if i >= len(messages) - self.window_size:
                recent_msgs.append(msg)
            # System messages if preservation enabled
            elif self.preserve_system_messages and self._is_system_msg(msg):
                system_msgs.append(msg)
            # Everything else is dropped in sliding window

        compressed = system_msgs + recent_msgs

        # Calculate metrics
        tokens_before = self._estimate_tokens(messages)
        tokens_after = self._estimate_tokens(compressed)
        compression_time = (perf_counter() - start_time) * 1000

        metrics = CompressionMetrics(
            messages_before=len(messages),
            messages_after=len(compressed),
            estimated_tokens_before=tokens_before,
            estimated_tokens_after=tokens_after,
            tokens_saved=tokens_before - tokens_after,
            compression_ratio=len(compressed) / len(messages) if messages else 1.0,
            compression_strategy=self.name,
            compression_time_ms=compression_time,
            metadata={
                "window_size": self.window_size,
                "system_messages_preserved": len(system_msgs),
                "messages_dropped": len(messages) - len(compressed),
            },
        )

        return compressed, metrics

    def _is_system_msg(self, msg: Any) -> bool:
        """Check if message is a system message (handles dicts and objects).

        Args:
            msg: Message to check.

        Returns:
            True if message is a system message.

        """
        # Handle dict
        if isinstance(msg, dict):
            return msg.get("role") == "system" or msg.get("kind") == "system"

        # Handle object with attributes
        role = getattr(msg, "role", None)
        if role is not None:
            return role == "system"
        kind = getattr(msg, "kind", None)
        if kind is not None and kind == "system":
            return True

        # For pydantic_ai messages, check if any part is a SystemPromptPart
        parts = getattr(msg, "parts", None)
        if parts is not None:
            for part in parts:
                part_type_name = type(part).__name__
                if part_type_name == "SystemPromptPart":
                    return True

        return False


class SummarizationCompressor(BaseModel, BaseMemoryCompressor):
    """Use an LLM agent to summarize compressible messages into a summary.

    Higher-quality compression that preserves semantic meaning via LLMs.
    The summarizer creates a new system message with the compressed context.

    Compression flow:
    1. Partition messages into system/compressible/recent
    2. Format compressible messages for summarization
    3. Run agent with custom prompt to generate summary
    4. Create new system message with summary
    5. Return system + summary + recent messages

    Example:
        ```python
        from pydantic_ai import Agent

        summarizer_agent = Agent("openai:gpt-4o-mini")
        compressor = SummarizationCompressor(
            agent=summarizer_agent,
            max_tokens=4000,
            preserve_recent_messages=5,
            prompt_template="Summarize this conversation: {messages}"
        )

        compressed, metrics = await compressor.compress(messages)
        ```

    Attributes:
        agent: Pydantic AI agent for summarization
        prompt_template: Template for summarization prompt (default provided)
        max_summary_tokens: Max tokens for the summary (default: 500)
        max_tokens: Token limit triggering compression (default: 4000)
        preserve_system_messages: Whether to keep system messages (default: True)
        preserve_recent_messages: Number of recent messages to keep (default: 5)

    """

    MIN_SUMMARY_TOKENS: ClassVar[int] = 50

    agent: AgentType
    prompt_template: str = Field(
        default=(
            "Summarize the following conversation history concisely, "
            "preserving key context, decisions, and important details:\n\n{messages}"
        )
    )
    max_summary_tokens: int = Field(default=500, ge=50)
    max_tokens: int = Field(default=4000, ge=1)
    preserve_system_messages: bool = True
    preserve_recent_messages: int = Field(default=5, ge=0)

    @field_validator("max_summary_tokens")
    @classmethod
    def _validate_max_summary_tokens(cls, v: int) -> int:
        """Ensure max_summary_tokens is reasonable."""
        if v < cls.MIN_SUMMARY_TOKENS:
            msg = f"max_summary_tokens must be at least {cls.MIN_SUMMARY_TOKENS}"
            raise ValueError(msg)
        return v

    @property
    def name(self) -> str:
        """Return the name of this compression strategy."""
        return "summarization"

    def _format_messages_for_summary(self, messages: Sequence[Any]) -> str:
        """Format messages into a text block for summarization.

        Args:
            messages: Messages to format

        Returns:
            Formatted string suitable for LLM summarization

        """
        lines = []
        for msg in messages:
            # Handle both dict and object messages
            if isinstance(msg, dict):
                role = msg.get("role", "unknown")
                content = msg.get("content")
            else:
                role = getattr(msg, "role", "unknown")
                content = getattr(msg, "content", None)

            # Handle different content types
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                # Extract text from content blocks
                text = " ".join(
                    str(item.get("text", "")) if isinstance(item, dict) else str(item)
                    for item in content
                )
            else:
                text = str(content) if content else ""

            if text:
                lines.append(f"{role}: {text}")

        return "\n".join(lines)

    def _create_summary_message(self, summary: str) -> dict[str, Any]:
        """Create a system message containing the summary.

        Args:
            summary: Summarized conversation text

        Returns:
            Dictionary representing a system message with the summary

        """
        return {
            "role": "system",
            "content": f"[Conversation Summary]\n{summary}",
            "kind": "request",
        }

    async def compress(
        self, messages: Sequence[Any]
    ) -> tuple[list[Any], CompressionMetrics]:
        """Compress messages using LLM-based summarization.

        Args:
            messages: Full conversation history to compress

        Returns:
            Tuple of (compressed_messages, compression_metrics)

        Raises:
            RuntimeError: If summarization agent fails

        Example:
            ```python
            try:
                compressed, metrics = await compressor.compress(messages)
                print(f"Compression quality: {metrics.percentage_reduction}%")
            except RuntimeError as e:
                print(f"Summarization failed: {e}")
            ```

        """
        start_time = perf_counter()

        system, compressible, recent = self._partition_messages(messages)

        # Nothing to compress
        if not compressible:
            tokens_before = self._estimate_tokens(messages)
            return list(messages), CompressionMetrics(
                messages_before=len(messages),
                messages_after=len(messages),
                estimated_tokens_before=tokens_before,
                estimated_tokens_after=tokens_before,
                tokens_saved=0,
                compression_ratio=1.0,
                compression_strategy=self.name,
                compression_time_ms=(perf_counter() - start_time) * 1000,
                metadata={"reason": "no_compressible_messages"},
            )

        # Format and summarize
        formatted = self._format_messages_for_summary(compressible)
        prompt = self.prompt_template.format(messages=formatted)

        try:
            result = await self.agent.run(prompt)
            summary = str(result.output)
        except Exception as e:
            msg = f"Summarization agent failed: {e}"
            raise RuntimeError(msg) from e

        # Create compressed message list
        summary_message = self._create_summary_message(summary)
        compressed = [*system, summary_message, *recent]

        # Calculate metrics
        tokens_before = self._estimate_tokens(messages)
        tokens_after = self._estimate_tokens(list(compressed))
        compression_time = (perf_counter() - start_time) * 1000

        metrics = CompressionMetrics(
            messages_before=len(messages),
            messages_after=len(compressed),
            estimated_tokens_before=tokens_before,
            estimated_tokens_after=tokens_after,
            tokens_saved=tokens_before - tokens_after,
            compression_ratio=len(compressed) / len(messages) if messages else 1.0,
            compression_strategy=self.name,
            compression_time_ms=compression_time,
            metadata={
                "system_messages_preserved": len(system),
                "messages_summarized": len(compressible),
                "recent_messages_preserved": len(recent),
                "summary_length": len(summary),
                "summary_tokens": len(summary) // 4,  # Rough estimate
            },
        )

        return compressed, metrics


class HybridCompressor(BaseModel, BaseMemoryCompressor):
    """Adaptive compressor that selects strategy based on context.

    Intelligently chooses between sliding window and summarization based on:
    - Number of messages to compress
    - Available time/resources
    - Quality requirements

    Strategy selection heuristics:
    - Few messages (< threshold): Use sliding window (fast)
    - Many messages (>= threshold): Use summarization (quality)
    - Fallback to sliding window if summarization unavailable

    Example:
        ```python
        from pydantic_ai import Agent

        summarizer = Agent("openai:gpt-4o-mini")
        compressor = HybridCompressor(
            summarizer_agent=summarizer,
            max_tokens=4000,
            summarization_threshold=20,  # Use LLM if >= 20 messages
            window_size=10,
        )

        # Automatically selects best strategy
        compressed, metrics = await compressor.compress(messages)
        print(f"Strategy used: {metrics.compression_strategy}")
        ```

    Attributes:
        summarizer_agent: Optional agent for summarization
            (if None, always use sliding window)
        summarization_threshold: Min messages to trigger summarization
            (default: 15)
        window_size: Window size for sliding window fallback (default: 10)
        max_tokens: Token limit triggering compression (default: 4000)
        preserve_system_messages: Whether to keep system messages (default: True)
        preserve_recent_messages: Number of recent messages to keep (default: 5)

    """

    summarizer_agent: AgentType | None = None
    summarization_threshold: int = Field(default=15, ge=1)
    window_size: int = Field(default=10, ge=1)
    max_tokens: int = Field(default=4000, ge=1)
    preserve_system_messages: bool = True
    preserve_recent_messages: int = Field(default=5, ge=0)

    @field_validator("summarization_threshold")
    @classmethod
    def _validate_summarization_threshold(cls, v: int) -> int:
        """Ensure summarization_threshold is positive."""
        if v < 1:
            msg = "summarization_threshold must be at least 1"
            raise ValueError(msg)
        return v

    @field_validator("window_size")
    @classmethod
    def _validate_window_size(cls, v: int) -> int:
        """Ensure window_size is positive."""
        if v < 1:
            msg = "window_size must be at least 1"
            raise ValueError(msg)
        return v

    @property
    def name(self) -> str:
        """Return the name of this compression strategy."""
        return "hybrid"

    def _select_strategy(self, messages: Sequence[Any]) -> str:
        """Select compression strategy based on message count and agent availability.

        Args:
            messages: Messages to compress

        Returns:
            Strategy name: "sliding_window" or "summarization"

        """
        _, compressible, _ = self._partition_messages(messages)
        compressible_count = len(compressible)

        # No agent available - must use sliding window
        if self.summarizer_agent is None:
            return "sliding_window"

        # Too few messages - sliding window is sufficient
        if compressible_count < self.summarization_threshold:
            return "sliding_window"

        # Many messages - use summarization for quality
        return "summarization"

    async def compress(
        self, messages: Sequence[Any]
    ) -> tuple[list[Any], CompressionMetrics]:
        """Compress messages using the most appropriate strategy.

        Args:
            messages: Full conversation history to compress

        Returns:
            Tuple of (compressed_messages, compression_metrics)

        Example:
            ```python
            compressed, metrics = await compressor.compress(messages)

            if metrics.compression_strategy == "summarization":
                print("Used LLM summarization for quality")
            else:
                print("Used sliding window for speed")
            ```

        """
        start_time = perf_counter()
        strategy = self._select_strategy(messages)

        # Delegate to appropriate strategy
        if strategy == "sliding_window":
            delegate = SlidingWindowCompressor(
                max_tokens=self.max_tokens,
                preserve_system_messages=self.preserve_system_messages,
                preserve_recent_messages=self.preserve_recent_messages,
                window_size=self.window_size,  # SlidingWindow-specific field
            )
        else:  # summarization
            delegate = SummarizationCompressor(
                agent=self.summarizer_agent,  # type: ignore[arg-type]
                max_tokens=self.max_tokens,
                preserve_system_messages=self.preserve_system_messages,
                preserve_recent_messages=self.preserve_recent_messages,
            )

        compressed, metrics = await delegate.compress(messages)

        # Update metrics to reflect hybrid strategy
        compression_time = (perf_counter() - start_time) * 1000
        metrics = CompressionMetrics(
            messages_before=metrics.messages_before,
            messages_after=metrics.messages_after,
            estimated_tokens_before=metrics.estimated_tokens_before,
            estimated_tokens_after=metrics.estimated_tokens_after,
            tokens_saved=metrics.tokens_saved,
            compression_ratio=metrics.compression_ratio,
            compression_strategy=self.name,
            compression_time_ms=compression_time,
            metadata={
                "selected_strategy": strategy,
                "summarization_threshold": self.summarization_threshold,
                "window_size": self.window_size,
                **metrics.metadata,
            },
        )

        return compressed, metrics
