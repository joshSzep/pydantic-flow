"""Memory Compression - Custom Compressor Implementation.

This example demonstrates how to implement a custom compression strategy
by conforming to the MemoryCompressor protocol.

Key concepts:
1. Implementing the MemoryCompressor protocol
2. should_compress() - Determine when compression is needed
3. compress() - Execute compression logic
4. name property - Identify your strategy
5. Using Pydantic BaseModel for configuration

Run with: uv run python examples/memory_compression_custom.py
"""

import asyncio
from collections.abc import Sequence
from time import perf_counter
from typing import Any

from pydantic import BaseModel
from pydantic import Field

from pydantic_flow import CompressionMetrics
from pydantic_flow import Flow
from pydantic_flow import MemoryConfig
from pydantic_flow.memory.compression import BaseMemoryCompressor


class ChatInput(BaseModel):
    """User message input."""

    message: str


class ChatOutput(BaseModel):
    """Agent response output."""

    response: str


# ============================================================================
# Example 1: Priority-Based Compressor
# ============================================================================


class PriorityCompressor(BaseModel, BaseMemoryCompressor):
    """Compressor that keeps messages based on priority tags.

    Messages can be tagged with priority levels. During compression,
    low-priority messages are removed first, preserving high-priority context.

    Example message format:
        {"role": "user", "content": "...", "priority": 3}

    Attributes:
        max_tokens: Token limit triggering compression
        priority_threshold: Minimum priority to keep (default: 2)
        preserve_recent_messages: Always keep N recent messages (default: 3)

    """

    max_tokens: int = Field(default=4000, ge=1)
    priority_threshold: int = Field(default=2, ge=1, le=5)
    preserve_recent_messages: int = Field(default=3, ge=0)

    @property
    def name(self) -> str:
        """Return compressor name."""
        return f"priority_p{self.priority_threshold}"

    async def compress(
        self, messages: Sequence[Any]
    ) -> tuple[list[Any], CompressionMetrics]:
        """Compress by removing low-priority messages."""
        start_time = perf_counter()

        # Calculate initial metrics
        tokens_before = self._estimate_tokens(messages)
        messages_before = len(messages)

        # Always preserve recent messages
        if len(messages) <= self.preserve_recent_messages:
            compressed = list(messages)
        else:
            # Split into old and recent
            recent_count = self.preserve_recent_messages
            old_messages = messages[:-recent_count] if recent_count > 0 else messages
            recent_messages = messages[-recent_count:] if recent_count > 0 else []

            # Filter old messages by priority
            filtered = [
                msg
                for msg in old_messages
                if self._get_priority(msg) >= self.priority_threshold
            ]

            # Combine filtered old + recent
            compressed = filtered + list(recent_messages)

        # Calculate final metrics
        tokens_after = self._estimate_tokens(compressed)
        compression_time_ms = (perf_counter() - start_time) * 1000

        metrics = CompressionMetrics(
            messages_before=messages_before,
            messages_after=len(compressed),
            estimated_tokens_before=tokens_before,
            estimated_tokens_after=tokens_after,
            tokens_saved=tokens_before - tokens_after,
            compression_ratio=(
                tokens_after / tokens_before if tokens_before > 0 else 1.0
            ),
            compression_time_ms=compression_time_ms,
            compression_strategy=self.name,
        )

        return compressed, metrics

    def _get_priority(self, msg: Any) -> int:
        """Extract priority from message (default: 3)."""
        if isinstance(msg, dict):
            return msg.get("priority", 3)
        return getattr(msg, "priority", 3)


# ============================================================================
# Example 2: Token-Budget Compressor
# ============================================================================


class TokenBudgetCompressor(BaseModel, BaseMemoryCompressor):
    """Compressor that enforces strict token budget.

    Keeps messages from newest to oldest until token budget is exhausted.
    More precise than sliding window as it respects actual token counts.

    Attributes:
        token_budget: Maximum tokens to preserve (default: 3000)
        preserve_system_messages: Keep system messages regardless (default: True)

    """

    token_budget: int = Field(default=3000, ge=100)
    preserve_system_messages: bool = True
    max_tokens: int = Field(default=4000, ge=1)

    @property
    def name(self) -> str:
        """Return compressor name."""
        return f"token_budget_{self.token_budget}"

    async def compress(
        self, messages: Sequence[Any]
    ) -> tuple[list[Any], CompressionMetrics]:
        """Compress by enforcing token budget."""
        start_time = perf_counter()

        tokens_before = self._estimate_tokens(messages)
        messages_before = len(messages)

        # Separate system and non-system messages
        system_msgs = []
        non_system_msgs = []

        for msg in messages:
            if self.preserve_system_messages and self._is_system(msg):
                system_msgs.append(msg)
            else:
                non_system_msgs.append(msg)

        # Calculate system message tokens
        system_tokens = self._estimate_tokens(system_msgs) if system_msgs else 0
        remaining_budget = self.token_budget - system_tokens

        # Add non-system messages from newest until budget exhausted
        kept_msgs = []
        used_tokens = 0

        for msg in reversed(non_system_msgs):
            msg_tokens = self._estimate_tokens([msg])
            if used_tokens + msg_tokens <= remaining_budget:
                kept_msgs.insert(0, msg)  # Maintain order
                used_tokens += msg_tokens
            else:
                break  # Budget exhausted

        # Combine system + kept messages
        compressed = system_msgs + kept_msgs

        tokens_after = self._estimate_tokens(compressed)
        compression_time_ms = (perf_counter() - start_time) * 1000

        metrics = CompressionMetrics(
            messages_before=messages_before,
            messages_after=len(compressed),
            estimated_tokens_before=tokens_before,
            estimated_tokens_after=tokens_after,
            tokens_saved=tokens_before - tokens_after,
            compression_ratio=(
                tokens_after / tokens_before if tokens_before > 0 else 1.0
            ),
            compression_time_ms=compression_time_ms,
            compression_strategy=self.name,
        )

        return compressed, metrics

    def _is_system(self, msg: Any) -> bool:
        """Check if message is a system message."""
        if isinstance(msg, dict):
            return msg.get("role") == "system"
        return getattr(msg, "role", None) == "system"


# ============================================================================
# Example 3: Semantic Clustering Compressor
# ============================================================================


class SemanticClusterCompressor(BaseModel, BaseMemoryCompressor):
    """Compressor that groups similar messages.

    Groups messages into semantic clusters and keeps one representative
    from each cluster, reducing redundancy.

    Note: This is a simplified example. Real implementation would use
    embeddings and proper clustering algorithms.

    Attributes:
        max_tokens: Token limit triggering compression
        target_cluster_count: Target number of clusters (default: 5)
        preserve_recent_messages: Always keep N recent messages (default: 5)

    """

    max_tokens: int = Field(default=4000, ge=1)
    target_cluster_count: int = Field(default=5, ge=1)
    preserve_recent_messages: int = Field(default=5, ge=0)

    @property
    def name(self) -> str:
        """Return compressor name."""
        return f"semantic_cluster_{self.target_cluster_count}"

    async def compress(
        self, messages: Sequence[Any]
    ) -> tuple[list[Any], CompressionMetrics]:
        """Compress by clustering similar messages."""
        start_time = perf_counter()

        tokens_before = self._estimate_tokens(messages)
        messages_before = len(messages)

        # Split into old (compressible) and recent (always keep)
        if len(messages) <= self.preserve_recent_messages:
            compressed = list(messages)
        else:
            recent_count = self.preserve_recent_messages
            old_messages = (
                list(messages[:-recent_count]) if recent_count > 0 else list(messages)
            )
            recent_messages = list(messages[-recent_count:]) if recent_count > 0 else []

            # Simplified clustering: group by message length
            # Real implementation would use embeddings + k-means
            clusters = self._simple_cluster(old_messages)

            # Keep one representative per cluster
            representatives = [cluster[0] for cluster in clusters]

            compressed = representatives + recent_messages

        tokens_after = self._estimate_tokens(compressed)
        compression_time_ms = (perf_counter() - start_time) * 1000

        metrics = CompressionMetrics(
            messages_before=messages_before,
            messages_after=len(compressed),
            estimated_tokens_before=tokens_before,
            estimated_tokens_after=tokens_after,
            tokens_saved=tokens_before - tokens_after,
            compression_ratio=(
                tokens_after / tokens_before if tokens_before > 0 else 1.0
            ),
            compression_time_ms=compression_time_ms,
            compression_strategy=self.name,
        )

        return compressed, metrics

    def _simple_cluster(self, messages: list[Any]) -> list[list[Any]]:
        """Simple clustering by message length (demo only)."""
        if not messages:
            return []

        # Group by content length buckets
        buckets: dict[int, list[Any]] = {}
        for msg in messages:
            content = self._get_content(msg)
            length = len(str(content))
            bucket = length // 50  # 50-char buckets
            buckets.setdefault(bucket, []).append(msg)

        # Return cluster count limited to target
        clusters = list(buckets.values())
        if len(clusters) > self.target_cluster_count:
            # Simple merging: combine smallest clusters
            clusters.sort(key=len)
            while len(clusters) > self.target_cluster_count:
                smallest = clusters.pop(0)
                clusters[0].extend(smallest)

        return clusters

    def _get_content(self, msg: Any) -> str:
        """Extract content from message."""
        if isinstance(msg, dict):
            return str(msg.get("content", ""))
        return str(getattr(msg, "content", ""))


# ============================================================================
# Demonstration Functions
# ============================================================================


async def demonstrate_priority_compressor():
    """Demonstrate priority-based compression."""
    print("\n" + "=" * 70)
    print("Example 1: Priority-Based Compressor")
    print("=" * 70)

    compressor = PriorityCompressor(
        max_tokens=100,
        priority_threshold=3,  # Keep priority >= 3
        preserve_recent_messages=2,
    )

    _ = Flow[ChatInput, ChatOutput](
        input_type=ChatInput,
        output_type=ChatOutput,
        memory_config=MemoryConfig(
            enable_conversation_memory=True,
            compressor=compressor,
            emit_compression_events=True,
        ),
    )

    print(f"\nCompressor: {compressor.name}")
    print(f"Priority threshold: {compressor.priority_threshold}")
    print(f"Preserve recent: {compressor.preserve_recent_messages}")

    print("\nBehavior:")
    print("  • Low-priority messages (priority < 3) are removed first")
    print("  • Last 2 messages always preserved regardless of priority")
    print("  • High-priority context retained even if old")

    print("\nUse cases:")
    print("  • Keeping important system announcements")
    print("  • Preserving critical user preferences")
    print("  • Maintaining essential context clues")


async def demonstrate_token_budget_compressor():
    """Demonstrate token budget enforcement."""
    print("\n" + "=" * 70)
    print("Example 2: Token-Budget Compressor")
    print("=" * 70)

    compressor = TokenBudgetCompressor(
        token_budget=1000,
        max_tokens=1500,
        preserve_system_messages=True,
    )

    _ = Flow[ChatInput, ChatOutput](
        input_type=ChatInput,
        output_type=ChatOutput,
        memory_config=MemoryConfig(
            enable_conversation_memory=True,
            compressor=compressor,
            emit_compression_events=True,
        ),
    )

    print(f"\nCompressor: {compressor.name}")
    print(f"Token budget: {compressor.token_budget}")
    print(f"Max tokens: {compressor.max_tokens}")

    print("\nBehavior:")
    print("  • Enforces strict token limit (1000 tokens)")
    print("  • Adds messages from newest until budget exhausted")
    print("  • System messages preserved regardless of budget")
    print("  • More precise than simple message-count windows")

    print("\nUse cases:")
    print("  • Fine-grained control over context size")
    print("  • Predictable LLM API costs")
    print("  • Optimizing for specific model context limits")


async def demonstrate_semantic_cluster_compressor():
    """Demonstrate semantic clustering compression."""
    print("\n" + "=" * 70)
    print("Example 3: Semantic Clustering Compressor")
    print("=" * 70)

    compressor = SemanticClusterCompressor(
        max_tokens=200,
        target_cluster_count=3,
        preserve_recent_messages=5,
    )

    _ = Flow[ChatInput, ChatOutput](
        input_type=ChatInput,
        output_type=ChatOutput,
        memory_config=MemoryConfig(
            enable_conversation_memory=True,
            compressor=compressor,
            emit_compression_events=True,
        ),
    )

    print(f"\nCompressor: {compressor.name}")
    print(f"Target clusters: {compressor.target_cluster_count}")
    print(f"Preserve recent: {compressor.preserve_recent_messages}")

    print("\nBehavior:")
    print("  • Groups similar messages into clusters")
    print("  • Keeps one representative from each cluster")
    print("  • Reduces redundancy while preserving diversity")
    print("  • Recent messages always preserved")

    print("\nUse cases:")
    print("  • Long conversations with repetitive topics")
    print("  • Maintaining topic diversity in context")
    print("  • Reducing redundant follow-up questions")

    print("\nNote: This example uses simplified clustering.")
    print("Production implementation would use:")
    print("  • Embedding models (OpenAI, Sentence Transformers)")
    print("  • K-means or hierarchical clustering")
    print("  • Semantic similarity metrics")


async def implementation_guidelines():
    """Display guidelines for implementing custom compressors."""
    print("\n" + "=" * 70)
    print("Custom Compressor Implementation Guidelines")
    print("=" * 70)

    print("\n1. Protocol Requirements:")
    print("   ✓ Inherit from BaseModel and BaseMemoryCompressor")
    print("   ✓ Implement: name property (returns str)")
    print("   ✓ Implement: compress(messages) → (list, CompressionMetrics)")
    print("   ✓ Implement: should_compress(messages, tokens) → bool (optional)")

    print("\n2. Configuration:")
    print("   • Use Pydantic Field() for validated attributes")
    print("   • Always include max_tokens parameter")
    print("   • Provide sensible defaults")
    print("   • Add field validators for constraints")

    print("\n3. Compression Logic:")
    print("   • Use self._estimate_tokens() for token counting")
    print("   • Track compression time (perf_counter)")
    print("   • Return CompressionMetrics with all fields")
    print("   • Handle edge cases (empty messages, single message)")

    print("\n4. Best Practices:")
    print("   • Keep compression fast (<100ms typical)")
    print("   • Be deterministic (same input → same output)")
    print("   • Preserve message order where possible")
    print("   • Document compression behavior clearly")

    print("\n5. Testing:")
    print("   • Test with various message counts")
    print("   • Verify metrics accuracy")
    print("   • Check edge cases (empty, single message)")
    print("   • Validate token estimation")

    print("\n6. Integration:")
    print("   • Pass to MemoryConfig(compressor=...)")
    print("   • Works with interrupt handlers automatically")
    print("   • Compatible with all memory modes")
    print("   • Metrics available in completion events")


async def main():
    """Run all custom compressor examples."""
    print("\n" + "=" * 70)
    print("Memory Compression - Custom Compressor Implementation")
    print("=" * 70)
    print("\nThis demonstrates implementing custom compression strategies:")
    print("1. Priority-based compression")
    print("2. Token-budget enforcement")
    print("3. Semantic clustering")

    await demonstrate_priority_compressor()
    await demonstrate_token_budget_compressor()
    await demonstrate_semantic_cluster_compressor()
    await implementation_guidelines()

    print("\n" + "=" * 70)
    print("For more examples, see:")
    print("  • memory_compression_basic.py - Built-in compression strategies")
    print("  • memory_compression_approval.py - HITL approval workflows")
    print("  • memory_compression_metrics.py - Monitoring and analytics")
    print("\nFor protocol documentation, see: src/pydantic_flow/memory/compression.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
