"""Memory Compression - Adaptive Strategy Selection.

This example demonstrates dynamic strategy selection and runtime strategy
replacement using the replacement_value feature in interrupt handlers.

Key concepts:
1. Dynamic strategy selection based on context
2. Runtime strategy replacement via InterruptDecision.interrupt()
3. Conditional compression triggers
4. Strategy switching for different conversation phases
5. Cost-aware strategy selection

Note: To replace the compressor, use InterruptDecision.interrupt() with
replacement_value. The system will abort the current compression and retry
with the new compressor.

Run with: uv run python examples/memory_compression_adaptive.py
"""

import asyncio

from pydantic import BaseModel
from pydantic_ai import Agent

from pydantic_flow import Flow
from pydantic_flow import HybridCompressor
from pydantic_flow import MemoryConfig
from pydantic_flow import SlidingWindowCompressor
from pydantic_flow import SummarizationCompressor
from pydantic_flow.streaming.events import InterruptDecision
from pydantic_flow.streaming.events import MemoryCompressionPending
from pydantic_flow.streaming.events import ProgressItem


class ChatInput(BaseModel):
    """User message input."""

    message: str
    conversation_phase: str = "exploration"


class ChatOutput(BaseModel):
    """Agent response output."""

    response: str


def create_summarizer() -> Agent[None, str]:
    """Create agent for summarization."""
    return Agent("test")


# ============================================================================
# Example 1: Context-Aware Strategy Replacement
# ============================================================================


async def example_1_context_aware_replacement():
    """Replace compression strategy based on conversation context."""
    print("\n" + "=" * 70)
    print("Example 1: Context-Aware Strategy Replacement")
    print("=" * 70)

    # Start with fast sliding window
    initial_compressor = SlidingWindowCompressor(
        window_size=10,
        max_tokens=100,
    )

    _ = Flow[ChatInput, ChatOutput](
        input_type=ChatInput,
        output_type=ChatOutput,
        memory_config=MemoryConfig(
            enable_conversation_memory=True,
            compressor=initial_compressor,
            emit_compression_events=True,
        ),
    )

    # Create alternative strategies
    summarizer = create_summarizer()
    quality_compressor = SummarizationCompressor(
        agent=summarizer,
        prompt_template="Summarize: {messages}",
        max_tokens=100,
    )

    async def select_by_phase(item: ProgressItem) -> InterruptDecision:
        """Select compression strategy based on conversation phase."""
        if not isinstance(item, MemoryCompressionPending):
            return InterruptDecision.proceed()

        # Access conversation phase from metadata (simulated)
        phase = "decision"  # Would come from flow context

        print(f"\n🔄 Compression triggered in '{phase}' phase:")
        print(f"   Current strategy: {item.compressor_name}")
        print(f"   Messages: {item.message_count}")

        # Strategy replacement logic
        if phase == "exploration":
            # Fast compression OK during exploration
            print("   ✓ Using sliding window (speed priority)")
            return InterruptDecision.proceed()

        elif phase == "decision":
            # Quality matters during decision-making
            print("   🔄 Replacing with summarization (quality priority)")
            return InterruptDecision.interrupt(
                "Upgrading to quality compressor",
                replacement_value=quality_compressor,
            )

        elif phase == "conclusion":
            # Preserve everything during conclusion
            print("   ❌ Blocking compression (preserve full context)")
            return InterruptDecision.interrupt("Preserve context for conclusion")

        return InterruptDecision.proceed()

    print("\nStrategy Replacement Logic:")
    print("  • Exploration phase: Allow sliding window")
    print("  • Decision phase: Interrupt & replace with summarization")
    print("  • Conclusion phase: Interrupt & block compression")

    print("\nHow replacement works:")
    print("  1. Interrupt handler returns InterruptDecision.interrupt(...)")
    print("  2. Provides replacement_value=new_compressor")
    print("  3. System aborts current compression")
    print("  4. Retries compression with new compressor")
    print("  5. New compressor becomes active for this operation")


async def example_2_cost_aware_replacement():
    """Replace strategy based on cost constraints."""
    print("\n" + "=" * 70)
    print("Example 2: Cost-Aware Strategy Replacement")
    print("=" * 70)

    initial_compressor = HybridCompressor(
        summarization_threshold=15,
        window_size=10,
        summarizer_agent=create_summarizer(),
        max_tokens=100,
    )

    _ = Flow[ChatInput, ChatOutput](
        input_type=ChatInput,
        output_type=ChatOutput,
        memory_config=MemoryConfig(
            enable_conversation_memory=True,
            compressor=initial_compressor,
            emit_compression_events=True,
        ),
    )

    # Cost-free fallback
    cheap_compressor = SlidingWindowCompressor(
        window_size=10,
        max_tokens=100,
    )

    # Track API costs (simulated)
    api_cost_this_hour = 0.09  # dollars
    cost_limit = 0.10  # dollars/hour

    async def cost_aware_replacement(item: ProgressItem) -> InterruptDecision:
        """Replace strategy when approaching cost limits."""
        if not isinstance(item, MemoryCompressionPending):
            return InterruptDecision.proceed()

        print("\n💰 Cost-Aware Compression:")
        print(f"   Current strategy: {item.compressor_name}")
        print(f"   API cost this hour: ${api_cost_this_hour:.2f}")
        print(f"   Cost limit: ${cost_limit:.2f}")

        # Check if we're approaching cost limit
        if (
            api_cost_this_hour >= cost_limit * 0.8
            and "summarization" in item.compressor_name
        ):
            print("   ⚠️  Approaching cost limit (90% of budget)")
            print("   🔄 Replacing with cost-free sliding window")
            return InterruptDecision.interrupt(
                "Cost limit approaching",
                replacement_value=cheap_compressor,
            )

        print("   ✓ Using configured strategy")
        return InterruptDecision.proceed()

    print("\nCost Management:")
    print("  • Monitor API costs in real-time")
    print("  • Replace with free strategies when limits approach")
    print("  • Preserve quality when budget allows")
    print("  • Automatic fallback prevents cost overruns")


async def example_3_message_count_based():
    """Replace strategy based on message count thresholds."""
    print("\n" + "=" * 70)
    print("Example 3: Message Count-Based Replacement")
    print("=" * 70)

    initial_compressor = SlidingWindowCompressor(
        window_size=5,
        max_tokens=100,
    )

    _ = Flow[ChatInput, ChatOutput](
        input_type=ChatInput,
        output_type=ChatOutput,
        memory_config=MemoryConfig(
            enable_conversation_memory=True,
            compressor=initial_compressor,
            emit_compression_events=True,
        ),
    )

    # Create strategy alternatives
    large_window = SlidingWindowCompressor(window_size=15, max_tokens=100)
    summarizer = SummarizationCompressor(
        agent=create_summarizer(),
        prompt_template="Summarize: {messages}",
        max_tokens=100,
    )

    async def select_by_message_count(item: ProgressItem) -> InterruptDecision:
        """Replace strategy based on message count."""
        if not isinstance(item, MemoryCompressionPending):
            return InterruptDecision.proceed()

        message_count = item.message_count

        print(f"\n📊 Message Count: {message_count}")
        print(f"   Current strategy: {item.compressor_name}")

        # Strategy replacement thresholds
        SMALL_CONVERSATION = 20
        LARGE_CONVERSATION = 50

        if message_count < SMALL_CONVERSATION:
            # Small conversations: use initial strategy
            print("   ✓ Using small window (5 messages)")
            return InterruptDecision.proceed()

        elif message_count < LARGE_CONVERSATION:
            # Medium conversations: upgrade to larger window
            print("   🔄 Replacing with large window (15 messages)")
            return InterruptDecision.interrupt(
                "Upgrading to large window",
                replacement_value=large_window,
            )

        else:
            # Large conversations: use summarization
            print("   🔄 Replacing with summarization (semantic preservation)")
            return InterruptDecision.interrupt(
                "Upgrading to summarization",
                replacement_value=summarizer,
            )

    print("\nThreshold-Based Replacement:")
    print("  • <20 messages: Keep small window (5)")
    print("  • 20-49 messages: Replace with large window (15)")
    print("  • ≥50 messages: Replace with summarization")
    print("\nBenefit: Optimal strategy for conversation size")


async def example_4_quality_gating():
    """Block compression if quality requirements not met."""
    print("\n" + "=" * 70)
    print("Example 4: Quality Gating")
    print("=" * 70)

    compressor = SlidingWindowCompressor(
        window_size=5,
        max_tokens=100,
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

    # Define quality requirements
    MIN_MESSAGES_TO_PRESERVE = 10

    async def quality_gate(item: ProgressItem) -> InterruptDecision:
        """Block compression if quality requirements not met."""
        if not isinstance(item, MemoryCompressionPending):
            return InterruptDecision.proceed()

        message_count = item.message_count

        # Calculate expected outcome
        expected_messages_after = compressor.window_size
        expected_reduction = message_count - expected_messages_after

        print("\n✓ Quality Gate Check:")
        print(f"   Messages: {message_count}")
        print(f"   Will preserve: {expected_messages_after}")
        print(f"   Will remove: {expected_reduction}")

        # Check if preservation threshold met
        if expected_messages_after < MIN_MESSAGES_TO_PRESERVE:
            print(f"   ❌ Below minimum ({MIN_MESSAGES_TO_PRESERVE} messages)")
            print("   🚫 Blocking compression")
            return InterruptDecision.interrupt("Insufficient message preservation")

        print("   ✓ Quality requirements met")
        return InterruptDecision.proceed()

    print("\nQuality Gates:")
    print(f"  • Minimum messages to preserve: {MIN_MESSAGES_TO_PRESERVE}")
    print("  • Block compression if threshold not met")
    print("  • Ensures minimum context always available")


async def best_practices():
    """Display best practices for adaptive strategies."""
    print("\n" + "=" * 70)
    print("Adaptive Strategy Best Practices")
    print("=" * 70)

    print("\n1. Strategy Replacement Pattern:")
    print("   return InterruptDecision.interrupt(")
    print("       'reason for replacement',")
    print("       replacement_value=new_compressor")
    print("   )")

    print("\n2. When to Replace:")
    print("   • Context change (conversation phase)")
    print("   • Resource constraints (cost, time)")
    print("   • Message count thresholds")
    print("   • Quality requirements")
    print("   • System load conditions")

    print("\n3. Replacement Behavior:")
    print("   • Aborts current compression attempt")
    print("   • Retries with replacement compressor")
    print("   • Replacement becomes active compressor")
    print("   • Can chain replacements (replacement triggers another)")

    print("\n4. Strategy Pool:")
    print("   • Pre-create compressor instances")
    print("   • Reuse instances across replacements")
    print("   • Configure with appropriate parameters")
    print("   • Keep pool small (2-3 strategies typical)")

    print("\n5. Testing:")
    print("   • Verify replacement logic with unit tests")
    print("   • Monitor replacement frequency")
    print("   • Track which strategies get selected")
    print("   • Validate quality after replacement")

    print("\n6. Blocking Compression:")
    print("   • Use InterruptDecision.interrupt(reason) without replacement")
    print("   • Blocks compression, preserves full history")
    print("   • Use for: critical phases, debugging, quality gates")


async def main():
    """Run all adaptive strategy examples."""
    print("\n" + "=" * 70)
    print("Memory Compression - Adaptive Strategy Selection")
    print("=" * 70)
    print("\nThis demonstrates dynamic strategy replacement:")
    print("1. Context-aware replacement (conversation phase)")
    print("2. Cost-aware replacement (budget management)")
    print("3. Message count-based replacement")
    print("4. Quality gating (block if insufficient)")

    await example_1_context_aware_replacement()
    await example_2_cost_aware_replacement()
    await example_3_message_count_based()
    await example_4_quality_gating()
    await best_practices()

    print("\n" + "=" * 70)
    print("For more examples, see:")
    print("  • memory_compression_basic.py - Basic compression strategies")
    print("  • memory_compression_approval.py - HITL approval workflows")
    print("  • memory_compression_custom.py - Custom compressor implementation")
    print("  • memory_compression_metrics.py - Monitoring and analytics")
    print("\nFor interrupt documentation, see: docs/hitl.md")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
