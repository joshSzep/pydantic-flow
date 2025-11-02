"""Memory Compression - Basic Examples.

This example demonstrates the three built-in compression strategies:
1. SlidingWindowCompressor - Fast, simple window-based compression
2. SummarizationCompressor - LLM-based semantic summarization
3. HybridCompressor - Adaptive strategy selection

Run with: uv run python examples/memory_compression_basic.py
"""

import asyncio

from pydantic import BaseModel
from pydantic_ai import Agent

from pydantic_flow import Flow
from pydantic_flow import HybridCompressor
from pydantic_flow import MemoryConfig
from pydantic_flow import SlidingWindowCompressor
from pydantic_flow import SummarizationCompressor


# Helper to extract result from stream
async def extract_result_from_stream(stream):
    """Extract final result from async stream of progress items."""
    result = None
    async for item in stream:
        if hasattr(item, "result"):
            result = item.result
    return result


class ChatInput(BaseModel):
    """User message input."""

    message: str


class ChatOutput(BaseModel):
    """Agent response output."""

    response: str


def create_test_agent() -> Agent[None, str]:
    """Create a simple test agent."""
    return Agent("test")


async def demonstrate_sliding_window():
    """Example 1: Sliding Window Compression.

    - Keeps only the N most recent messages
    - Fast (no LLM calls)
    - Good for conversations where recent context is sufficient
    """
    print("\n" + "=" * 70)
    print("Example 1: Sliding Window Compression")
    print("=" * 70)

    # Create compressor that keeps last 5 messages
    compressor = SlidingWindowCompressor(
        window_size=5,
        preserve_system_messages=True,
        max_tokens=100,  # Low threshold for demo
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
    print(f"Window size: {compressor.window_size}")
    print(f"Max tokens: {compressor.max_tokens}")

    # Simulate adding many messages to trigger compression
    print("\nSimulating conversation with 10 messages...")
    print("(Compression should trigger after ~5 messages at 100 token limit)")

    # Note: In a real scenario, messages would be added through agent execution
    # This is just a demonstration of the compression mechanics


async def demonstrate_summarization():
    """Example 2: LLM-Based Summarization.

    - Uses an LLM to create semantic summaries
    - Preserves meaning while reducing tokens
    - Slower but more intelligent than sliding window
    """
    print("\n" + "=" * 70)
    print("Example 2: LLM Summarization Compression")
    print("=" * 70)

    # Create summarization agent
    summarizer = Agent(
        "test",
        instructions="Summarize concisely while preserving key information.",
    )

    # Create summarization compressor
    compressor = SummarizationCompressor(
        agent=summarizer,
        prompt_template="""
        Summarize this conversation history:

        {messages}

        Preserve:
        - Key facts and decisions
        - Important context
        - User preferences

        Be concise but complete.
        """,
        preserve_system_messages=True,
        max_tokens=200,
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
    print("Strategy: LLM-based summarization")
    print(f"Max tokens: {compressor.max_tokens}")
    print("\nNote: Summarization creates semantic summaries of older messages")
    print("while keeping recent messages intact.")


async def demonstrate_hybrid():
    """Example 3: Hybrid Compression Strategy.

    - Combines sliding window and summarization
    - Uses sliding window for small histories (<threshold)
    - Switches to summarization for larger histories
    - Best of both worlds: fast when possible, intelligent when needed
    """
    print("\n" + "=" * 70)
    print("Example 3: Hybrid Compression Strategy")
    print("=" * 70)

    # Create summarization agent
    Agent(
        "test",
        instructions="Create concise conversation summaries.",
    )

    # Create hybrid compressor
    compressor = HybridCompressor(
        summarization_threshold=10,  # Switch to summarization at 10+ messages
        window_size=5,  # Sliding window size
        max_tokens=150,
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
    print(f"Threshold: {compressor.summarization_threshold} messages")
    print(f"Window size: {compressor.window_size}")
    print(f"Max tokens: {compressor.max_tokens}")
    print("\nStrategy:")
    print("  • < 10 messages: Use sliding window (fast)")
    print("  • ≥ 10 messages: Use summarization (intelligent)")


async def observe_compression_events():
    """Example 4: Observing Compression Events.

    Shows how to monitor compression operations via streaming events.
    """
    print("\n" + "=" * 70)
    print("Example 4: Observing Compression Events")
    print("=" * 70)

    compressor = SlidingWindowCompressor(window_size=5, max_tokens=100)

    _ = Flow[ChatInput, ChatOutput](
        input_type=ChatInput,
        output_type=ChatOutput,
        memory_config=MemoryConfig(
            enable_conversation_memory=True,
            compressor=compressor,
            emit_compression_events=True,  # Enable event emission
        ),
    )

    print("\nCompression events will appear in the stream as:")
    print("  • MemoryCompressionPending - Before compression")
    print("  • MemoryCompressionComplete - After compression with metrics")
    print("\nExample event handling:")
    print("""
    async for event in flow.astream(input_data):
        if event.type == ProgressType.MEMORY_COMPRESSION_PENDING:
            print(f"⏳ About to compress {event.message_count} messages")
            print(f"   Estimated: {event.estimated_tokens} tokens")

        elif event.type == ProgressType.MEMORY_COMPRESSION_COMPLETE:
            metrics = event.metrics
            print(f"✅ Compression complete!")
            print(f"   Reduction: {metrics.percentage_reduction:.1f}%")
            print(f"   Tokens saved: {metrics.tokens_saved}")
            print(f"   Time: {metrics.compression_time_ms:.0f}ms")
    """)


async def compare_strategies():
    """Example 5: Strategy Comparison.

    Compare the characteristics of different compression strategies.
    """
    print("\n" + "=" * 70)
    print("Example 5: Strategy Comparison")
    print("=" * 70)

    print("\n┌─────────────────┬──────────────┬──────────────┬──────────────┐")
    print("│ Strategy        │ Speed        │ Quality      │ Use Case     │")
    print("├─────────────────┼──────────────┼──────────────┼──────────────┤")
    print("│ Sliding Window  │ ⚡⚡⚡ Fast    │ ⭐ Basic     │ Recent only  │")
    print("│ Summarization   │ 🐌 Slow      │ ⭐⭐⭐ Best  │ Semantic     │")
    print("│ Hybrid          │ ⚡⚡ Adaptive │ ⭐⭐ Good    │ General      │")
    print("└─────────────────┴──────────────┴──────────────┴──────────────┘")

    print("\nChoosing a Strategy:")
    print("\n  Sliding Window:")
    print("    ✓ You only need recent context")
    print("    ✓ Speed is critical")
    print("    ✓ No LLM API costs acceptable")
    print("    ✗ Loses older context completely")

    print("\n  Summarization:")
    print("    ✓ Need to preserve semantic meaning")
    print("    ✓ Long conversations with important history")
    print("    ✓ Quality over speed")
    print("    ✗ Slower (LLM call required)")
    print("    ✗ Additional API costs")

    print("\n  Hybrid:")
    print("    ✓ General purpose solution")
    print("    ✓ Adapts to conversation length")
    print("    ✓ Good balance of speed and quality")
    print("    ✓ Recommended for most use cases")


async def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("Memory Compression - Basic Examples")
    print("=" * 70)
    print("\nThis demonstrates the three built-in compression strategies:")
    print("1. Sliding Window - Keep N recent messages")
    print("2. Summarization - LLM-based semantic compression")
    print("3. Hybrid - Adaptive strategy selection")

    await demonstrate_sliding_window()
    await demonstrate_summarization()
    await demonstrate_hybrid()
    await observe_compression_events()
    await compare_strategies()

    print("\n" + "=" * 70)
    print("For more advanced examples, see:")
    print("  • memory_compression_approval.py - HITL approval workflows")
    print("  • memory_compression_custom.py - Custom compressor implementation")
    print("  • memory_compression_metrics.py - Monitoring and analytics")
    print("\nFor detailed documentation, see: docs/memory_compression.md")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
