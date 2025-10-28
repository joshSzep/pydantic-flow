"""Memory Compression - Human Approval Workflow.

This example demonstrates how to use Human-in-the-Loop (HITL) interrupts
to request approval before compressing conversation memory.

Key concepts:
1. MemoryCompressionPending - Request approval before compression
2. MemoryCompressionComplete - Review results after compression
3. InterruptDecision.proceed() - Continue with compression
4. InterruptDecision.interrupt() - Reject compression

Run with: uv run python examples/memory_compression_approval.py
"""

import asyncio

from pydantic import BaseModel

from pydantic_flow import Flow
from pydantic_flow import MemoryConfig
from pydantic_flow import SlidingWindowCompressor
from pydantic_flow.streaming.events import InterruptDecision
from pydantic_flow.streaming.events import MemoryCompressionComplete
from pydantic_flow.streaming.events import MemoryCompressionPending
from pydantic_flow.streaming.events import ProgressItem


class ChatInput(BaseModel):
    """User message input."""

    message: str


class ChatOutput(BaseModel):
    """Agent response output."""

    response: str


async def example_1_approve_before_compression():
    """Example 1: Request Approval Before Compression.

    Shows how to intercept compression and ask for user approval.
    """
    print("\n" + "=" * 70)
    print("Example 1: Approve Before Compression")
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

    print("\nSetup: Sliding window compressor with max_tokens=100")
    print("Handler: Request approval before compression proceeds\n")

    async def request_approval(item: ProgressItem) -> InterruptDecision:
        """Request user approval before compressing."""
        if isinstance(item, MemoryCompressionPending):
            print("\n⚠️  Compression Pending:")
            print(f"   Messages: {item.message_count}")
            print(f"   Estimated tokens: {item.estimated_tokens}")
            print(f"   Compressor: {item.compressor_name}")
            print(f"   Reason: {item.compression_reason}")

            # Simulate user decision
            print("\n   [In production, prompt user here]")
            user_approves = True  # Simulate approval

            if user_approves:
                print("   ✅ User approved - proceeding with compression")
                return InterruptDecision.proceed()
            else:
                print("   ❌ User rejected - preserving full history")
                return InterruptDecision.interrupt(
                    "User rejected compression",
                    metadata={"user_id": "demo_user"},
                )

        return InterruptDecision.proceed()

    print("Interrupt handler code:")
    print("""
    async def request_approval(item: ProgressItem) -> InterruptDecision:
        if isinstance(item, MemoryCompressionPending):
            print(f"About to compress {item.message_count} messages")

            user_approves = input("Approve? (y/n): ") == "y"

            if user_approves:
                return InterruptDecision.proceed()  # Allow compression
            else:
                return InterruptDecision.interrupt("Rejected")  # Block

        return InterruptDecision.proceed()
    """)

    print("\nBehavior:")
    print("  • If approved: Compression proceeds, older messages removed")
    print("  • If rejected: Interrupt raised, full history preserved")


async def example_2_conditional_approval():
    """Example 2: Conditional Approval Based on Metrics.

    Auto-approve if reduction is significant, else ask user.
    """
    print("\n" + "=" * 70)
    print("Example 2: Conditional Approval Based on Metrics")
    print("=" * 70)

    compressor = SlidingWindowCompressor(window_size=10, max_tokens=200)

    _ = Flow[ChatInput, ChatOutput](
        input_type=ChatInput,
        output_type=ChatOutput,
        memory_config=MemoryConfig(
            enable_conversation_memory=True,
            compressor=compressor,
            emit_compression_events=True,
        ),
    )

    print("\nSetup: Conditional approval based on message count")
    print("Logic: Auto-approve if >50% messages will be removed\n")

    async def conditional_approval(item: ProgressItem) -> InterruptDecision:
        """Conditionally approve based on compression metrics."""
        if isinstance(item, MemoryCompressionPending):
            message_count = item.message_count
            window_size = compressor.window_size

            # Calculate reduction percentage
            messages_after = min(message_count, window_size)
            reduction_pct = (
                (message_count - messages_after) / message_count * 100
                if message_count > 0
                else 0
            )

            print("\n📊 Compression Analysis:")
            print(f"   Messages before: {message_count}")
            print(f"   Messages after: {messages_after}")
            print(f"   Reduction: {reduction_pct:.1f}%")

            # Auto-approve significant reductions
            if reduction_pct >= 50:
                print("   ✅ Auto-approved (significant reduction)")
                return InterruptDecision.proceed()

            # Reject marginal compressions
            print("   ❌ Auto-rejected (insufficient reduction)")
            return InterruptDecision.interrupt(
                f"Only {reduction_pct:.1f}% reduction - not worth it",
                metadata={"reduction_pct": reduction_pct},
            )

        return InterruptDecision.proceed()

    print("Interrupt handler code:")
    print("""
    async def conditional_approval(item: ProgressItem) -> InterruptDecision:
        if isinstance(item, MemoryCompressionPending):
            reduction_pct = calculate_reduction(item.message_count)

            if reduction_pct >= 50:
                return InterruptDecision.proceed()  # Auto-approve
            else:
                return InterruptDecision.interrupt("Too small")  # Reject

        return InterruptDecision.proceed()
    """)

    print("\nBehavior:")
    print("  • ≥50% reduction: Auto-approved, compression proceeds")
    print("  • <50% reduction: Auto-rejected, interrupt raised")


async def example_3_review_after_compression():
    """Example 3: Review Results After Compression.

    Monitor completed compressions and log metrics.
    """
    print("\n" + "=" * 70)
    print("Example 3: Review Results After Compression")
    print("=" * 70)

    compressor = SlidingWindowCompressor(window_size=5, max_tokens=100)

    _ = Flow[ChatInput, ChatOutput](
        input_type=ChatInput,
        output_type=ChatOutput,
        memory_config=MemoryConfig(
            enable_conversation_memory=True,
            compressor=compressor,
            emit_compression_events=True,
        ),
    )

    print("\nSetup: Monitor compression completion events")
    print("Purpose: Log metrics, display results, collect analytics\n")

    async def review_completion(item: ProgressItem) -> InterruptDecision:
        """Review compression results after completion."""
        if isinstance(item, MemoryCompressionComplete):
            metrics = item.metrics

            print("\n✅ Compression Complete:")
            print(f"   Strategy: {item.type.value}")
            print(f"   Messages: {metrics.messages_before} → {metrics.messages_after}")
            print(f"   Tokens: {metrics.tokens_before} → {metrics.tokens_after}")
            print(f"   Saved: {metrics.tokens_saved} tokens")
            print(f"   Reduction: {metrics.percentage_reduction:.1f}%")
            print(f"   Ratio: {metrics.compression_ratio:.2f}")
            print(f"   Time: {metrics.compression_time_ms:.0f}ms")

            # Could optionally reject based on results
            if metrics.compression_ratio > 0.8:
                print("   ⚠️  Low compression ratio - rejecting")
                return InterruptDecision.interrupt(
                    "Insufficient compression achieved",
                    metadata={"ratio": metrics.compression_ratio},
                )

            print("   ✓ Compression accepted")

        return InterruptDecision.proceed()

    print("Interrupt handler code:")
    print("""
    async def review_completion(item: ProgressItem) -> InterruptDecision:
        if isinstance(item, MemoryCompressionComplete):
            metrics = item.metrics

            print(f"Compressed: {metrics.percentage_reduction:.1f}% reduction")
            print(f"Saved: {metrics.tokens_saved} tokens")

            # Could reject if insufficient
            if metrics.compression_ratio > 0.8:
                return InterruptDecision.interrupt("Not enough compression")

        return InterruptDecision.proceed()
    """)

    print("\nBehavior:")
    print("  • Displays detailed metrics after compression")
    print("  • Can reject based on compression quality")
    print("  • Useful for monitoring and analytics")


async def example_4_combined_workflow():
    """Example 4: Combined Pending + Complete Workflow.

    Handle both before and after compression in one handler.
    """
    print("\n" + "=" * 70)
    print("Example 4: Combined Pending + Complete Workflow")
    print("=" * 70)

    compressor = SlidingWindowCompressor(window_size=5, max_tokens=100)

    _ = Flow[ChatInput, ChatOutput](
        input_type=ChatInput,
        output_type=ChatOutput,
        memory_config=MemoryConfig(
            enable_conversation_memory=True,
            compressor=compressor,
            emit_compression_events=True,
        ),
    )

    print("\nSetup: Handle both pending and complete events")
    print("Workflow: Approve before → Execute → Review after\n")

    async def full_workflow(item: ProgressItem) -> InterruptDecision:
        """Handle complete compression workflow."""
        if isinstance(item, MemoryCompressionPending):
            print("\n⏳ Compression Pending:")
            print(f"   Messages: {item.message_count}")
            print(f"   Tokens: ~{item.estimated_tokens}")
            print("   [Approval logic here]")
            print("   ✅ Approved")
            return InterruptDecision.proceed()

        elif isinstance(item, MemoryCompressionComplete):
            metrics = item.metrics
            print("\n✅ Compression Complete:")
            print(f"   Reduction: {metrics.percentage_reduction:.1f}%")
            print(f"   Time: {metrics.compression_time_ms:.0f}ms")
            print("   ✓ Results accepted")
            return InterruptDecision.proceed()

        return InterruptDecision.proceed()

    print("Interrupt handler code:")
    print("""
    async def full_workflow(item: ProgressItem) -> InterruptDecision:
        if isinstance(item, MemoryCompressionPending):
            print("About to compress...")
            return InterruptDecision.proceed()  # Approve

        elif isinstance(item, MemoryCompressionComplete):
            print(f"Done: {item.metrics.percentage_reduction:.1f}% reduction")
            return InterruptDecision.proceed()  # Accept

        return InterruptDecision.proceed()
    """)

    print("\nWorkflow:")
    print("  1. MemoryCompressionPending → Request approval")
    print("  2. If approved → Compression executes")
    print("  3. MemoryCompressionComplete → Review results")
    print("  4. If accepted → Changes applied")


async def best_practices():
    """Display best practices for compression approval workflows."""
    print("\n" + "=" * 70)
    print("Best Practices for Compression Approval")
    print("=" * 70)

    print("\n1. When to Use Approval:")
    print("   ✓ Critical conversations where history matters")
    print("   ✓ Legal/compliance contexts requiring audit trails")
    print("   ✓ User-facing applications (transparency)")
    print("   ✗ High-throughput batch processing")
    print("   ✗ Automated background tasks")

    print("\n2. Approval Strategy Guidelines:")
    print("   • Auto-approve significant compressions (>50% reduction)")
    print("   • Require manual approval for marginal cases")
    print("   • Consider compression frequency (avoid approval fatigue)")
    print("   • Provide clear metrics in approval prompts")

    print("\n3. User Experience:")
    print("   • Show estimated impact before compression")
    print("   • Make decisions easy (clear metrics)")
    print("   • Display results for transparency")
    print("   • Log decisions for audit trails")

    print("\n4. Error Handling:")
    print("   • Handle InterruptionRequested when user rejects")
    print("   • Preserve full history on rejection")
    print("   • Provide clear feedback on decisions")
    print("   • Consider retry logic for edge cases")

    print("\n5. Integration with Flow:")
    print("   • Register handler via flow.register_interrupt_handler()")
    print("   • Or pass to flow.astream(interrupt=handler)")
    print("   • Use async handlers for non-blocking execution")
    print("   • Combine with other interrupt types (HITL, etc.)")


async def main():
    """Run all approval workflow examples."""
    print("\n" + "=" * 70)
    print("Memory Compression - Human Approval Workflows")
    print("=" * 70)
    print("\nThis demonstrates HITL integration with memory compression:")
    print("1. Request approval before compression")
    print("2. Conditional approval based on metrics")
    print("3. Review results after compression")
    print("4. Combined workflow (before + after)")

    await example_1_approve_before_compression()
    await example_2_conditional_approval()
    await example_3_review_after_compression()
    await example_4_combined_workflow()
    await best_practices()

    print("\n" + "=" * 70)
    print("For more examples, see:")
    print("  • memory_compression_basic.py - Basic compression strategies")
    print("  • memory_compression_custom.py - Custom compressor implementation")
    print("  • memory_compression_metrics.py - Monitoring and analytics")
    print("\nFor detailed HITL documentation, see: docs/hitl.md")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
