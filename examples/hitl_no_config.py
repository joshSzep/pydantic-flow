"""HITL example showing out-of-the-box functionality with zero configuration.

This example demonstrates that HITL interrupts work automatically without
explicitly configuring a checkpoint backend. An in-memory SQLite backend
is created automatically behind the scenes.
"""

import asyncio

from pydantic import BaseModel

from pydantic_flow import AgentNode
from pydantic_flow import Flow
from pydantic_flow.hitl.decisions import InterruptDecision
from pydantic_flow.hitl.interrupts import HandlerPriority
from pydantic_flow.hitl.interrupts import InterruptionRequested
from pydantic_flow.streaming.base import ProgressItem
from pydantic_flow.streaming.core_events import StreamEnd


# Helper to extract result from stream
async def extract_result_from_stream(stream):
    """Extract final result from async stream of progress items."""
    result = None
    async for item in stream:
        if hasattr(item, "result"):
            result = item.result
    return result


class ContentInput(BaseModel):
    """Input content to process."""

    text: str


class Summary(BaseModel):
    """Output summary."""

    text: str


async def main():
    """Run HITL workflow with ZERO configuration - it just works."""
    print("=" * 70)
    print("HITL WITH ZERO CONFIGURATION")
    print("=" * 70)
    print()
    print("🎯 This example shows HITL working out of the box")
    print("   No checkpoint backend configured!")
    print("   An in-memory SQLite backend is created automatically.")
    print()

    # Create a simple flow
    summarizer = AgentNode.from_prompt(
        model="openai:gpt-4",
        prompt_template="Summarize this briefly: {text}",
        name="summarizer",
    )

    flow = Flow(input_type=ContentInput, output_type=Summary)
    flow.add_nodes(summarizer)

    # Register an interrupt handler that triggers on completion
    async def require_approval(item: ProgressItem) -> InterruptDecision:
        """Request approval when processing completes."""
        if isinstance(item, StreamEnd):
            return InterruptDecision.interrupt(
                "Content requires approval before publishing",
                metadata={"approval_type": "content_review"},
            )
        return InterruptDecision.proceed()

    flow.register_interrupt_handler(require_approval, priority=HandlerPriority.HIGH)

    # Run the flow WITHOUT any RunConfig - it still works!
    input_data = ContentInput(text="AI is transforming the world of software.")

    try:
        print("🚀 Running flow (no RunConfig provided)...")
        result = await extract_result_from_stream(flow.astream(input_data))
        print(f"Unexpected success: {result}\n")
    except InterruptionRequested as exc:
        print()
        print("✋ HITL Interrupt Caught Successfully!")
        print("-" * 70)
        print(f"   Run ID: {exc.snapshot.run_id}")
        print(f"   Snapshot ID: {exc.snapshot.snapshot_id}")
        print(f"   Interrupted Node: {exc.snapshot.interrupted_node_id}")
        print(f"   Wave Number: {exc.snapshot.wave_number}")
        print(f"   Decision: {exc.decision.reason}")
        print()
        print("✅ HITL works out of the box with zero configuration!")
        print("   The snapshot was saved to an automatic in-memory backend.")
        print()
        print("💡 Production tip: Provide a persistent checkpoint_backend")
        print("   in RunConfig to save interrupts across restarts.")
        print()


if __name__ == "__main__":
    asyncio.run(main())
