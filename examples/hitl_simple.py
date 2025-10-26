"""Simple Human-in-the-Loop (HITL) example.

This example demonstrates interrupt handlers that conditionally request human review.
"""

import asyncio

from pydantic import BaseModel

from pydantic_flow import Flow
from pydantic_flow import PromptConfig
from pydantic_flow import PromptNode
from pydantic_flow.core.errors import HandlerPriority
from pydantic_flow.core.errors import InterruptionRequested
from pydantic_flow.streaming.events import InterruptDecision
from pydantic_flow.streaming.events import ProgressItem
from pydantic_flow.streaming.events import StreamEnd


class ContentInput(BaseModel):
    """Input content to process."""

    text: str
    requires_review: bool = False


class Summary(BaseModel):
    """Output summary."""

    text: str


async def main():
    """Run simple HITL workflow with conditional interruption."""
    # Create a processing node
    processor = PromptNode[ContentInput, Summary](
        prompt="Summarize this text: {input.text}",
        config=PromptConfig(model="openai:gpt-4"),
        name="processor",
    )

    # Build flow
    flow = Flow(input_type=ContentInput, output_type=Summary)
    flow.add_nodes(processor)

    print("=" * 60)
    print("Example 1: Content that does NOT require review")
    print("=" * 60 + "\n")

    # First execution - no interruption
    input_data = ContentInput(text="The sky is blue.", requires_review=False)

    try:
        result = await flow.run(input_data)
        print("✅ Workflow completed without interruption")
        print(f"   Result: {result.text}\n")

    except InterruptionRequested as exc:
        print(f"Unexpected interruption: {exc.checkpoint}\n")

    print("=" * 60)
    print("Example 2: Flow-level interrupt handler")
    print("=" * 60 + "\n")

    # Register a flow-level handler that always interrupts at stream end
    async def always_review(item: ProgressItem) -> InterruptDecision:
        """Request review at stream end."""
        if isinstance(item, StreamEnd):
            return InterruptDecision.interrupt(
                "Final review required", metadata={"review_type": "final"}
            )
        return InterruptDecision.proceed()

    flow.register_interrupt_handler(always_review, priority=HandlerPriority.HIGH)

    input_data_with_review = ContentInput(
        text="Important company announcement.", requires_review=True
    )

    try:
        result = await flow.run(input_data_with_review)
        print(f"Unexpected success: {result}\n")

    except InterruptionRequested as exc:
        checkpoint = exc.checkpoint
        print("✋ Workflow interrupted for human review")
        print(f"   Flow ID: {checkpoint.flow_id}")
        print(f"   Interrupted at node: {checkpoint.interrupted_node_id}")
        print(f"   Checkpoint metadata: {checkpoint.metadata}")
        print("\n👤 Human review would occur here...")
        print("   (In production, present content to reviewer UI)\n")


if __name__ == "__main__":
    asyncio.run(main())
