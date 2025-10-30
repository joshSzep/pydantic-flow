"""Simple Human-in-the-Loop (HITL) example.

This example demonstrates interrupt handlers with checkpoint persistence and resumption.
"""

import asyncio

from pydantic import BaseModel

from pydantic_flow import Flow
from pydantic_flow import PromptConfig
from pydantic_flow import PromptNode
from pydantic_flow.core.run_config import RunConfig
from pydantic_flow.hitl.checkpoints.interface import CheckpointQuery
from pydantic_flow.hitl.checkpoints.interface import RunId
from pydantic_flow.hitl.checkpoints.interface import list_interrupted
from pydantic_flow.hitl.checkpoints.memory import InMemoryCheckpointStore
from pydantic_flow.hitl.decisions import InterruptDecision
from pydantic_flow.hitl.interrupts import HandlerPriority
from pydantic_flow.hitl.interrupts import InterruptionRequested
from pydantic_flow.streaming.events import ProgressItem
from pydantic_flow.streaming.events import StreamEnd


class ContentInput(BaseModel):
    """Input content to process."""

    text: str
    requires_review: bool = False


class Summary(BaseModel):
    """Output summary."""

    text: str


async def example_no_review(flow: Flow, input_data: ContentInput) -> None:
    """Example 1: Content that does NOT require review."""
    print("=" * 60)
    print("Example 1: Content that does NOT require review")
    print("=" * 60 + "\n")

    try:
        result = await flow.run(input_data)
        print("✅ Workflow completed without interruption")
        print(f"   Result: {result.text}\n")
    except InterruptionRequested as exc:
        print(f"Unexpected interruption: {exc.checkpoint}\n")


async def example_with_persistence(
    flow: Flow, input_data: ContentInput, store: InMemoryCheckpointStore, run_id: str
) -> None:
    """Example 2: Interrupt with checkpoint persistence."""
    print("=" * 60)
    print("Example 2: Interrupt with checkpoint persistence")
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

    # Configure run with checkpoint store
    config = RunConfig(checkpoint_store=store, run_id=run_id)

    try:
        result = await flow.run(input_data, config=config)
        print(f"Unexpected success: {result}\n")
    except InterruptionRequested as exc:
        checkpoint = exc.checkpoint
        print("✋ Workflow interrupted for human review")
        print(f"   Flow ID: {checkpoint.flow_id}")
        print(f"   Interrupted at node: {checkpoint.interrupted_node_id}")
        print(f"   Checkpoint ID: {checkpoint.metadata.get('checkpoint_id')}")
        print("\n📦 Checkpoint automatically saved to store")
        print("   Interrupt reason: Final review required")
        print(f"   Node state captured: {'processor' in checkpoint.node_states}")


async def example_query_checkpoints(
    store: InMemoryCheckpointStore, run_id: str
) -> None:
    """Example 3: Query interrupted checkpoints."""
    print("\n=" * 60)
    print("Example 3: Query interrupted checkpoints")
    print("=" * 60 + "\n")

    query = CheckpointQuery(run_id=RunId(run_id))
    interrupted_checkpoints, _ = await list_interrupted(store, query)

    print(f"📋 Found {len(interrupted_checkpoints)} interrupted checkpoint(s)")
    for envelope in interrupted_checkpoints:
        print(f"   - ID: {envelope.id}")
        print(f"   - Reason: {envelope.interrupt_reason}")
        print(f"   - Metadata: {envelope.interrupt_metadata}")
        print(f"   - Node: {envelope.node_id}")
        print(f"   - Created: {envelope.created_at}")

    print("\n💡 In production, you would:")
    print("   1. Present the checkpoint to a reviewer UI")
    print("   2. Wait for human approval/rejection")
    print("   3. Resume execution with flow.resume_from_store()")
    print("   4. Or implement custom approval workflows")


async def main():
    """Run simple HITL workflow with checkpoint persistence."""
    # Create a processing node
    processor = PromptNode[ContentInput, Summary](
        prompt="Summarize this text: {input.text}",
        config=PromptConfig(model="openai:gpt-4"),
        name="processor",
    )

    # Build flow
    flow = Flow(input_type=ContentInput, output_type=Summary)
    flow.add_nodes(processor)

    # Create checkpoint store for persistence
    store = InMemoryCheckpointStore()

    # Run examples
    await example_no_review(
        flow, ContentInput(text="The sky is blue.", requires_review=False)
    )

    await example_with_persistence(
        flow,
        ContentInput(text="Important company announcement.", requires_review=True),
        store,
        run_id="review_run_001",
    )

    await example_query_checkpoints(store, "review_run_001")


if __name__ == "__main__":
    asyncio.run(main())
