"""Checkpoint persistence integration example.

This example demonstrates:
1. Flow.run() accepts a RunConfig with checkpoint_store
2. Checkpoints are automatically persisted when InterruptionRequested is raised
3. The checkpoint ID and run ID are attached to the exception metadata
4. Checkpoints can be retrieved from the store after interruption
"""

import asyncio

from pydantic import BaseModel

from pydantic_flow import Flow
from pydantic_flow import PromptNode
from pydantic_flow.core.run_config import RunConfig
from pydantic_flow.hitl.checkpoints.interface import CheckpointId
from pydantic_flow.hitl.checkpoints.interface import CheckpointQuery
from pydantic_flow.hitl.checkpoints.interface import RunId
from pydantic_flow.hitl.checkpoints.memory import InMemoryCheckpointStore
from pydantic_flow.hitl.decisions import InterruptDecision
from pydantic_flow.hitl.interrupts import InterruptionRequested
from pydantic_flow.streaming.events import ProgressItem


class Query(BaseModel):
    """Input query."""

    question: str


class Response(BaseModel):
    """Output response."""

    answer: str


async def main() -> None:
    """Test checkpoint persistence on interruption."""
    # Create a simple flow with a prompt node
    flow = Flow(input_type=Query, output_type=Response)

    prompt_node = PromptNode[Query, Response](
        name="answer",
        prompt="Answer this question briefly: {question}",
    )
    flow.add_nodes(prompt_node)

    # Create checkpoint store
    store = InMemoryCheckpointStore()

    # Create run config with store
    config = RunConfig(checkpoint_store=store, run_id="test_run_123")

    # Register an interrupt handler that always interrupts
    async def interrupt_handler(item: ProgressItem) -> InterruptDecision:
        # Interrupt on any progress item
        return InterruptDecision(should_interrupt=True, reason="Test interruption")

    flow.register_interrupt_handler(callback=interrupt_handler, priority=0)

    # Run flow - should trigger interruption
    try:
        await flow.run(Query(question="What is 2+2?"), config=config)
        print("ERROR: Flow should have been interrupted!")
    except InterruptionRequested as e:
        print("✅ Flow interrupted as expected")

        # Verify checkpoint metadata
        checkpoint_id = e.checkpoint.metadata.get("checkpoint_id")
        run_id_str = e.checkpoint.metadata.get("run_id")

        if checkpoint_id and run_id_str:
            print(f"✅ Checkpoint ID attached: {checkpoint_id}")
            print(f"✅ Run ID attached: {run_id_str}")

            # Verify checkpoint in store
            stored = await store.get(
                run_id=RunId(run_id_str), checkpoint_id=CheckpointId(checkpoint_id)
            )
            if stored:
                print("✅ Checkpoint persisted to store")
                print(f"   - Envelope ID: {stored.id}")
                print(f"   - Run ID: {stored.run_id}")
                print(f"   - Node ID: {stored.node_id}")
                print(f"   - Created at: {stored.created_at}")

                # List all checkpoints for this run
                query = CheckpointQuery(run_id=RunId(run_id_str))
                all_checkpoints, _ = await store.list(query)
                print(f"✅ Total checkpoints in store for run: {len(all_checkpoints)}")
            else:
                print("❌ Checkpoint not found in store")
        else:
            print("❌ Missing checkpoint ID or run ID in metadata")


if __name__ == "__main__":
    asyncio.run(main())
