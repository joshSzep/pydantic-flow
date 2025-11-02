"""Checkpoint persistence integration example with unified checkpoint system.

This example demonstrates:
1. extract_result_from_stream(Flow.astream() accepts a RunConfig with checkpoint_backend
2. Checkpoints are automatically persisted when InterruptionRequested is raised
3. The snapshot ID and run ID are available in the exception
4. Snapshots can be queried from the backend after interruption
"""

import asyncio
from pathlib import Path

from pydantic import BaseModel

from pydantic_flow import Flow
from pydantic_flow import PromptNode
from pydantic_flow.checkpoints.backends.sqlite import SQLiteCheckpointBackend
from pydantic_flow.checkpoints.backends.sqlite import SQLiteCheckpointConfig
from pydantic_flow.checkpoints.inspection import CheckpointInspector
from pydantic_flow.core.run_config import RunConfig
from pydantic_flow.hitl.decisions import InterruptDecision
from pydantic_flow.hitl.interrupts import InterruptionRequested
from pydantic_flow.streaming.base import ProgressItem


# Helper to extract result from stream
async def extract_result_from_stream(stream):
    """Extract final result from async stream of progress items."""
    result = None
    async for item in stream:
        if hasattr(item, "result"):
            result = item.result
    return result


class Query(BaseModel):
    """Input query."""

    question: str


class Response(BaseModel):
    """Output response."""

    answer: str


async def main() -> None:
    """Test V2 checkpoint persistence on interruption."""
    # Create V2 checkpoint backend
    config = SQLiteCheckpointConfig(db_path=Path(":memory:"))
    backend = SQLiteCheckpointBackend(config)
    await backend.initialize()

    try:
        # Create a simple flow with a prompt node
        flow = Flow(input_type=Query, output_type=Response)

        prompt_node = PromptNode[Query, Response](
            name="answer",
            prompt="Answer this question briefly: {question}",
        )
        flow.add_nodes(prompt_node)

        # Create run config with backend
        run_config = RunConfig(checkpoint_backend=backend, run_id="test_run_123")

        # Register an interrupt handler that always interrupts
        async def interrupt_handler(item: ProgressItem) -> InterruptDecision:
            # Interrupt on any progress item
            return InterruptDecision.interrupt(
                reason="Test interruption", metadata={"test": True}
            )

        flow.register_interrupt_handler(callback=interrupt_handler, priority=0)

        # Run flow - should trigger interruption
        try:
            await extract_result_from_stream(
                flow.astream(Query(question="What is 2+2?"), config=run_config)
            )
            print("ERROR: Flow should have been interrupted!")
        except InterruptionRequested as e:
            print("✅ Flow interrupted as expected")

            # Verify snapshot details
            snapshot = e.snapshot
            print(f"✅ Snapshot ID: {snapshot.snapshot_id}")
            print(f"✅ Run ID: {snapshot.run_id}")
            print(f"✅ Wave number: {snapshot.wave_number}")
            print(f"✅ Interrupt reason: {snapshot.reason}")

            # Query the snapshot from backend
            inspector = CheckpointInspector(backend)
            stored_snapshot = await inspector.get_interrupt_snapshot(snapshot.run_id)
            if stored_snapshot:
                print("✅ Snapshot persisted to backend")
                print(f"   - Snapshot ID: {stored_snapshot.snapshot_id}")
                print(f"   - Run ID: {stored_snapshot.run_id}")
                print(f"   - Node: {stored_snapshot.interrupted_node_id}")
                print(f"   - Metadata: {stored_snapshot.metadata}")

                # List all interrupted runs
                interrupted_runs = await inspector.list_interrupted_runs(limit=10)
                print(f"✅ Total interrupted runs in backend: {len(interrupted_runs)}")
            else:
                print("❌ Snapshot not found in backend")

    finally:
        await backend.close()


if __name__ == "__main__":
    asyncio.run(main())
