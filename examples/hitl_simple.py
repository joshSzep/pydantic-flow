"""Simple Human-in-the-Loop (HITL) example with unified checkpoint system.

This example demonstrates:
1. Interrupt handlers with checkpoint persistence
2. Querying interrupted runs using CheckpointInspector
3. Viewing interrupt context with CheckpointDebugger
"""

import asyncio
from pathlib import Path

from pydantic import BaseModel

from pydantic_flow import AgentNode
from pydantic_flow import Flow
from pydantic_flow.checkpoints.backends.sqlite import SQLiteCheckpointBackend
from pydantic_flow.checkpoints.backends.sqlite import SQLiteCheckpointConfig
from pydantic_flow.checkpoints.debugger import CheckpointDebugger
from pydantic_flow.checkpoints.inspection import CheckpointInspector
from pydantic_flow.core.run_config import RunConfig
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
        result = await extract_result_from_stream(flow.astream(input_data))
        print("✅ Workflow completed without interruption")
        print(f"   Result: {result.text}\n")
    except InterruptionRequested as exc:
        print(f"Unexpected interruption: {exc.snapshot}\n")


async def example_with_persistence(
    flow: Flow,
    input_data: ContentInput,
    backend: SQLiteCheckpointBackend,
    run_id: str,
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

    # Configure run with checkpoint backend
    config = RunConfig(checkpoint_backend=backend, run_id=run_id)

    try:
        result = await extract_result_from_stream(
            flow.astream(input_data, config=config)
        )
        print(f"Unexpected success: {result}\n")
    except InterruptionRequested as exc:
        snapshot = exc.snapshot
        print("✋ Workflow interrupted for human review")
        print(f"   Run ID: {snapshot.run_id}")
        print(f"   Snapshot ID: {snapshot.snapshot_id}")
        print(f"   Interrupted at node: {snapshot.interrupted_node_id}")
        print(f"   Wave number: {snapshot.wave_number}")
        print("\n📦 Checkpoint automatically saved to V2 backend")
        print("   Interrupt reason: Final review required")
        state_count = len(snapshot.full_state) if snapshot.full_state else 0
        print(f"   State captured: {state_count} nodes")


async def example_query_interrupted_runs(
    inspector: CheckpointInspector,
    debugger: CheckpointDebugger,
) -> None:
    """Example 3: Query interrupted runs using V2 APIs."""
    print("\n=" * 60)
    print("Example 3: Query interrupted runs (V2 APIs)")
    print("=" * 60 + "\n")

    # List all interrupted runs
    interrupted_runs = await inspector.list_interrupted_runs(limit=10)

    print(f"📋 Found {len(interrupted_runs)} interrupted run(s)")

    if interrupted_runs:
        # Show details for the first interrupted run
        run = interrupted_runs[0]
        print(f"\n🔍 Inspecting run: {run.run_id}")
        print(f"   Flow ID: {run.flow_id}")
        print(f"   Status: {run.status}")
        print(f"   Started: {run.started_at}")
        print(f"   Total waves: {run.total_waves}")

        # Get the interrupt snapshot
        snapshot = await inspector.get_interrupt_snapshot(run.run_id)
        if snapshot:
            print("\n📸 Interrupt snapshot found:")
            print(f"   Snapshot ID: {snapshot.snapshot_id}")
            print(f"   Wave: {snapshot.wave_number}")
            print(f"   Reason: {snapshot.reason}")
            print(f"   Metadata: {snapshot.metadata}")

        # Show rich formatted context
        print("\n" + "=" * 60)
        print("Rich formatted interrupt context:")
        print("=" * 60)
        await debugger.show_interrupt_context(run.run_id)

    print("\n💡 In production, you would:")
    print("   1. Use CheckpointInspector to find interrupted runs")
    print("   2. Present the snapshot to a reviewer UI")
    print("   3. Wait for human approval/rejection")
    print("   4. Resume with flow.resume_from_snapshot()")


async def main() -> None:
    """Run simple HITL workflow with V2 checkpoint persistence."""
    # Create V2 checkpoint backend (in-memory SQLite)
    config = SQLiteCheckpointConfig(db_path=Path(":memory:"))
    backend = SQLiteCheckpointBackend(config)
    await backend.initialize()

    # Create inspector and debugger for querying
    inspector = CheckpointInspector(backend)
    debugger = CheckpointDebugger(backend)

    try:
        # Create a processing node
        processor = AgentNode.from_prompt(
            model="openai:gpt-4",
            prompt_template="Summarize this text: {text}",
            name="processor",
        )

        # Build flow
        flow = Flow(input_type=ContentInput, output_type=Summary)
        flow.add_nodes(processor)

        # Run examples
        await example_no_review(
            flow, ContentInput(text="The sky is blue.", requires_review=False)
        )

        await example_with_persistence(
            flow,
            ContentInput(text="Important company announcement.", requires_review=True),
            backend,
            run_id="review_run_001",
        )

        await example_query_interrupted_runs(inspector, debugger)

    finally:
        await backend.close()


if __name__ == "__main__":
    asyncio.run(main())
