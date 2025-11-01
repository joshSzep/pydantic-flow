"""Time-travel debugging example with checkpoint v2.

This example demonstrates the checkpoint v2 debugging API:
1. Run a flow and capture checkpoints automatically
2. Use CheckpointDebugger to inspect execution history
3. Replay specific waves to see what happened
4. Rewind to previous states (time-travel debugging)
5. Compare state evolution across waves

This shows how to programmatically debug flows using the checkpoint v2 API.
"""

import asyncio
from pathlib import Path

from pydantic import BaseModel

from pydantic_flow import Flow
from pydantic_flow import PromptNode
from pydantic_flow.checkpoints import CheckpointDebugger
from pydantic_flow.checkpoints import SQLiteCheckpointBackend
from pydantic_flow.checkpoints import SQLiteCheckpointConfig
from pydantic_flow.checkpoints.config import CheckpointConfig
from pydantic_flow.checkpoints.types import RunId
from pydantic_flow.core.run_config import RunConfig


class Question(BaseModel):
    """A question to answer."""

    text: str


class Analysis(BaseModel):
    """Analysis of the question."""

    category: str
    complexity: str


class Answer(BaseModel):
    """Final answer."""

    response: str
    confidence: float


async def run_flow_with_checkpoints(
    db_path: Path,
) -> tuple[SQLiteCheckpointBackend, RunId]:
    """Run flow and return backend, run_id."""
    flow = Flow(input_type=Question, output_type=Answer)

    analyze = PromptNode[Question, Analysis](
        name="analyze",
        prompt=(
            "Analyze this question:\n{text}\n\n"
            "Provide category and complexity (simple/medium/complex)."
        ),
    )

    answer = PromptNode[Analysis, Answer](
        name="answer",
        prompt=(
            "Generate an answer for a {category} question.\n"
            "Complexity level: {complexity}\n\n"
            "Provide response and confidence score (0-1)."
        ),
    )

    flow.add_nodes(analyze, answer)
    flow.add_edge(analyze, answer)
    flow.set_entry_nodes(analyze)

    config = SQLiteCheckpointConfig(db_path=db_path)
    backend = SQLiteCheckpointBackend(config=config)

    checkpoint_config = CheckpointConfig(
        enabled=True,
        storage_backend=backend,
        trace_sample_rate=1.0,
        save_full_snapshot_every=1,
    )

    run_config = RunConfig(
        checkpoint_backend=backend,
        checkpoint_config=checkpoint_config,
    )

    question = Question(text="What is the capital of France?")

    await backend.initialize()
    result = await flow.run(question, config=run_config)
    print("✅ Flow completed successfully")
    print(f"   Answer: {result.response}")
    print(f"   Confidence: {result.confidence}")

    runs = await backend.list_runs(limit=1)
    if not runs:
        raise RuntimeError("No checkpoint created")

    run_id = runs[0].run_id
    print(f"\n📝 Checkpoint saved - Run ID: {run_id[:12]}...")

    return backend, run_id


async def demonstrate_debugging(debugger: CheckpointDebugger, run_id: RunId) -> None:
    """Demonstrate various debugging operations."""
    print("\n" + "=" * 60)
    print("Time-Travel Debugging with CheckpointDebugger")
    print("=" * 60)

    # 1. List all runs
    print("\n1. Listing all checkpoint runs:")
    await debugger.show_runs(limit=5)

    # 2. Show execution timeline
    print("\n2. Execution timeline:")
    await debugger.show_timeline(run_id=run_id)

    # 3. Get state at wave 1
    print("\n3. Getting state at wave 1 (after analysis):")
    state_wave_1 = await debugger.get_state(run_id=run_id, wave=1)
    for node_id, node_state in state_wave_1.items():
        print(f"   {node_id}: {node_state}")

    # 4. Rewind to wave 1
    print("\n4. Rewinding to wave 1 (time-travel debugging):")
    rewind_result = await debugger.rewind_to_wave(run_id=run_id, target_wave=1)
    print(f"   Restored to wave {rewind_result['target_wave']}")
    print(f"   State keys: {list(rewind_result['state'].keys())}")

    # 5. Compare state evolution
    print("\n5. State evolution across waves:")
    state_wave_0 = await debugger.get_state(run_id=run_id, wave=0)
    state_wave_2 = await debugger.get_latest_state(run_id=run_id)

    print(f"   Wave 0 (initial): {list(state_wave_0.keys())}")
    print(f"   Wave 1 (analysis): {list(state_wave_1.keys())}")
    if state_wave_2:
        print(f"   Wave 2 (final): {list(state_wave_2.keys())}")

    # 6. Show detailed run information
    print("\n6. Detailed run information:")
    await debugger.show_run_details(run_id=run_id)

    # 7. Replay from checkpoint
    print("\n7. Replaying execution from wave 1:")
    await debugger.replay_from_checkpoint(run_id=run_id, wave=1)


async def main() -> None:
    """Run time-travel debugging example."""
    print("=" * 60)
    print("Time-Travel Debugging Example")
    print("=" * 60)

    db_path = Path("checkpoints_timetravel.db")
    if db_path.exists():
        db_path.unlink()

    print("\nRunning flow with checkpoints enabled...")
    backend, run_id = await run_flow_with_checkpoints(db_path)

    try:
        debugger = CheckpointDebugger(backend=backend)
        await demonstrate_debugging(debugger, run_id)
    finally:
        await backend.close()

    print("\n" + "=" * 60)
    print("Time-Travel Debugging Complete!")
    print("=" * 60)
    print(f"\nCheckpoint database: {db_path}")
    print("\n💡 Key Takeaways:")
    print("   - CheckpointDebugger provides programmatic access")
    print("   - get_state() retrieves state at any wave")
    print("   - rewind_to_wave() enables time-travel")
    print("   - show_timeline() visualizes execution")
    print("   - replay_from_checkpoint() re-executes from any point")
    print("\n📚 CLI equivalents:")
    print(f"   python -m pydantic_flow debug timeline {run_id[:12]}")
    print(f"   python -m pydantic_flow debug rewind {run_id[:12]} --to-wave 1")


if __name__ == "__main__":
    asyncio.run(main())
