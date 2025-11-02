"""Fork and branch execution example with checkpoint v2.

This example demonstrates checkpoint forking workflow using the API:
1. Run a flow and capture checkpoints
2. Use CheckpointDebugger.fork_from_wave() to create execution branches
3. Modify state during fork to explore alternatives
4. Compare original and forked execution paths

This shows how to programmatically fork executions and explore "what-if"
scenarios without re-running expensive computation.
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


# Helper to extract result from stream
async def extract_result_from_stream(stream):
    """Extract final result from async stream of progress items."""
    result = None
    async for item in stream:
        if hasattr(item, "result"):
            result = item.result
    return result


class Query(BaseModel):
    """User query."""

    text: str


class Strategy(BaseModel):
    """Analysis strategy."""

    approach: str


class Result(BaseModel):
    """Final result."""

    answer: str


async def run_initial_flow(
    db_path: Path,
) -> tuple[SQLiteCheckpointBackend, RunId]:
    """Run initial flow and return backend, run_id."""
    flow = Flow(input_type=Query, output_type=Result)

    # Step 1: Choose strategy
    strategize = PromptNode[Query, Strategy](
        name="strategize",
        prompt="Choose an approach for: {text}",
    )

    # Step 2: Execute strategy
    execute = PromptNode[Strategy, Result](
        name="execute",
        prompt="Execute {approach} approach",
    )

    flow.add_nodes(strategize, execute)
    flow.add_edge(strategize, execute)
    flow.set_entry_nodes(strategize)

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

    query = Query(text="Optimize database performance")

    await backend.initialize()
    result = await extract_result_from_stream(flow.astream(query, config=run_config))
    print("✅ Initial flow completed")
    print(f"   Result: {result.answer[:100]}...")

    runs = await backend.list_runs(limit=1)
    if not runs:
        raise RuntimeError("No checkpoint created")

    run_id = runs[0].run_id
    print(f"\n📝 Run ID: {run_id[:12]}...")

    return backend, run_id


async def demonstrate_forking(debugger: CheckpointDebugger, run_id: RunId) -> None:
    """Demonstrate forking and state branching."""
    print("\n" + "=" * 60)
    print("Forking and Branching with CheckpointDebugger")
    print("=" * 60)

    # 1. Show original timeline
    print("\n1. Original execution timeline:")
    await debugger.show_timeline(run_id=run_id)

    # 2. Fork from wave 1 (after strategy selection)
    print("\n2. Forking from wave 1 (creating alternative branch)...")
    print("   This creates a new execution path from the checkpoint")

    fork_result = await debugger.fork_from_wave(
        run_id=run_id,
        source_wave=1,
        state_modifications=None,  # Could modify state here
    )

    forked_run_id = fork_result["forked_run_id"]
    print(f"   ✅ Created fork: {forked_run_id[:12]}...")

    # 3. Compare timelines
    print("\n3. Comparing original vs forked execution:")
    print("\n   Original Run:")
    await debugger.show_timeline(run_id=run_id)

    print("\n   Forked Run:")
    await debugger.show_timeline(run_id=forked_run_id)

    # 4. Show state at fork point
    print("\n4. State at fork point (wave 1):")
    original_state = await debugger.get_state(run_id=run_id, wave=1)
    forked_state = await debugger.get_state(run_id=forked_run_id, wave=1)

    print(f"   Original nodes: {list(original_state.keys())}")
    print(f"   Forked nodes: {list(forked_state.keys())}")

    # 5. Fork with state modifications
    print("\n5. Creating fork WITH state modifications:")
    print("   Modifying 'strategize' node to try different approach...")

    modified_strategy = Strategy(approach="alternative caching strategy")
    modified_fork_result = await debugger.fork_from_wave(
        run_id=run_id,
        source_wave=1,
        state_modifications={"strategize": modified_strategy},
    )

    modified_run_id = modified_fork_result["forked_run_id"]
    print(f"   ✅ Created modified fork: {modified_run_id[:12]}...")

    # 6. Show modified fork timeline
    print("\n6. Modified fork timeline:")
    await debugger.show_timeline(run_id=modified_run_id)

    # 7. List all runs to see branches
    print("\n7. All runs (showing branching):")
    await debugger.show_runs(limit=10)


async def main() -> None:
    """Run fork and branch example."""
    print("=" * 60)
    print("Fork and Branch Execution Example")
    print("=" * 60)

    db_path = Path("checkpoints_fork.db")
    if db_path.exists():
        db_path.unlink()

    print("\nRunning initial flow with checkpoints...")
    backend, run_id = await run_initial_flow(db_path)

    try:
        debugger = CheckpointDebugger(backend=backend)
        await demonstrate_forking(debugger, run_id)
    finally:
        await backend.close()

    print("\n" + "=" * 60)
    print("Fork Example Complete!")
    print("=" * 60)
    print(f"\nCheckpoint database: {db_path}")
    print("\n💡 Key Takeaways:")
    print("   - fork_from_wave() creates execution branches")
    print("   - Pass state_modifications to explore alternatives")
    print("   - Forks share history up to the fork point")
    print("   - Enables A/B testing without re-running expensive steps")
    print("\n📚 CLI equivalent:")
    print(f"   python -m pydantic_flow debug fork {run_id[:12]} --from-wave 1")


if __name__ == "__main__":
    asyncio.run(main())
