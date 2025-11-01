"""Checkpoint v2 basic usage example.

This example demonstrates the core capabilities of checkpoint v2:
1. Automatic checkpoint persistence during flow execution
2. State reconstruction from checkpoints
3. Debugging interface (inspection + rendering)

NOTE: This is the v2 checkpoint system for debugging/time-travel.
For production resume/interrupt, see checkpoint_integration_test.py (v1 system).
"""

import asyncio
from pathlib import Path

from pydantic import BaseModel

from pydantic_flow import Flow
from pydantic_flow import PromptNode
from pydantic_flow.checkpoints.backends.sqlite import SQLiteCheckpointBackend
from pydantic_flow.checkpoints.backends.sqlite import SQLiteCheckpointConfig
from pydantic_flow.checkpoints.config import CheckpointConfig
from pydantic_flow.checkpoints.debugger import CheckpointDebugger
from pydantic_flow.core.run_config import RunConfig


class Question(BaseModel):
    """Input question."""

    text: str


class Answer(BaseModel):
    """Output answer."""

    response: str


async def run_flow_with_checkpoints() -> tuple[Answer, Path, SQLiteCheckpointBackend]:
    """Run flow with checkpoint v2 and return result."""
    # Create a simple flow
    flow = Flow(input_type=Question, output_type=Answer)
    node1 = PromptNode[Question, Answer](
        name="answer",
        prompt="Answer this question in one sentence: {text}",
    )
    flow.add_nodes(node1)

    # Set up checkpoint backend
    db_path = Path("checkpoints.db")
    backend = SQLiteCheckpointBackend(config=SQLiteCheckpointConfig(db_path=db_path))
    await backend.initialize()

    # Configure checkpoint
    config = RunConfig(
        checkpoint_backend=backend,
        checkpoint_config=CheckpointConfig(
            enabled=True,
            storage_backend=backend,
            save_full_snapshot_every=5,
            trace_sample_rate=1.0,
        ),
    )

    # Execute the flow
    result = await flow.run(
        Question(text="What is the capital of France?"), config=config
    )
    return result, db_path, backend


async def demonstrate_debugging(backend: SQLiteCheckpointBackend) -> None:
    """Demonstrate debugging interface."""
    debugger = CheckpointDebugger(backend)

    # List all runs
    print("📋 All runs in database:")
    await debugger.show_runs(limit=10)
    print()

    # Get the most recent run
    runs = await backend.list_runs(limit=1)
    if not runs:
        return

    run = runs[0]
    run_id = run.run_id

    print(f"🔍 Inspecting run: {run_id[:12]}...")
    print()

    # Show detailed run information
    await debugger.show_run_details(run_id)
    print()

    # Show wave timeline
    print("📊 Wave timeline:")
    await debugger.show_timeline(run_id)
    print()

    # Reconstruct state
    print("🔄 Reconstructing state at wave 0...")
    state = await debugger.get_state(run_id, wave=0)
    print(f"   State keys: {list(state.keys())}")
    print()

    # Get latest state
    print("📦 Latest state:")
    latest_state = await debugger.get_latest_state(run_id)
    if latest_state:
        print(f"   State keys: {list(latest_state.keys())}")
        for key, value in latest_state.items():
            print(f"   {key}: {value}")
    print()


async def main() -> None:
    """Demonstrate checkpoint v2 basic usage."""
    print("=" * 60)
    print("Checkpoint v2 Basic Usage Example")
    print("=" * 60)
    print()

    # Run flow with checkpoints
    print("🚀 Running flow with checkpoint v2 enabled...")
    print()
    result, db_path, backend = await run_flow_with_checkpoints()
    print(f"✅ Flow completed: {result.response}")
    print()

    # Demonstrate debugging
    print("=" * 60)
    print("Debugging Interface")
    print("=" * 60)
    print()
    await demonstrate_debugging(backend)

    # Cleanup
    await backend.close()
    if db_path.exists():
        db_path.unlink()

    print("=" * 60)
    print("✅ Example completed successfully!")
    print()
    print("💡 Key Features Demonstrated:")
    print("   - Automatic checkpoint persistence during execution")
    print("   - SQLite backend for local debugging")
    print("   - Debugging interface (list runs, show timeline)")
    print("   - State reconstruction from checkpoints")
    print()
    print("📝 Next Steps:")
    print("   - Try advanced time-travel features (Phase 4)")
    print("   - Explore CLI commands for debugging workflows")
    print("   - Use different backends (Filesystem, Postgres, S3)")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
