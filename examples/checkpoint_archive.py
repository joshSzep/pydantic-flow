"""Archive export/import example with checkpoint v2.

This example demonstrates checkpoint archive workflow using the API:
1. Run a flow and capture checkpoints
2. Use CheckpointDebugger.export_to_archive() for portable tar.gz export
3. Use CheckpointDebugger.load_from_archive() to import checkpoints
4. Verify imported checkpoints work correctly

This shows how to programmatically share debugging sessions across
environments, create reproducible test cases, and archive execution traces.
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


class Input(BaseModel):
    """Input data."""

    value: str


class Output(BaseModel):
    """Output data."""

    result: str


async def create_checkpoint_session(
    db_path: Path,
) -> tuple[SQLiteCheckpointBackend, RunId]:
    """Create checkpoint session and return backend, run_id."""
    flow = Flow(input_type=Input, output_type=Output)

    process = PromptNode[Input, Output](
        name="process",
        prompt="Process: {value}",
    )
    flow.add_nodes(process)
    flow.set_entry_nodes(process)

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

    await backend.initialize()
    input_data = Input(value="test data")
    result = await extract_result_from_stream(
        flow.astream(input_data, config=run_config)
    )
    print(f"✅ Created checkpoint session: {result.result[:50]}...")

    runs = await backend.list_runs(limit=1)
    if not runs:
        raise RuntimeError("No checkpoint created")

    run_id = runs[0].run_id
    print(f"📝 Run ID: {run_id[:12]}...")

    return backend, run_id


async def demonstrate_export(
    debugger: CheckpointDebugger, run_id: RunId, archive_path: Path
) -> None:
    """Demonstrate archive export."""
    print("\n" + "=" * 60)
    print("Exporting Checkpoint Archive")
    print("=" * 60)

    # 1. Show what we're exporting
    print("\n1. Checkpoint to export:")
    await debugger.show_timeline(run_id=run_id)

    # 2. Export to archive
    print("\n2. Exporting to archive...")
    await debugger.export_to_archive(
        run_id=run_id,
        output_path=str(archive_path),
    )

    size_kb = archive_path.stat().st_size / 1024
    print(f"   ✅ Exported {size_kb:.1f} KB to {archive_path}")

    # 3. Show archive contents
    print("\n3. Archive format:")
    print("   - metadata.json (run metadata)")
    print("   - snapshots/*.msgpack (binary state snapshots)")
    print("   - traces/*.json (execution traces)")


async def demonstrate_import(archive_path: Path, dest_db: Path) -> None:
    """Demonstrate archive import."""
    print("\n" + "=" * 60)
    print("Importing Checkpoint Archive")
    print("=" * 60)

    if dest_db.exists():
        dest_db.unlink()

    config = SQLiteCheckpointConfig(db_path=dest_db)
    backend = SQLiteCheckpointBackend(config=config)

    await backend.initialize()
    try:
        debugger = CheckpointDebugger(backend=backend)

        # 1. Import from archive
        print("\n1. Importing from archive...")
        imported_run_id = await debugger.load_from_archive(
            archive_path=str(archive_path)
        )
        print(f"   ✅ Imported run: {imported_run_id[:12]}...")

        # 2. Verify the import
        print("\n2. Verifying imported checkpoint:")
        await debugger.show_timeline(run_id=imported_run_id)  # type: ignore

        # 3. List imported runs
        print("\n3. All runs in new database:")
        await debugger.show_runs(limit=5)

        # 4. Get imported state
        print("\n4. Verifying imported state:")
        imported_state = await debugger.get_latest_state(run_id=imported_run_id)  # type: ignore
        if imported_state:
            print(f"   Nodes in imported checkpoint: {list(imported_state.keys())}")

    finally:
        await backend.close()


async def main() -> None:
    """Run archive export/import example."""
    print("=" * 60)
    print("Archive Export/Import Example")
    print("=" * 60)

    # Step 1: Create checkpoint session
    print("\n1. Creating checkpoint session...")
    source_db = Path("checkpoints_source.db")
    if source_db.exists():
        source_db.unlink()

    backend, run_id = await create_checkpoint_session(source_db)

    # Step 2: Export to archive
    archive_path = Path("checkpoint_export.tar.gz")
    if archive_path.exists():
        archive_path.unlink()

    try:
        debugger = CheckpointDebugger(backend=backend)
        await demonstrate_export(debugger, run_id, archive_path)
    finally:
        await backend.close()

    # Step 3: Import to new database
    dest_db = Path("checkpoints_imported.db")
    await demonstrate_import(archive_path, dest_db)

    print("\n" + "=" * 60)
    print("Archive Example Complete!")
    print("=" * 60)
    print("\n📁 Files created:")
    print(f"   Source DB: {source_db}")
    print(f"   Archive: {archive_path}")
    print(f"   Imported DB: {dest_db}")
    print("\n💡 Key Takeaways:")
    print("   - export_to_archive() creates portable tar.gz archives")
    print("   - load_from_archive() imports checkpoints to new databases")
    print("   - Archives use binary msgpack for type-safe serialization")
    print("   - Perfect for sharing debug sessions with teammates")
    print("\n📚 CLI equivalents:")
    print(f"   python -m pydantic_flow debug export {run_id[:12]} -o session.tar.gz")
    print("   python -m pydantic_flow debug import session.tar.gz --db new.db")


if __name__ == "__main__":
    asyncio.run(main())
