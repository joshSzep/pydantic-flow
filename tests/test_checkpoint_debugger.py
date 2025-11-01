"""Tests for CheckpointDebugger advanced debugging features."""

from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path

from pydantic import BaseModel
import pytest

from pydantic_flow.checkpoints.backends.sqlite import SQLiteCheckpointBackend
from pydantic_flow.checkpoints.backends.sqlite import SQLiteCheckpointConfig
from pydantic_flow.checkpoints.debugger import CheckpointDebugger
from pydantic_flow.checkpoints.types import EventSummary
from pydantic_flow.checkpoints.types import ExecutionTrace
from pydantic_flow.checkpoints.types import NodeExecutionTrace
from pydantic_flow.checkpoints.types import RunMetadata
from pydantic_flow.checkpoints.types import StateRef
from pydantic_flow.checkpoints.types import StateSnapshot
from pydantic_flow.checkpoints.types import generate_run_id
from pydantic_flow.checkpoints.types import generate_snapshot_id


class SimpleState(BaseModel):
    """Simple state model for testing."""

    value: int
    name: str


@pytest.fixture
async def backend(tmp_path: Path):
    """Create and initialize SQLite backend."""
    db_path = tmp_path / "test_debugger.db"
    config = SQLiteCheckpointConfig(db_path=db_path)
    backend = SQLiteCheckpointBackend(config=config)
    await backend.initialize()
    yield backend
    await backend.close()


@pytest.fixture
async def debugger(backend: SQLiteCheckpointBackend):
    """Create CheckpointDebugger instance."""
    return CheckpointDebugger(backend=backend)


@pytest.fixture
async def sample_run(backend: SQLiteCheckpointBackend):
    """Create a sample run with multiple waves for testing."""
    run_id = generate_run_id()
    now = datetime.now(UTC)

    # Create run metadata
    metadata = RunMetadata(
        run_id=run_id,
        flow_id="test_flow",
        started_at=now,
        status=RunMetadata.Status.RUNNING,
        total_waves=3,
    )
    await backend.save_run_metadata(metadata)

    # Create snapshots for waves 0, 1, 2
    snapshots = []
    for wave in range(3):
        snapshot_id = generate_snapshot_id()

        state: dict[str, BaseModel] = {
            "test_node": SimpleState(value=wave * 10, name=f"wave_{wave}")
        }

        snapshot = StateSnapshot(
            snapshot_id=snapshot_id,
            run_id=run_id,
            wave_number=wave,
            full_state=state,
            state_hash="test_hash",
            next_frontier=["next_node"],
            routing_ended=False,
            created_at=now + timedelta(seconds=wave),
        )
        await backend.save_state_snapshot(snapshot)
        snapshots.append(snapshot)

        # Create execution trace for this wave
        node_trace = NodeExecutionTrace(
            log_id=f"log_{wave}",
            node_id="test_node",
            wave_number=wave,
            snapshot_id=snapshot_id,
            input_ref=StateRef(snapshot_id=snapshot_id, state_key="input"),
            output_ref=StateRef(snapshot_id=snapshot_id, state_key="output"),
            event_log_id=f"event_log_{wave}",
            total_events=5,
            event_summary=EventSummary(
                total_events=5,
                token_count=100,
                tool_call_count=2,
                cache_hits=0,
                tool_calls=["tool_a", "tool_b"],
            ),
            started_at=now + timedelta(seconds=wave, milliseconds=0),
            completed_at=now + timedelta(seconds=wave, milliseconds=500),
            next_nodes=["next_node"],
        )

        trace = ExecutionTrace(
            trace_id=f"trace_{wave}",
            run_id=run_id,
            wave_number=wave,
            checkpoint_snapshot_id=snapshot_id,
            node_traces=[node_trace],
            parallel_batch_id=f"batch_{wave}",
            started_at=now + timedelta(seconds=wave, milliseconds=0),
            completed_at=now + timedelta(seconds=wave, milliseconds=500),
        )
        await backend.save_trace(trace)

    return {
        "run_id": run_id,
        "snapshots": snapshots,
        "metadata": metadata,
    }


async def test_replay_from_checkpoint(debugger: CheckpointDebugger, sample_run: dict):
    """Test replaying execution from a checkpoint."""
    run_id = sample_run["run_id"]

    # Replay wave 1
    result = await debugger.replay_from_checkpoint(
        run_id=run_id, wave=1, show_events=False
    )

    assert result is not None
    assert "snapshot" in result
    assert "trace" in result
    assert "node_count" in result

    assert result["snapshot"].wave_number == 1
    assert result["trace"].wave_number == 1
    assert result["node_count"] == 1

    # Verify trace details
    trace = result["trace"]
    assert len(trace.node_traces) == 1
    assert trace.node_traces[0].node_id == "test_node"
    assert trace.node_traces[0].total_events == 5
    assert trace.node_traces[0].event_summary.token_count == 100


async def test_replay_nonexistent_wave(debugger: CheckpointDebugger, sample_run: dict):
    """Test replaying a nonexistent wave raises ValueError."""
    run_id = sample_run["run_id"]

    with pytest.raises(ValueError, match="No snapshot found"):
        await debugger.replay_from_checkpoint(run_id=run_id, wave=999)


async def test_rewind_to_wave(debugger: CheckpointDebugger, sample_run: dict):
    """Test rewinding to a previous wave."""
    run_id = sample_run["run_id"]

    # Rewind to wave 0
    result = await debugger.rewind_to_wave(run_id=run_id, target_wave=0)

    assert result is not None
    assert result["wave"] == 0
    assert "state" in result
    assert "snapshot" in result
    assert "next_waves" in result

    # Should have waves 1 and 2 as next waves
    assert 1 in result["next_waves"]
    assert 2 in result["next_waves"]

    # Verify state was reconstructed
    state = result["state"]
    assert "test_node" in state
    assert state["test_node"].value == 0
    assert state["test_node"].name == "wave_0"


async def test_rewind_to_middle_wave(debugger: CheckpointDebugger, sample_run: dict):
    """Test rewinding to a middle wave."""
    run_id = sample_run["run_id"]

    # Rewind to wave 1
    result = await debugger.rewind_to_wave(run_id=run_id, target_wave=1)

    assert result["wave"] == 1
    assert 2 in result["next_waves"]

    # Verify state
    state = result["state"]
    assert state["test_node"].value == 10
    assert state["test_node"].name == "wave_1"


async def test_rewind_to_latest_wave(debugger: CheckpointDebugger, sample_run: dict):
    """Test rewinding to the latest wave."""
    run_id = sample_run["run_id"]

    # Rewind to wave 2 (latest)
    result = await debugger.rewind_to_wave(run_id=run_id, target_wave=2)

    assert result["wave"] == 2
    assert result["next_waves"] == []  # No waves after this

    # Verify state
    state = result["state"]
    assert state["test_node"].value == 20
    assert state["test_node"].name == "wave_2"


async def test_rewind_nonexistent_wave(debugger: CheckpointDebugger, sample_run: dict):
    """Test rewinding to nonexistent wave raises ValueError."""
    run_id = sample_run["run_id"]

    with pytest.raises(ValueError, match="No snapshot found"):
        await debugger.rewind_to_wave(run_id=run_id, target_wave=999)


async def test_show_runs_with_sample_run(
    debugger: CheckpointDebugger, sample_run: dict
):
    """Test show_runs includes the sample run."""
    # This tests the integration - should not raise
    await debugger.show_runs(limit=10)


async def test_show_timeline_with_sample_run(
    debugger: CheckpointDebugger, sample_run: dict
):
    """Test show_timeline displays the wave timeline."""
    run_id = sample_run["run_id"]

    # This tests the integration - should not raise
    await debugger.show_timeline(run_id=run_id)


async def test_show_run_details_with_sample_run(
    debugger: CheckpointDebugger, sample_run: dict
):
    """Test show_run_details displays run information."""
    run_id = sample_run["run_id"]

    # This tests the integration - should not raise
    await debugger.show_run_details(run_id=run_id)


async def test_get_state_at_wave(debugger: CheckpointDebugger, sample_run: dict):
    """Test getting state at a specific wave."""
    run_id = sample_run["run_id"]

    state = await debugger.get_state(run_id=run_id, wave=1)

    assert "test_node" in state
    assert state["test_node"].value == 10
    assert state["test_node"].name == "wave_1"


async def test_get_latest_state(debugger: CheckpointDebugger, sample_run: dict):
    """Test getting the latest state."""
    run_id = sample_run["run_id"]

    state = await debugger.get_latest_state(run_id=run_id)

    assert state is not None
    assert "test_node" in state
    assert state["test_node"].value == 20
    assert state["test_node"].name == "wave_2"


async def test_get_latest_state_empty_run(backend: SQLiteCheckpointBackend):
    """Test getting latest state for run with no waves."""
    debugger = CheckpointDebugger(backend=backend)
    run_id = generate_run_id()

    # Create run metadata but no snapshots
    metadata = RunMetadata(
        run_id=run_id,
        flow_id="test_flow",
        started_at=datetime.now(UTC),
        status=RunMetadata.Status.RUNNING,
    )
    await backend.save_run_metadata(metadata)

    state = await debugger.get_latest_state(run_id=run_id)
    assert state is None


async def test_fork_from_wave(debugger: CheckpointDebugger, sample_run: dict):
    """Test forking from a checkpoint wave without modifications."""
    run_id = sample_run["run_id"]

    result = await debugger.fork_from_wave(run_id=run_id, source_wave=1)

    assert result["source_run_id"] == run_id
    assert result["source_wave"] == 1
    assert "forked_state" in result
    assert "test_node" in result["forked_state"]
    assert result["modifications"] == {}


async def test_fork_with_modifications(debugger: CheckpointDebugger, sample_run: dict):
    """Test forking with state modifications."""
    run_id = sample_run["run_id"]

    new_state = SimpleState(value=999, name="modified")
    modifications: dict[str, BaseModel] = {"test_node": new_state}

    result = await debugger.fork_from_wave(
        run_id=run_id, source_wave=1, state_modifications=modifications
    )

    assert result["source_run_id"] == run_id
    assert result["source_wave"] == 1
    assert result["forked_state"]["test_node"] == new_state
    assert "test_node" in result["modifications"]
    assert result["modifications"]["test_node"]["new"] == new_state


async def test_fork_invalid_wave(debugger: CheckpointDebugger):
    """Test forking from non-existent wave raises error."""
    run_id = generate_run_id()

    with pytest.raises(ValueError, match="No snapshot found"):
        await debugger.fork_from_wave(run_id=run_id, source_wave=999)


async def test_fork_invalid_node(debugger: CheckpointDebugger, sample_run: dict):
    """Test forking with invalid node ID raises error."""
    run_id = sample_run["run_id"]

    modifications: dict[str, BaseModel] = {
        "nonexistent_node": SimpleState(value=1, name="test")
    }

    with pytest.raises(ValueError, match="not found in state"):
        await debugger.fork_from_wave(
            run_id=run_id, source_wave=1, state_modifications=modifications
        )


async def test_fork_preserves_unmodified_nodes(
    debugger: CheckpointDebugger, sample_run: dict
):
    """Test that forking preserves nodes that aren't modified."""
    run_id = sample_run["run_id"]

    new_state = SimpleState(value=777, name="changed")
    modifications: dict[str, BaseModel] = {"test_node": new_state}

    result = await debugger.fork_from_wave(
        run_id=run_id, source_wave=1, state_modifications=modifications
    )

    forked_state = result["forked_state"]
    assert len(forked_state) == 1
    assert forked_state["test_node"] == new_state


async def test_export_to_archive(
    debugger: CheckpointDebugger, sample_run: dict, tmp_path
):
    """Test exporting a run to a tar.gz archive."""
    run_id = sample_run["run_id"]
    archive_path = tmp_path / "export.tar.gz"

    result = await debugger.export_to_archive(
        run_id=run_id, output_path=str(archive_path)
    )

    assert result["run_id"] == run_id
    assert result["snapshot_count"] == 3
    assert result["total_size_bytes"] > 0
    assert archive_path.exists()


async def test_export_nonexistent_run(debugger: CheckpointDebugger, tmp_path):
    """Test exporting a non-existent run raises error."""
    run_id = generate_run_id()
    archive_path = tmp_path / "export.tar.gz"

    with pytest.raises(ValueError, match="not found"):
        await debugger.export_to_archive(run_id=run_id, output_path=str(archive_path))


async def test_load_from_archive(
    debugger: CheckpointDebugger, sample_run: dict, tmp_path
):
    """Test loading a run from an archive."""
    run_id = sample_run["run_id"]
    archive_path = tmp_path / "export.tar.gz"

    # First export
    await debugger.export_to_archive(run_id=run_id, output_path=str(archive_path))

    # Create new backend/debugger to test import
    new_db_path = tmp_path / "imported.db"
    new_config = SQLiteCheckpointConfig(db_path=new_db_path)
    new_backend = SQLiteCheckpointBackend(config=new_config)

    try:
        await new_backend.initialize()
        new_debugger = CheckpointDebugger(backend=new_backend)

        # Import
        result = await new_debugger.load_from_archive(str(archive_path))

        assert result["run_id"] == run_id
        assert result["snapshot_count"] == 3
        assert result["metadata"].run_id == run_id

        # Verify snapshots are accessible
        state = await new_debugger.get_latest_state(run_id=run_id)
        assert state is not None
    finally:
        await new_backend.close()


async def test_load_from_nonexistent_archive(debugger: CheckpointDebugger, tmp_path):
    """Test loading from non-existent archive raises error."""
    archive_path = tmp_path / "nonexistent.tar.gz"

    with pytest.raises(ValueError, match="not found"):
        await debugger.load_from_archive(str(archive_path))


async def test_roundtrip_export_import(
    debugger: CheckpointDebugger, sample_run: dict, tmp_path
):
    """Test export-import roundtrip preserves data."""
    run_id = sample_run["run_id"]
    archive_path = tmp_path / "roundtrip.tar.gz"

    # Export
    export_result = await debugger.export_to_archive(
        run_id=run_id, output_path=str(archive_path)
    )

    # Import into new backend
    new_db_path = tmp_path / "roundtrip.db"
    new_config = SQLiteCheckpointConfig(db_path=new_db_path)
    new_backend = SQLiteCheckpointBackend(config=new_config)

    try:
        await new_backend.initialize()
        new_debugger = CheckpointDebugger(backend=new_backend)

        import_result = await new_debugger.load_from_archive(str(archive_path))

        # Verify counts match
        assert import_result["run_id"] == export_result["run_id"]
        assert import_result["snapshot_count"] == export_result["snapshot_count"]

        # Verify we can replay from imported data
        result = await new_debugger.replay_from_checkpoint(
            run_id=run_id, wave=1, show_events=False
        )
        assert result["snapshot"].wave_number == 1
        assert result["node_count"] > 0
    finally:
        await new_backend.close()
