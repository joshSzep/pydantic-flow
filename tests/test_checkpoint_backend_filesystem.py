"""Tests for FilesystemCheckpointBackend."""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
from pathlib import Path
import tempfile
import uuid

from pydantic import BaseModel
import pytest

from pydantic_flow.checkpoints.backends.filesystem import FilesystemCheckpointBackend
from pydantic_flow.checkpoints.backends.filesystem import FilesystemCheckpointConfig
from pydantic_flow.checkpoints.types import EventSummary
from pydantic_flow.checkpoints.types import ExecutionTrace
from pydantic_flow.checkpoints.types import NodeExecutionTrace
from pydantic_flow.checkpoints.types import RunMetadata
from pydantic_flow.checkpoints.types import StateRef
from pydantic_flow.checkpoints.types import StateSnapshot
from pydantic_flow.checkpoints.types import generate_run_id
from pydantic_flow.checkpoints.types import generate_snapshot_id


class SampleState(BaseModel):
    """Sample state model for testing."""

    value: int
    name: str


@pytest.fixture
async def filesystem_backend():
    """Create filesystem backend with temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = FilesystemCheckpointConfig(root_dir=Path(tmpdir))
        backend = FilesystemCheckpointBackend(config)
        await backend.initialize()
        yield backend
        await backend.close()


@pytest.mark.asyncio
async def test_filesystem_healthcheck(filesystem_backend):
    """Test backend health check."""
    assert await filesystem_backend.healthcheck()


@pytest.mark.asyncio
async def test_save_and_get_snapshot(filesystem_backend):
    """Test saving and retrieving state snapshots."""
    run_id = generate_run_id()
    snapshot = StateSnapshot(
        snapshot_id=generate_snapshot_id(),
        run_id=run_id,
        wave_number=0,
        full_state={"node1": SampleState(value=1, name="test")},
        state_hash="hash123",
        next_frontier=["node2"],
        routing_ended=False,
    )

    await filesystem_backend.save_state_snapshot(snapshot)

    retrieved = await filesystem_backend.get_state_snapshot(run_id, 0)
    assert retrieved is not None
    assert retrieved.wave_number == 0
    assert retrieved.run_id == run_id
    assert "node1" in retrieved.full_state
    assert isinstance(retrieved.full_state["node1"], SampleState)


@pytest.mark.asyncio
async def test_get_snapshots_range(filesystem_backend):
    """Test batch fetching snapshots."""
    run_id = generate_run_id()

    # Create 5 snapshots
    for i in range(5):
        snapshot = StateSnapshot(
            snapshot_id=generate_snapshot_id(),
            run_id=run_id,
            wave_number=i,
            full_state={"node1": SampleState(value=i, name=f"wave_{i}")},
            state_hash=f"hash{i}",
            next_frontier=[],
            routing_ended=False,
        )
        await filesystem_backend.save_state_snapshot(snapshot)

    # Fetch range
    snapshots = await filesystem_backend.get_snapshots_range(run_id, 1, 3)
    assert len(snapshots) == 3
    assert snapshots[0].wave_number == 1
    assert snapshots[-1].wave_number == 3


@pytest.mark.asyncio
async def test_save_and_get_trace(filesystem_backend):
    """Test saving and retrieving execution traces."""
    run_id = generate_run_id()
    snapshot_id = generate_snapshot_id()

    # Create checkpoint first
    snapshot = StateSnapshot(
        snapshot_id=snapshot_id,
        run_id=run_id,
        wave_number=0,
        full_state={"node1": SampleState(value=1, name="test")},
        state_hash="hash123",
        next_frontier=[],
        routing_ended=False,
    )
    await filesystem_backend.save_state_snapshot(snapshot)

    # Create trace
    trace = ExecutionTrace(
        trace_id=str(uuid.uuid4()),
        run_id=run_id,
        wave_number=0,
        checkpoint_snapshot_id=snapshot_id,
        node_traces=[
            NodeExecutionTrace(
                log_id=str(uuid.uuid4()),
                node_id="node1",
                wave_number=0,
                snapshot_id=snapshot_id,
                input_ref=StateRef(snapshot_id=snapshot_id, state_key="node1"),
                output_ref=StateRef(snapshot_id=snapshot_id, state_key="node1"),
                event_log_id=str(uuid.uuid4()),
                total_events=0,
                event_summary=EventSummary(
                    total_events=0,
                    token_count=0,
                    tool_call_count=0,
                    cache_hits=0,
                    tool_calls=[],
                ),
                started_at=datetime.now(UTC),
                completed_at=datetime.now(UTC),
                next_nodes=[],
            )
        ],
        parallel_batch_id="batch1",
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )

    await filesystem_backend.save_trace(trace)

    retrieved = await filesystem_backend.get_trace(run_id, 0)
    assert retrieved is not None
    assert retrieved.wave_number == 0
    assert len(retrieved.node_traces) == 1


@pytest.mark.asyncio
async def test_list_runs(filesystem_backend):
    """Test listing runs."""
    # Create multiple runs
    for i in range(3):
        run_id = generate_run_id()
        metadata = RunMetadata(
            run_id=run_id,
            flow_id=f"flow_{i}",
            started_at=datetime.now(UTC),
            status=RunMetadata.Status.COMPLETED,
        )
        await filesystem_backend.save_run_metadata(metadata)

    runs = await filesystem_backend.list_runs()
    assert len(runs) == 3


@pytest.mark.asyncio
async def test_delete_run(filesystem_backend):
    """Test deleting run data."""
    run_id = generate_run_id()

    metadata = RunMetadata(
        run_id=run_id,
        flow_id="test_flow",
        started_at=datetime.now(UTC),
        status=RunMetadata.Status.COMPLETED,
    )
    await filesystem_backend.save_run_metadata(metadata)

    snapshot = StateSnapshot(
        snapshot_id=generate_snapshot_id(),
        run_id=run_id,
        wave_number=0,
        full_state={"node1": SampleState(value=1, name="test")},
        state_hash="hash",
        next_frontier=[],
        routing_ended=False,
    )
    await filesystem_backend.save_state_snapshot(snapshot)

    # Delete run
    await filesystem_backend.delete_run(run_id)

    assert await filesystem_backend.get_run_metadata(run_id) is None
    assert await filesystem_backend.get_state_snapshot(run_id, 0) is None


@pytest.mark.asyncio
async def test_directory_structure(filesystem_backend):
    """Test correct directory structure is created."""
    run_id = generate_run_id()
    snapshot_id = generate_snapshot_id()

    # Create metadata
    metadata = RunMetadata(
        run_id=run_id,
        flow_id="test_flow",
        started_at=datetime.now(UTC),
        status=RunMetadata.Status.RUNNING,
    )
    await filesystem_backend.save_run_metadata(metadata)

    # Create snapshot
    snapshot = StateSnapshot(
        snapshot_id=snapshot_id,
        run_id=run_id,
        wave_number=0,
        full_state={"node1": SampleState(value=1, name="test")},
        state_hash="hash",
        next_frontier=[],
        routing_ended=False,
    )
    await filesystem_backend.save_state_snapshot(snapshot)

    # Verify structure
    run_dir = filesystem_backend._run_dir(run_id)
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "checkpoints" / "wave_0000.msgpack.gz").exists()
