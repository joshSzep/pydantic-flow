"""Integration tests for SQLite checkpoint backend (Phase 2)."""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
from pathlib import Path
import tempfile
import uuid

from pydantic import BaseModel
import pytest

from pydantic_flow.checkpoints import CheckpointIntegrityError
from pydantic_flow.checkpoints import SQLiteCheckpointBackend
from pydantic_flow.checkpoints import SQLiteCheckpointConfig
from pydantic_flow.checkpoints import StateReconstructor
from pydantic_flow.checkpoints import validate_and_save_trace
from pydantic_flow.checkpoints import validate_checkpoint_integrity
from pydantic_flow.checkpoints.delta import DeltaComputer
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
async def backend():
    """Create SQLite backend with temporary database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = SQLiteCheckpointConfig(
            db_path=Path(tmpdir) / "test.db",
            wal_mode=True,
        )
        backend = SQLiteCheckpointBackend(config)
        await backend.initialize()
        yield backend
        await backend.close()


@pytest.mark.asyncio
async def test_backend_healthcheck(backend):
    """Test backend health check."""
    assert await backend.healthcheck()


@pytest.mark.asyncio
async def test_save_and_get_snapshot(backend):
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

    await backend.save_state_snapshot(snapshot)

    retrieved = await backend.get_state_snapshot(run_id, 0)
    assert retrieved is not None
    assert retrieved.wave_number == 0
    assert retrieved.run_id == run_id
    assert "node1" in retrieved.full_state
    assert isinstance(retrieved.full_state["node1"], SampleState)


@pytest.mark.asyncio
async def test_get_snapshots_range(backend):
    """Test batch fetching snapshots."""
    run_id = generate_run_id()

    # Create 15 snapshots
    for i in range(15):
        base_state = {"node1": SampleState(value=i, name=f"wave_{i}")}
        snapshot = StateSnapshot(
            snapshot_id=generate_snapshot_id(),
            run_id=run_id,
            wave_number=i,
            full_state=base_state if i % 10 == 0 else None,
            forward_delta=(
                {} if i == 0 else {"node1": SampleState(value=i, name=f"wave_{i}")}
            ),
            state_hash=f"hash{i}",
            next_frontier=[],
            routing_ended=False,
        )
        await backend.save_state_snapshot(snapshot)

    # Fetch range
    snapshots = await backend.get_snapshots_range(run_id, 5, 12, order="ASC")

    assert len(snapshots) == 8  # Waves 5-12 inclusive
    assert snapshots[0].wave_number == 5
    assert snapshots[-1].wave_number == 12

    # Test descending order
    snapshots_desc = await backend.get_snapshots_range(run_id, 5, 12, order="DESC")
    assert len(snapshots_desc) == 8
    assert snapshots_desc[0].wave_number == 12
    assert snapshots_desc[-1].wave_number == 5


@pytest.mark.asyncio
async def test_save_and_get_trace(backend):
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
    await backend.save_state_snapshot(snapshot)

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
                event_log_id="log123",
                total_events=5,
                event_summary=EventSummary(
                    total_events=5,
                    token_count=100,
                    tool_call_count=1,
                    cache_hits=0,
                    tool_calls=["tool1"],
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

    await backend.save_trace(trace)

    retrieved = await backend.get_trace(run_id, 0)
    assert retrieved is not None
    assert retrieved.trace_id == trace.trace_id
    assert len(retrieved.node_traces) == 1


@pytest.mark.asyncio
async def test_trace_checkpoint_validation(backend):
    """Test that traces must reference valid checkpoints."""
    run_id = generate_run_id()
    snapshot_id = generate_snapshot_id()

    # Try to save trace without checkpoint
    trace = ExecutionTrace(
        trace_id=str(uuid.uuid4()),
        run_id=run_id,
        wave_number=0,
        checkpoint_snapshot_id=snapshot_id,
        node_traces=[],
        parallel_batch_id="batch1",
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )

    with pytest.raises(ValueError, match="Invalid checkpoint reference"):
        await backend.save_trace(trace)


@pytest.mark.asyncio
async def test_state_reconstruction_forward(backend):
    """Test forward state reconstruction with deltas."""
    run_id = generate_run_id()
    reconstructor = StateReconstructor(backend)

    # Create base snapshot at wave 0 (full state)
    base_state = {
        "node1": SampleState(value=10, name="base"),
        "node2": SampleState(value=20, name="base2"),
    }
    base_snapshot = StateSnapshot(
        snapshot_id=generate_snapshot_id(),
        run_id=run_id,
        wave_number=0,
        full_state=base_state,
        state_hash="hash0",
        next_frontier=[],
        routing_ended=False,
    )
    await backend.save_state_snapshot(base_snapshot)

    # Create delta snapshots (waves 1-9)
    prev_state = dict(base_state)
    for i in range(1, 10):
        current_state = {
            "node1": SampleState(value=10 + i, name=f"delta_{i}"),
            "node2": SampleState(value=20, name="base2"),
        }

        forward_delta = DeltaComputer.compute_forward_delta(prev_state, current_state)

        snapshot = StateSnapshot(
            snapshot_id=generate_snapshot_id(),
            run_id=run_id,
            wave_number=i,
            forward_delta=forward_delta,
            state_hash=f"hash{i}",
            next_frontier=[],
            routing_ended=False,
        )
        await backend.save_state_snapshot(snapshot)
        prev_state = current_state

    # Reconstruct state at wave 5
    reconstructed = await reconstructor.reconstruct_state_at(run_id, 5)

    assert "node1" in reconstructed
    assert reconstructed["node1"].value == 15  # 10 + 5  # type: ignore[attr-defined]
    assert reconstructed["node1"].name == "delta_5"  # type: ignore[attr-defined]
    assert reconstructed["node2"].value == 20  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_validate_and_save_trace_with_update(backend):
    """Test bidirectional reference validation and update."""
    run_id = generate_run_id()
    snapshot_id = generate_snapshot_id()

    # Create checkpoint
    snapshot = StateSnapshot(
        snapshot_id=snapshot_id,
        run_id=run_id,
        wave_number=0,
        full_state={"node1": SampleState(value=1, name="test")},
        state_hash="hash123",
        next_frontier=[],
        routing_ended=False,
    )
    await backend.save_state_snapshot(snapshot)

    # Create and validate trace
    trace_id = str(uuid.uuid4())
    trace = ExecutionTrace(
        trace_id=trace_id,
        run_id=run_id,
        wave_number=0,
        checkpoint_snapshot_id=snapshot_id,
        node_traces=[],
        parallel_batch_id="batch1",
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )

    # Validate and save with checkpoint update
    await validate_and_save_trace(backend, trace, update_checkpoint=True)

    # Verify bidirectional link
    updated_snapshot = await backend.get_state_snapshot(run_id, 0)
    assert updated_snapshot.trace_id == trace_id


@pytest.mark.asyncio
async def test_validate_and_save_trace_invalid_checkpoint(backend):
    """Test validation fails for non-existent checkpoint."""
    run_id = generate_run_id()

    trace = ExecutionTrace(
        trace_id=str(uuid.uuid4()),
        run_id=run_id,
        wave_number=0,
        checkpoint_snapshot_id=generate_snapshot_id(),
        node_traces=[],
        parallel_batch_id="batch1",
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )

    with pytest.raises(CheckpointIntegrityError, match="non-existent checkpoint"):
        await validate_and_save_trace(backend, trace)


@pytest.mark.asyncio
async def test_checkpoint_integrity_validation(backend):
    """Test checkpoint integrity validation."""
    run_id = generate_run_id()
    snapshot_id = generate_snapshot_id()

    # Create checkpoint
    snapshot = StateSnapshot(
        snapshot_id=snapshot_id,
        run_id=run_id,
        wave_number=0,
        full_state={"node1": SampleState(value=1, name="test")},
        state_hash="hash123",
        next_frontier=[],
        routing_ended=False,
    )
    await backend.save_state_snapshot(snapshot)

    # Should be valid (no trace yet)
    assert await validate_checkpoint_integrity(backend, run_id, 0)

    # Add trace
    trace = ExecutionTrace(
        trace_id=str(uuid.uuid4()),
        run_id=run_id,
        wave_number=0,
        checkpoint_snapshot_id=snapshot_id,
        node_traces=[],
        parallel_batch_id="batch1",
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )
    await validate_and_save_trace(backend, trace, update_checkpoint=True)

    # Should still be valid (bidirectional link established)
    assert await validate_checkpoint_integrity(backend, run_id, 0)


@pytest.mark.asyncio
async def test_run_metadata_crud(backend):
    """Test run metadata save and retrieve."""
    run_id = generate_run_id()

    metadata = RunMetadata(
        run_id=run_id,
        flow_id="test_flow",
        started_at=datetime.now(UTC),
        status=RunMetadata.Status.RUNNING,
        total_waves=5,
    )

    await backend.save_run_metadata(metadata)

    retrieved = await backend.get_run_metadata(run_id)
    assert retrieved is not None
    assert retrieved.run_id == run_id
    assert retrieved.flow_id == "test_flow"
    assert retrieved.status == RunMetadata.Status.RUNNING


@pytest.mark.asyncio
async def test_list_runs(backend):
    """Test listing runs with filters."""
    now = datetime.now(UTC)

    # Create 3 runs
    for i in range(3):
        run_id = generate_run_id()
        metadata = RunMetadata(
            run_id=run_id,
            flow_id=f"flow_{i}",
            started_at=now,
            status=RunMetadata.Status.RUNNING,
            total_waves=i,
        )
        await backend.save_run_metadata(metadata)

    # List all
    runs = await backend.list_runs()
    assert len(runs) == 3

    # List with limit
    limited = await backend.list_runs(limit=2)
    assert len(limited) == 2


@pytest.mark.asyncio
async def test_delete_run(backend):
    """Test deleting run data."""
    run_id = generate_run_id()

    # Create metadata, checkpoint, and trace
    metadata = RunMetadata(
        run_id=run_id,
        flow_id="test_flow",
        started_at=datetime.now(UTC),
        status=RunMetadata.Status.RUNNING,
        total_waves=1,
    )
    await backend.save_run_metadata(metadata)

    snapshot_id = generate_snapshot_id()
    snapshot = StateSnapshot(
        snapshot_id=snapshot_id,
        run_id=run_id,
        wave_number=0,
        full_state={"node1": SampleState(value=1, name="test")},
        state_hash="hash123",
        next_frontier=[],
        routing_ended=False,
    )
    await backend.save_state_snapshot(snapshot)

    # Delete run (keep checkpoints)
    await backend.delete_run(run_id, keep_checkpoints=True)

    # Metadata should be gone
    assert await backend.get_run_metadata(run_id) is None

    # Checkpoint should remain
    assert await backend.get_state_snapshot(run_id, 0) is not None

    # Delete run completely
    await backend.delete_run(run_id, keep_checkpoints=False)
    assert await backend.get_state_snapshot(run_id, 0) is None
