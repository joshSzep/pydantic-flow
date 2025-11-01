"""Tests for VacuumManager checkpoint lifecycle management."""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
import tempfile
import uuid

from pydantic import BaseModel
import pytest

from pydantic_flow.checkpoints.backends.sqlite import SQLiteCheckpointBackend
from pydantic_flow.checkpoints.backends.sqlite import SQLiteCheckpointConfig
from pydantic_flow.checkpoints.types import EventSummary
from pydantic_flow.checkpoints.types import ExecutionTrace
from pydantic_flow.checkpoints.types import NodeExecutionTrace
from pydantic_flow.checkpoints.types import RunMetadata
from pydantic_flow.checkpoints.types import StateRef
from pydantic_flow.checkpoints.types import StateSnapshot
from pydantic_flow.checkpoints.types import generate_run_id
from pydantic_flow.checkpoints.types import generate_snapshot_id
from pydantic_flow.checkpoints.vacuum import VacuumManager
from pydantic_flow.checkpoints.vacuum import VacuumPolicy


class SampleState(BaseModel):
    """Sample state for testing."""

    value: int


@pytest.fixture
async def vacuum_backend():
    """Create temporary SQLite backend for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        config = SQLiteCheckpointConfig(db_path=db_path)
        backend = SQLiteCheckpointBackend(config=config)

        await backend.initialize()
        yield backend
        await backend.close()


async def create_test_run(
    backend: SQLiteCheckpointBackend,
    days_ago: int,
    status: str = "completed",
) -> str:
    """Create test run with snapshots and traces, return run_id."""
    run_id = generate_run_id()
    started_at = datetime.now(UTC) - timedelta(days=days_ago)
    completed_at = started_at + timedelta(hours=1)

    # Create metadata
    metadata = RunMetadata(
        run_id=run_id,
        flow_id="test_flow",
        started_at=started_at,
        completed_at=completed_at,
        status=RunMetadata.Status(status),
        total_waves=3,
    )
    await backend.save_run_metadata(metadata)

    # Create snapshots for 3 waves
    snapshot_ids = []
    for wave in range(3):
        snapshot_id = generate_snapshot_id()
        snapshot_ids.append(snapshot_id)
        snapshot = StateSnapshot(
            snapshot_id=snapshot_id,
            run_id=run_id,
            wave_number=wave,
            full_state={"node1": SampleState(value=wave)},
            state_hash=f"hash_{wave}",
            next_frontier=[],
            routing_ended=False,
        )
        await backend.save_state_snapshot(snapshot)

    # Create trace for wave 0 using the first snapshot
    trace = ExecutionTrace(
        trace_id=str(uuid.uuid4()),
        run_id=run_id,
        wave_number=0,
        checkpoint_snapshot_id=snapshot_ids[0],
        node_traces=[
            NodeExecutionTrace(
                log_id=str(uuid.uuid4()),
                node_id="node1",
                wave_number=0,
                snapshot_id=snapshot_ids[0],
                input_ref=StateRef(snapshot_id=snapshot_ids[0], state_key="node1"),
                output_ref=StateRef(snapshot_id=snapshot_ids[0], state_key="node1"),
                event_log_id="log123",
                total_events=1,
                event_summary=EventSummary(
                    total_events=1,
                    token_count=10,
                    tool_call_count=0,
                    cache_hits=0,
                    tool_calls=[],
                ),
                started_at=started_at,
                completed_at=completed_at,
                next_nodes=[],
            )
        ],
        parallel_batch_id="batch1",
        started_at=started_at,
        completed_at=completed_at,
    )
    await backend.save_trace(trace)
    return run_id


@pytest.mark.asyncio
async def test_vacuum_traces_before(vacuum_backend):
    """Test time-based trace cleanup."""
    manager = VacuumManager(backend=vacuum_backend)

    # Create old and new runs
    await create_test_run(vacuum_backend, days_ago=60)
    await create_test_run(vacuum_backend, days_ago=10)

    # Vacuum traces older than 30 days
    cutoff = datetime.now(UTC) - timedelta(days=30)
    report = await manager.vacuum_traces_before(cutoff, dry_run=False)

    # Should delete old run's trace only
    assert report.traces_deleted >= 1
    assert not report.dry_run


@pytest.mark.asyncio
async def test_vacuum_traces_dry_run(vacuum_backend):
    """Test dry-run mode for trace cleanup."""
    manager = VacuumManager(backend=vacuum_backend)

    await create_test_run(vacuum_backend, days_ago=60)

    # Dry run should report but not delete
    cutoff = datetime.now(UTC) - timedelta(days=30)
    report = await manager.vacuum_traces_before(cutoff, dry_run=True)

    assert report.traces_deleted >= 1
    assert report.dry_run


@pytest.mark.asyncio
async def test_vacuum_run(vacuum_backend):
    """Test deletion of specific run."""
    manager = VacuumManager(backend=vacuum_backend)

    run_id = await create_test_run(vacuum_backend, days_ago=10)

    # Verify run exists
    runs = await vacuum_backend.list_runs()
    assert len(runs) == 1

    # Delete the run
    report = await manager.vacuum_run(run_id, keep_checkpoints=False, dry_run=False)

    assert report.runs_deleted == 1
    assert report.traces_deleted == 3
    assert not report.dry_run

    # Verify run deleted
    runs = await vacuum_backend.list_runs()
    assert len(runs) == 0


@pytest.mark.asyncio
async def test_vacuum_by_policy_completed_runs(vacuum_backend):
    """Test policy-based vacuum for completed runs."""
    manager = VacuumManager(backend=vacuum_backend)

    # Create old and new completed runs
    await create_test_run(vacuum_backend, days_ago=60, status="completed")
    new_run = await create_test_run(vacuum_backend, days_ago=10, status="completed")

    # Apply policy: keep completed runs for 30 days
    policy = VacuumPolicy(
        completed_run_retention_days=30,
        keep_checkpoints=False,
        dry_run=False,
    )
    report = await manager.vacuum_by_policy(policy)

    # Should delete old run only
    assert report.runs_deleted == 1

    # Verify only new run remains
    runs = await vacuum_backend.list_runs()
    assert len(runs) == 1
    assert runs[0].run_id == new_run


@pytest.mark.asyncio
async def test_vacuum_by_policy_trace_retention(vacuum_backend):
    """Test policy-based vacuum for trace retention."""
    manager = VacuumManager(backend=vacuum_backend)

    # Create run with old completion time
    await create_test_run(vacuum_backend, days_ago=60)

    # Apply policy: keep traces for 30 days, don't delete runs
    policy = VacuumPolicy(
        trace_retention_days=30,
        completed_run_retention_days=None,  # Don't delete runs
        keep_checkpoints=False,
        dry_run=False,
    )
    report = await manager.vacuum_by_policy(policy)

    # Should delete traces but not runs
    assert report.traces_deleted >= 1
    assert report.runs_deleted == 0

    # Verify run still exists
    runs = await vacuum_backend.list_runs()
    assert len(runs) == 1
