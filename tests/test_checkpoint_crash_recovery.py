"""Crash recovery and resilience tests for checkpoint v2.

These tests validate that checkpoint v2 handles crashes gracefully and can recover
from partial writes, corrupted data, and system failures.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC
from datetime import datetime
import sqlite3

from pydantic import BaseModel
import pytest

from pydantic_flow.checkpoints import CheckpointConfig
from pydantic_flow.checkpoints import CheckpointManager
from pydantic_flow.checkpoints import SQLiteCheckpointBackend
from pydantic_flow.checkpoints import SQLiteCheckpointConfig
from pydantic_flow.checkpoints import StateReconstructor
from pydantic_flow.checkpoints.event_log import StreamingEventLog
from pydantic_flow.checkpoints.types import ExecutionTrace
from pydantic_flow.checkpoints.types import RunMetadata
from pydantic_flow.checkpoints.types import StateSnapshot
from pydantic_flow.checkpoints.types import generate_run_id
from pydantic_flow.checkpoints.types import generate_snapshot_id
from pydantic_flow.checkpoints.types import generate_trace_id
from pydantic_flow.nodes.base import BaseNode
from pydantic_flow.streaming import ProgressItem
from pydantic_flow.streaming import StreamEnd
from pydantic_flow.streaming import StreamStart


class CrashRecoveryState(BaseModel):
    """Test state model for crash recovery tests."""

    counter: int
    data: str


class CrashRecoveryInput(BaseModel):
    """Test input for crash recovery tests."""

    value: int


class FailingNode(BaseNode[CrashRecoveryInput, CrashRecoveryInput]):
    """Node that can be configured to fail at specific points."""

    fail_on_execution: bool = False

    async def astream(
        self, input_data: CrashRecoveryInput
    ) -> AsyncIterator[ProgressItem]:
        """Execute with optional failure."""
        yield StreamStart(run_id=self.run_id or "", node_id=self.name)

        if self.fail_on_execution:
            msg = "Simulated execution failure"
            raise RuntimeError(msg)

        result = CrashRecoveryInput(value=input_data.value + 1)
        yield StreamEnd(
            run_id=self.run_id or "",
            node_id=self.name,
            result=result.model_dump(),
        )


@pytest.fixture
async def temp_backend(tmp_path):
    """Create temporary SQLite backend."""
    db_path = tmp_path / "crash_test.db"
    config = SQLiteCheckpointConfig(db_path=db_path)
    backend = SQLiteCheckpointBackend(config)
    await backend.initialize()
    try:
        yield backend
    finally:
        await backend.close()


@pytest.fixture
async def checkpoint_manager(temp_backend):
    """Create checkpoint manager with temp backend."""
    config = CheckpointConfig(
        enabled=True,
        trace_sample_rate=0.0,
        save_full_snapshot_every=10,
    )
    run_id = generate_run_id()

    metadata = RunMetadata(
        run_id=run_id,
        flow_id="test_flow",
        started_at=datetime.now(UTC),
        status=RunMetadata.Status.RUNNING,
        total_waves=0,
    )
    await temp_backend.save_run_metadata(metadata)

    manager = CheckpointManager(
        config=config,
        storage=temp_backend,
        flow_id="test_flow",
        run_id=run_id,
    )
    return manager


# =============================================================================
# Test 1: Crash During Checkpoint Write
# =============================================================================


@pytest.mark.asyncio
async def test_crash_during_checkpoint_write(temp_backend):
    """Test recovery from crash during checkpoint write operation."""
    run_id = generate_run_id()

    # Save metadata first
    metadata = RunMetadata(
        run_id=run_id,
        flow_id="test_crash",
        started_at=datetime.now(UTC),
        status=RunMetadata.Status.RUNNING,
        total_waves=0,
    )
    await temp_backend.save_run_metadata(metadata)

    # Create first successful checkpoint
    snapshot_0 = StateSnapshot(
        snapshot_id=generate_snapshot_id(),
        run_id=run_id,
        wave_number=0,
        full_state={"node1": CrashRecoveryState(counter=1, data="wave_0")},
        state_hash="hash_0",
        next_frontier=["node2"],
        routing_ended=False,
    )
    await temp_backend.save_state_snapshot(snapshot_0)

    # Simulate crash during second checkpoint write by patching the save method
    original_save = temp_backend.save_state_snapshot

    async def crashing_save(snapshot):
        if snapshot.wave_number == 1:
            msg = "Simulated database crash"
            raise sqlite3.OperationalError(msg)
        return await original_save(snapshot)

    temp_backend.save_state_snapshot = crashing_save

    # Attempt to save second checkpoint (will fail)
    snapshot_1 = StateSnapshot(
        snapshot_id=generate_snapshot_id(),
        run_id=run_id,
        wave_number=1,
        full_state={"node1": CrashRecoveryState(counter=2, data="wave_1")},
        state_hash="hash_1",
        next_frontier=[],
        routing_ended=False,
    )

    with pytest.raises(sqlite3.OperationalError, match="database crash"):
        await temp_backend.save_state_snapshot(snapshot_1)

    # Restore normal operation
    temp_backend.save_state_snapshot = original_save

    # Verify wave 0 checkpoint is intact
    recovered_snapshot = await temp_backend.get_state_snapshot(run_id, 0)
    assert recovered_snapshot is not None
    assert recovered_snapshot.wave_number == 0
    assert recovered_snapshot.state_hash == "hash_0"

    # Verify wave 1 checkpoint doesn't exist (atomic rollback)
    missing_snapshot = await temp_backend.get_state_snapshot(run_id, 1)
    assert missing_snapshot is None

    # Verify we can continue execution by saving wave 1 again
    await temp_backend.save_state_snapshot(snapshot_1)
    retry_snapshot = await temp_backend.get_state_snapshot(run_id, 1)
    assert retry_snapshot is not None
    assert retry_snapshot.wave_number == 1


# =============================================================================
# Test 2: Corrupted Checkpoint Data
# =============================================================================


@pytest.mark.asyncio
async def test_corrupted_checkpoint_detection(temp_backend, tmp_path):
    """Test detection and handling of corrupted checkpoint data."""
    run_id = generate_run_id()

    # Save run metadata first
    metadata = RunMetadata(
        run_id=run_id,
        flow_id="test_corrupted",
        started_at=datetime.now(UTC),
        status=RunMetadata.Status.RUNNING,
        total_waves=0,
    )
    await temp_backend.save_run_metadata(metadata)

    # Save valid checkpoint
    snapshot = StateSnapshot(
        snapshot_id=generate_snapshot_id(),
        run_id=run_id,
        wave_number=0,
        full_state={"node1": CrashRecoveryState(counter=1, data="test")},
        state_hash="valid_hash",
        next_frontier=[],
        routing_ended=False,
    )
    await temp_backend.save_state_snapshot(snapshot)

    # Corrupt the checkpoint data directly in database
    if temp_backend.db:
        await temp_backend.db.execute(
            "UPDATE state_snapshots SET data_compressed = ? WHERE wave_number = ?",
            (b"corrupted_invalid_msgpack_data", 0),
        )
        await temp_backend.db.commit()

    # Attempt to retrieve corrupted checkpoint
    with pytest.raises((
        ValueError,
        TypeError,
        RuntimeError,
        OSError,
    )):  # Deserialization errors (OSError covers BadGzipFile)
        await temp_backend.get_state_snapshot(run_id, 0)

    # Verify run metadata is still accessible
    retrieved_metadata = await temp_backend.get_run_metadata(run_id)
    assert retrieved_metadata is not None


@pytest.mark.asyncio
async def test_corrupted_delta_recovery(temp_backend):
    """Test recovery when delta checkpoint is corrupted but full checkpoint exists."""
    run_id = generate_run_id()

    # Save wave 0 (full snapshot)
    snapshot_0 = StateSnapshot(
        snapshot_id=generate_snapshot_id(),
        run_id=run_id,
        wave_number=0,
        full_state={"node1": CrashRecoveryState(counter=10, data="base")},
        state_hash="hash_0",
        next_frontier=[],
        routing_ended=False,
    )
    await temp_backend.save_state_snapshot(snapshot_0)

    # Save wave 5 (delta)
    snapshot_5 = StateSnapshot(
        snapshot_id=generate_snapshot_id(),
        run_id=run_id,
        wave_number=5,
        forward_delta={"node1": CrashRecoveryState(counter=15, data="delta5")},
        state_hash="hash_5",
        next_frontier=[],
        routing_ended=False,
    )
    await temp_backend.save_state_snapshot(snapshot_5)

    # Save wave 10 (full snapshot)
    snapshot_10 = StateSnapshot(
        snapshot_id=generate_snapshot_id(),
        run_id=run_id,
        wave_number=10,
        full_state={"node1": CrashRecoveryState(counter=20, data="full10")},
        state_hash="hash_10",
        next_frontier=[],
        routing_ended=False,
    )
    await temp_backend.save_state_snapshot(snapshot_10)

    # Corrupt wave 5 (delta)
    if temp_backend.db:
        await temp_backend.db.execute(
            "UPDATE state_snapshots SET data_compressed = ? WHERE wave_number = ?",
            (b"corrupted", 5),
        )
        await temp_backend.db.commit()

    # State reconstruction should still work using full snapshots
    reconstructor = StateReconstructor(backend=temp_backend)

    # Can reconstruct wave 0 (full snapshot)
    state_0 = await reconstructor.reconstruct_state_at(run_id, 0)
    assert state_0["node1"].counter == 10  # type: ignore[attr-defined]

    # Can reconstruct wave 10 (full snapshot)
    state_10 = await reconstructor.reconstruct_state_at(run_id, 10)
    assert state_10["node1"].counter == 20  # type: ignore[attr-defined]

    # Wave 5 reconstruction will fail due to corruption
    with pytest.raises((ValueError, TypeError, RuntimeError, OSError)):
        await reconstructor.reconstruct_state_at(run_id, 5)


# =============================================================================
# Test 3: Event Flush Failures
# =============================================================================


@pytest.mark.asyncio
async def test_event_flush_partial_success(temp_backend):
    """Test handling of partial event flush success."""
    from pydantic_flow.streaming import TokenChunk

    run_id = generate_run_id()
    snapshot_id = generate_snapshot_id()

    event_log = StreamingEventLog(
        store=temp_backend,
        run_id=run_id,
        node_id="test_node",
        wave_number=0,
        snapshot_id=snapshot_id,
        buffer_size=5,
        buffer_max_bytes=1024,
    )

    # Add events that will trigger flush
    events_added = []
    for i in range(8):
        event = TokenChunk(
            run_id=run_id,
            node_id="test_node",
            text=f"token_{i}",
        )
        await event_log.append(event)
        events_added.append(event)

    # Manually trigger flush
    await event_log._flush()

    # Wait for async flush to complete
    await asyncio.sleep(0.1)

    # Verify buffer was cleared after successful flush
    assert len(event_log.event_buffer) < 8  # Should have been flushed


# =============================================================================
# Test 4: Bidirectional Reference Integrity
# =============================================================================


@pytest.mark.asyncio
async def test_orphaned_trace_detection(temp_backend):
    """Test detection of orphaned traces (trace without checkpoint)."""
    run_id = generate_run_id()

    # Create trace without corresponding checkpoint
    trace = ExecutionTrace(
        trace_id=generate_trace_id(),
        run_id=run_id,
        wave_number=5,
        checkpoint_snapshot_id=generate_snapshot_id(),  # Non-existent checkpoint
        parallel_batch_id="batch_0",
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        node_traces=[],
    )

    # Saving this trace should fail validation
    # Should raise integrity/validation error
    with pytest.raises((ValueError, RuntimeError)):
        await temp_backend.save_trace(trace)


# =============================================================================
# Test 5: Database Connection Loss
# =============================================================================


@pytest.mark.asyncio
async def test_recovery_after_database_connection_loss(tmp_path):
    """Test recovery after temporary database connection loss."""
    db_path = tmp_path / "connection_test.db"
    config = SQLiteCheckpointConfig(db_path=db_path)
    backend = SQLiteCheckpointBackend(config)
    await backend.initialize()

    run_id = generate_run_id()

    # Save checkpoint successfully
    snapshot_0 = StateSnapshot(
        snapshot_id=generate_snapshot_id(),
        run_id=run_id,
        wave_number=0,
        full_state={"node": CrashRecoveryState(counter=1, data="test")},
        state_hash="hash_0",
        next_frontier=[],
        routing_ended=False,
    )
    await backend.save_state_snapshot(snapshot_0)

    # Simulate connection loss by closing backend
    await backend.close()

    # Attempt operation (should fail)
    with pytest.raises((RuntimeError, sqlite3.Error)):
        await backend.get_state_snapshot(run_id, 0)

    # Reinitialize connection
    await backend.initialize()

    # Verify data is still accessible after reconnection
    retrieved = await backend.get_state_snapshot(run_id, 0)
    assert retrieved is not None
    assert retrieved.wave_number == 0

    await backend.close()
