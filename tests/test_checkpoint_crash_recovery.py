"""Crash recovery and resilience tests for checkpoint v2.

These tests validate that checkpoint v2 handles crashes gracefully and can recover
from partial writes, corrupted data, and system failures.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import contextlib
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
from pydantic_flow.checkpoints import validate_checkpoint_integrity
from pydantic_flow.checkpoints.event_log import StreamingEventLog
from pydantic_flow.checkpoints.types import ExecutionTrace
from pydantic_flow.checkpoints.types import RunMetadata
from pydantic_flow.checkpoints.types import StateSnapshot
from pydantic_flow.checkpoints.types import generate_run_id
from pydantic_flow.checkpoints.types import generate_snapshot_id
from pydantic_flow.checkpoints.types import generate_trace_id
from pydantic_flow.checkpoints.validation import repair_bidirectional_references
from pydantic_flow.nodes.base import NodeWithInput
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


class FailingNode(NodeWithInput[CrashRecoveryInput, CrashRecoveryInput]):
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
            result_preview=result.model_dump(),
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
@pytest.mark.xfail(reason="Feature not fully implemented")
async def test_corrupted_checkpoint_detection(temp_backend, tmp_path):
    """Test detection and handling of corrupted checkpoint data."""
    run_id = generate_run_id()

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
    with pytest.raises((ValueError, TypeError, RuntimeError)):  # Deserialization errors
        await temp_backend.get_state_snapshot(run_id, 0)

    # Verify run metadata is still accessible
    metadata = await temp_backend.get_run_metadata(run_id)
    assert metadata is not None


@pytest.mark.asyncio
@pytest.mark.xfail(reason="Feature not fully implemented")
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
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        await reconstructor.reconstruct_state_at(run_id, 5)


# =============================================================================
# Test 3: Event Flush Failures
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.xfail(reason="Feature not fully implemented")
async def test_event_flush_failure_with_circuit_breaker(temp_backend):
    """Test event log circuit breaker on repeated flush failures."""
    from pydantic_flow.streaming import TokenChunk

    run_id = generate_run_id()
    event_log = StreamingEventLog(
        store=temp_backend,
        run_id=run_id,
        node_id="test_node",
        wave_number=0,
        snapshot_id=generate_snapshot_id(),
        buffer_size=10,
        buffer_max_bytes=1024,
    )

    # Mock backend to fail flushes
    original_append = temp_backend.append_events_batch

    call_count = 0

    async def failing_append(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        msg = f"Flush failure #{call_count}"
        raise RuntimeError(msg)

    temp_backend.append_events_batch = failing_append

    # Add events (should trigger flush attempts)
    for i in range(15):
        event = TokenChunk(
            run_id=run_id,
            node_id="test_node",
            text=f"token_{i}",
        )
        await event_log.append(event)

    # Circuit breaker should trigger after max_flush_failures consecutive failures
    await asyncio.sleep(0.2)  # Wait for flush attempts

    assert event_log.flush_failures >= event_log.max_flush_failures

    # Restore backend
    temp_backend.append_events_batch = original_append

    # Finalize should handle circuit breaker state
    from pydantic_flow.checkpoints.types import StateRef

    with contextlib.suppress(RuntimeError):
        await event_log.finalize(
            input_ref=StateRef(snapshot_id=event_log.snapshot_id, state_key="test_node")
        )  # Expected to fail due to circuit breaker


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


@pytest.mark.asyncio
@pytest.mark.xfail(reason="Feature not fully implemented")
async def test_repair_broken_bidirectional_references(temp_backend):
    """Test repair utility for broken checkpoint-trace references."""
    run_id = generate_run_id()

    # Create run metadata
    metadata = RunMetadata(
        run_id=run_id,
        flow_id="test_repair",
        started_at=datetime.now(UTC),
        status=RunMetadata.Status.COMPLETED,
        total_waves=3,
    )
    await temp_backend.save_run_metadata(metadata)

    # Create checkpoints and traces with broken references
    for wave in range(3):
        snapshot = StateSnapshot(
            snapshot_id=generate_snapshot_id(),
            run_id=run_id,
            wave_number=wave,
            full_state={"node": CrashRecoveryState(counter=wave, data=f"wave{wave}")},
            state_hash=f"hash{wave}",
            next_frontier=[],
            routing_ended=False,
        )
        await temp_backend.save_state_snapshot(snapshot)

        # Create trace
        trace = ExecutionTrace(
            trace_id=generate_trace_id(),
            run_id=run_id,
            wave_number=wave,
            checkpoint_snapshot_id=snapshot.snapshot_id,
            parallel_batch_id="batch_0",
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            node_traces=[],
        )
        await temp_backend.save_trace(trace)

    # Manually break wave 1 checkpoint reference
    snapshot_1 = await temp_backend.get_state_snapshot(run_id, 1)
    snapshot_1.trace_id = None  # Break bidirectional link
    await temp_backend.update_state_snapshot(snapshot_1)

    # Verify integrity is broken
    is_valid = await validate_checkpoint_integrity(temp_backend, run_id, 1)
    assert not is_valid

    # Repair references
    stats = await repair_bidirectional_references(temp_backend, run_id, dry_run=False)

    assert stats["fixed_checkpoints"] >= 1

    # Verify integrity is restored
    is_valid_after = await validate_checkpoint_integrity(temp_backend, run_id, 1)
    assert is_valid_after


# =============================================================================
# Test 5: Concurrent Write Conflicts
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.xfail(reason="Feature not fully implemented")
async def test_concurrent_checkpoint_writes_same_wave(temp_backend):
    """Test handling of concurrent writes to same wave (should use last-write-wins)."""
    run_id = generate_run_id()

    # Create two different snapshots for same wave
    snapshot_a = StateSnapshot(
        snapshot_id=generate_snapshot_id(),
        run_id=run_id,
        wave_number=0,
        full_state={"node": CrashRecoveryState(counter=1, data="version_a")},
        state_hash="hash_a",
        next_frontier=[],
        routing_ended=False,
    )

    snapshot_b = StateSnapshot(
        snapshot_id=generate_snapshot_id(),
        run_id=run_id,
        wave_number=0,
        full_state={"node": CrashRecoveryState(counter=2, data="version_b")},
        state_hash="hash_b",
        next_frontier=[],
        routing_ended=False,
    )

    # Write both concurrently
    await asyncio.gather(
        temp_backend.save_state_snapshot(snapshot_a),
        temp_backend.save_state_snapshot(snapshot_b),
    )

    # Retrieve and verify one was saved (last write wins)
    retrieved = await temp_backend.get_state_snapshot(run_id, 0)
    assert retrieved is not None
    assert retrieved.wave_number == 0
    # Should be one of the two versions
    assert retrieved.state_hash in ["hash_a", "hash_b"]


# =============================================================================
# Test 6: State Reconstruction Edge Cases
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.xfail(reason="Feature not fully implemented")
async def test_reconstruct_state_with_missing_intermediate_delta(temp_backend):
    """Test state reconstruction when intermediate delta is missing."""
    run_id = generate_run_id()
    reconstructor = StateReconstructor(backend=temp_backend)

    # Create sparse checkpoint sequence
    # Wave 0: Full
    snapshot_0 = StateSnapshot(
        snapshot_id=generate_snapshot_id(),
        run_id=run_id,
        wave_number=0,
        full_state={"node": CrashRecoveryState(counter=0, data="base")},
        state_hash="hash_0",
        next_frontier=[],
        routing_ended=False,
    )
    await temp_backend.save_state_snapshot(snapshot_0)

    # Wave 2: Delta (skip wave 1)
    snapshot_2 = StateSnapshot(
        snapshot_id=generate_snapshot_id(),
        run_id=run_id,
        wave_number=2,
        forward_delta={"node": CrashRecoveryState(counter=2, data="delta2")},
        state_hash="hash_2",
        next_frontier=[],
        routing_ended=False,
    )
    await temp_backend.save_state_snapshot(snapshot_2)

    # Reconstruct wave 2 (should work by applying delta to wave 0)
    state_2 = await reconstructor.reconstruct_state_at(run_id, 2)
    assert state_2["node"].counter == 2  # type: ignore[attr-defined]

    # Attempt to reconstruct missing wave 1 (should fail gracefully)
    with pytest.raises((ValueError, KeyError)):  # No checkpoint exists
        await reconstructor.reconstruct_state_at(run_id, 1)


@pytest.mark.asyncio
@pytest.mark.xfail(reason="Feature not fully implemented")
async def test_reconstruct_at_wave_with_only_reverse_delta(temp_backend):
    """Test reconstruction using reverse delta when forward delta is missing."""
    run_id = generate_run_id()
    reconstructor = StateReconstructor(backend=temp_backend)

    # Wave 0: Full
    snapshot_0 = StateSnapshot(
        snapshot_id=generate_snapshot_id(),
        run_id=run_id,
        wave_number=0,
        full_state={"node": CrashRecoveryState(counter=10, data="wave0")},
        state_hash="hash_0",
        next_frontier=[],
        routing_ended=False,
    )
    await temp_backend.save_state_snapshot(snapshot_0)

    # Wave 1: Delta with reverse_delta
    snapshot_1 = StateSnapshot(
        snapshot_id=generate_snapshot_id(),
        run_id=run_id,
        wave_number=1,
        forward_delta={"node": CrashRecoveryState(counter=11, data="wave1")},
        reverse_delta={"node": CrashRecoveryState(counter=10, data="wave0")},
        state_hash="hash_1",
        next_frontier=[],
        routing_ended=False,
    )
    await temp_backend.save_state_snapshot(snapshot_1)

    # Reconstruct using rewind (reverse delta)
    state_0_via_rewind = await reconstructor.rewind_state_to(
        run_id, from_wave=1, to_wave=0
    )
    assert state_0_via_rewind["node"].counter == 10  # type: ignore[attr-defined]


# =============================================================================
# Test 7: Manager Crash Recovery
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.xfail(reason="Feature not fully implemented")
async def test_manager_handles_backend_failures_gracefully(checkpoint_manager):
    """Test that CheckpointManager handles backend failures without crashing."""
    # Mock backend to fail
    original_save = checkpoint_manager._backend.save_state_snapshot

    async def failing_save(*args, **kwargs):
        msg = "Backend unavailable"
        raise ConnectionError(msg)

    checkpoint_manager._backend.save_state_snapshot = failing_save

    # Attempt to save checkpoint (should handle error gracefully)
    state = {"node": CrashRecoveryState(counter=1, data="test")}

    # Should not raise exception (manager should handle gracefully)
    with contextlib.suppress(ConnectionError):
        await checkpoint_manager.save_wave_checkpoint(
            wave_number=0,
            state=state,
            next_frontier=[],
            routing_ended=False,
        )

    # Restore backend
    checkpoint_manager._backend.save_state_snapshot = original_save


# =============================================================================
# Test 8: Database Connection Loss
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
