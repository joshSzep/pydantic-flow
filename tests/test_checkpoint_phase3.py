"""Tests for checkpoint v2 Phase 3: Event streaming and engine integration."""

from pathlib import Path
import tempfile

from pydantic import BaseModel
import pytest

from pydantic_flow.checkpoints import CheckpointConfig
from pydantic_flow.checkpoints import CheckpointManager
from pydantic_flow.checkpoints import SQLiteCheckpointBackend
from pydantic_flow.checkpoints import SQLiteCheckpointConfig
from pydantic_flow.checkpoints import StreamingEventLog
from pydantic_flow.checkpoints.types import RunMetadata
from pydantic_flow.checkpoints.types import StateRef
from pydantic_flow.checkpoints.types import generate_run_id
from pydantic_flow.checkpoints.types import generate_snapshot_id
from pydantic_flow.streaming import TokenChunk


class SimpleState(BaseModel):
    """Simple state model for checkpoint tests."""

    value: int


@pytest.fixture
async def temp_storage():
    """Create temporary SQLite storage backend."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = SQLiteCheckpointConfig(db_path=Path(tmpdir) / "test.db")
        backend = SQLiteCheckpointBackend(config)
        await backend.initialize()
        yield backend
        await backend.close()


@pytest.mark.asyncio
async def test_streaming_event_log_basic(temp_storage):
    """Test basic event log functionality."""
    run_id = generate_run_id()
    snapshot_id = generate_snapshot_id()

    event_log = StreamingEventLog(
        store=temp_storage,
        run_id=run_id,
        node_id="test_node",
        wave_number=0,
        snapshot_id=snapshot_id,
        buffer_size=5,  # Small buffer for testing
    )

    # Append some events
    for i in range(3):
        event = TokenChunk(text=f"token_{i}")
        await event_log.append(event)

    assert event_log.total_events == 3
    assert event_log.token_count > 0

    # Finalize and create trace
    input_ref = StateRef(snapshot_id=snapshot_id, state_key="input")
    output_ref = StateRef(snapshot_id=snapshot_id, state_key="output")

    trace = await event_log.finalize(
        input_ref=input_ref,
        output_ref=output_ref,
        next_nodes=["node_b"],
    )

    assert trace.node_id == "test_node"
    assert trace.total_events == 3
    assert trace.next_nodes == ["node_b"]


@pytest.mark.asyncio
async def test_streaming_event_log_buffer_flush(temp_storage):
    """Test event log buffer flushing."""
    run_id = generate_run_id()
    snapshot_id = generate_snapshot_id()

    event_log = StreamingEventLog(
        store=temp_storage,
        run_id=run_id,
        node_id="test_node",
        wave_number=0,
        snapshot_id=snapshot_id,
        buffer_size=2,  # Very small buffer
    )

    # Append events that will trigger flush
    for i in range(5):
        event = TokenChunk(text=f"token_{i}")
        await event_log.append(event)

    assert event_log.total_events == 5
    # Buffer should have been flushed


@pytest.mark.asyncio
async def test_checkpoint_manager_initialization(temp_storage):
    """Test checkpoint manager initialization."""
    config = CheckpointConfig(
        enabled=True,
        storage_backend=temp_storage,
        trace_sample_rate=1.0,  # Always capture traces
    )

    manager = CheckpointManager(
        config=config,
        storage=temp_storage,
        flow_id="test_flow",
    )

    await manager.initialize_run()

    # Verify run metadata was created
    metadata = await temp_storage.get_run_metadata(manager.run_id)
    assert metadata is not None
    assert metadata.flow_id == "test_flow"
    assert metadata.status == RunMetadata.Status.RUNNING


@pytest.mark.asyncio
async def test_checkpoint_manager_save_wave(temp_storage):
    """Test saving wave checkpoints with delta compression."""
    config = CheckpointConfig(
        enabled=True,
        storage_backend=temp_storage,
        save_full_snapshot_every=3,
    )

    manager = CheckpointManager(
        config=config,
        storage=temp_storage,
        flow_id="test_flow",
    )

    await manager.initialize_run()

    # Save wave 0 (full snapshot)
    state_0 = {"node_a": SimpleState(value=1)}
    snapshot_0 = await manager.save_wave_checkpoint(
        current_state=state_0,
        next_frontier=["node_b"],
        routing_ended=False,
    )

    assert snapshot_0.wave_number == 0
    assert snapshot_0.full_state is not None  # First wave is always full
    assert snapshot_0.forward_delta is None

    # Save wave 1 (delta)
    state_1 = {"node_a": SimpleState(value=2), "node_b": SimpleState(value=3)}
    snapshot_1 = await manager.save_wave_checkpoint(
        current_state=state_1,
        next_frontier=["node_c"],
        routing_ended=False,
    )

    assert snapshot_1.wave_number == 1
    assert snapshot_1.full_state is None  # Should be delta
    assert snapshot_1.forward_delta is not None
    assert snapshot_1.reverse_delta is not None

    # Verify snapshots can be retrieved
    retrieved_0 = await temp_storage.get_state_snapshot(manager.run_id, 0)
    assert retrieved_0 is not None
    assert retrieved_0.wave_number == 0

    retrieved_1 = await temp_storage.get_state_snapshot(manager.run_id, 1)
    assert retrieved_1 is not None
    assert retrieved_1.wave_number == 1


@pytest.mark.asyncio
async def test_checkpoint_manager_trace_sampling(temp_storage):
    """Test trace sampling configuration."""
    # Config with 0% sampling
    config_no_trace = CheckpointConfig(
        enabled=True,
        storage_backend=temp_storage,
        trace_sample_rate=0.0,
    )

    manager_no_trace = CheckpointManager(
        config=config_no_trace,
        storage=temp_storage,
        flow_id="test_flow",
    )

    event_log = manager_no_trace.create_event_log("test_node")
    assert event_log is None  # No trace should be created

    # Config with 100% sampling
    config_full_trace = CheckpointConfig(
        enabled=True,
        storage_backend=temp_storage,
        trace_sample_rate=1.0,
    )

    manager_full_trace = CheckpointManager(
        config=config_full_trace,
        storage=temp_storage,
        flow_id="test_flow",
    )

    event_log = manager_full_trace.create_event_log("test_node")
    assert event_log is not None  # Trace should be created


@pytest.mark.asyncio
async def test_checkpoint_manager_finalize(temp_storage):
    """Test finalizing run metadata."""
    config = CheckpointConfig(
        enabled=True,
        storage_backend=temp_storage,
    )

    manager = CheckpointManager(
        config=config,
        storage=temp_storage,
        flow_id="test_flow",
    )

    await manager.initialize_run()

    # Finalize successfully
    await manager.finalize_run(status=RunMetadata.Status.COMPLETED)

    # Verify metadata
    metadata = await temp_storage.get_run_metadata(manager.run_id)
    assert metadata is not None
    assert metadata.status == RunMetadata.Status.COMPLETED
    assert metadata.completed_at is not None


@pytest.mark.asyncio
async def test_event_log_circuit_breaker(temp_storage):
    """Test circuit breaker triggers after repeated flush failures."""
    run_id = generate_run_id()
    snapshot_id = generate_snapshot_id()

    # Create event log with failing storage
    event_log = StreamingEventLog(
        store=temp_storage,
        run_id=run_id,
        node_id="test_node",
        wave_number=0,
        snapshot_id=snapshot_id,
        buffer_size=1,  # Trigger flush on every event
    )

    # Manually set flush_failures to max
    event_log.flush_failures = event_log.max_flush_failures

    # Next append should raise due to circuit breaker
    with pytest.raises(RuntimeError, match="circuit breaker"):
        await event_log.append(TokenChunk(text="test"))
