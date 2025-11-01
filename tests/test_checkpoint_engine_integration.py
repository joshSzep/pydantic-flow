"""Integration tests for checkpoint v2 with engine execution.

This module demonstrates how CheckpointManager integrates with flow execution,
capturing state snapshots after each wave of execution.
"""

from pydantic import BaseModel
import pytest

from pydantic_flow.checkpoints import CheckpointConfig
from pydantic_flow.checkpoints import CheckpointManager
from pydantic_flow.checkpoints import SQLiteCheckpointBackend
from pydantic_flow.checkpoints import SQLiteCheckpointConfig
from pydantic_flow.checkpoints.reconstructor import StateReconstructor
from pydantic_flow.checkpoints.types import RunMetadata


class NodeState(BaseModel):
    """State for test nodes."""

    value: int


@pytest.fixture
async def temp_backend(tmp_path):
    """Create temporary checkpoint backend."""
    db_path = tmp_path / "test_checkpoint_integration.db"
    config = SQLiteCheckpointConfig(db_path=db_path)
    backend = SQLiteCheckpointBackend(config)
    await backend.initialize()
    try:
        yield backend
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_simulated_engine_execution(temp_backend):
    """Test checkpoint manager with simulated wave-based execution.

    This test simulates how StepperEngine would use CheckpointManager:
    1. Initialize run
    2. Execute wave 0, save checkpoint
    3. Execute wave 1, save checkpoint with delta
    4. Finalize run
    """
    config = CheckpointConfig(
        trace_sample_rate=1.0,
        save_full_snapshot_every=10,
    )
    manager = CheckpointManager(
        config=config,
        storage=temp_backend,
        flow_id="test_flow",
    )

    await manager.initialize_run()
    run_id = manager.run_id

    # Simulate wave 0 execution
    wave_0_state = {
        "node_a": NodeState(value=1),
    }
    snapshot_0 = await manager.save_wave_checkpoint(
        current_state=wave_0_state,
        next_frontier=["node_b"],
        routing_ended=False,
    )

    assert snapshot_0.run_id == run_id
    assert snapshot_0.wave_number == 0
    assert snapshot_0.full_state is not None
    assert "node_a" in snapshot_0.full_state

    # Simulate wave 1 execution (adds node_b, updates node_a)
    wave_1_state = {
        "node_a": NodeState(value=2),
        "node_b": NodeState(value=3),
    }
    snapshot_1 = await manager.save_wave_checkpoint(
        current_state=wave_1_state,
        next_frontier=["node_c"],
        routing_ended=False,
    )

    assert snapshot_1.wave_number == 1
    assert snapshot_1.forward_delta is not None
    assert snapshot_1.full_state is None  # Delta snapshot

    # Finalize run
    await manager.finalize_run(status=RunMetadata.Status.COMPLETED)

    # Verify metadata
    metadata = await temp_backend.get_run_metadata(run_id)
    assert metadata is not None
    assert metadata.status == RunMetadata.Status.COMPLETED
    assert metadata.total_waves == 2


@pytest.mark.asyncio
async def test_checkpoint_with_trace_sampling(temp_backend):
    """Test that trace sampling configuration is respected."""
    config_no_traces = CheckpointConfig(
        trace_sample_rate=0.0,
        save_full_snapshot_every=5,
    )
    manager = CheckpointManager(
        config=config_no_traces,
        storage=temp_backend,
        flow_id="test_flow_no_traces",
    )

    await manager.initialize_run()

    # Create event log with sampling
    event_log = manager.create_event_log(node_id="node_a")
    assert event_log is None  # Should not create log when sample rate is 0%

    # Now test with 100% sampling
    config_with_traces = CheckpointConfig(
        trace_sample_rate=1.0,
        save_full_snapshot_every=5,
    )
    manager_with_traces = CheckpointManager(
        config=config_with_traces,
        storage=temp_backend,
        flow_id="test_flow_with_traces",
    )
    await manager_with_traces.initialize_run()

    event_log_2 = manager_with_traces.create_event_log(node_id="node_a")
    assert event_log_2 is not None  # Should create log when sample rate is 100%


@pytest.mark.asyncio
async def test_full_snapshot_every_nth_wave(temp_backend):
    """Test that full snapshots are created every Nth wave."""
    config = CheckpointConfig(
        trace_sample_rate=0.0,
        save_full_snapshot_every=3,  # Full snapshot every 3rd wave
    )
    manager = CheckpointManager(
        config=config,
        storage=temp_backend,
        flow_id="test_full_snapshots",
    )

    await manager.initialize_run()

    # Wave 0: Full (always full)
    state_0 = {"node": NodeState(value=0)}
    snap_0 = await manager.save_wave_checkpoint(
        current_state=state_0, next_frontier=[], routing_ended=False
    )
    assert snap_0.full_state is not None

    # Wave 1: Delta
    state_1 = {"node": NodeState(value=1)}
    snap_1 = await manager.save_wave_checkpoint(
        current_state=state_1, next_frontier=[], routing_ended=False
    )
    assert snap_1.full_state is None
    assert snap_1.forward_delta is not None

    # Wave 2: Delta
    state_2 = {"node": NodeState(value=2)}
    snap_2 = await manager.save_wave_checkpoint(
        current_state=state_2, next_frontier=[], routing_ended=False
    )
    assert snap_2.full_state is None

    # Wave 3: Full (3rd wave = full snapshot)
    state_3 = {"node": NodeState(value=3)}
    snap_3 = await manager.save_wave_checkpoint(
        current_state=state_3, next_frontier=[], routing_ended=False
    )
    assert snap_3.full_state is not None
    assert snap_3.forward_delta is None


@pytest.mark.asyncio
async def test_reconstruct_state_from_deltas(temp_backend):
    """Test that state can be reconstructed from checkpoints."""
    config = CheckpointConfig(
        trace_sample_rate=0.0,
        save_full_snapshot_every=10,
    )
    manager = CheckpointManager(
        config=config,
        storage=temp_backend,
        flow_id="test_reconstruction",
    )

    await manager.initialize_run()
    run_id = manager.run_id

    # Save 3 waves
    await manager.save_wave_checkpoint(
        current_state={"node": NodeState(value=1)},
        next_frontier=[],
        routing_ended=False,
    )
    await manager.save_wave_checkpoint(
        current_state={"node": NodeState(value=2)},
        next_frontier=[],
        routing_ended=False,
    )
    await manager.save_wave_checkpoint(
        current_state={"node": NodeState(value=3)},
        next_frontier=[],
        routing_ended=False,
    )

    # Reconstruct state at wave 2
    reconstructor = StateReconstructor(backend=temp_backend)
    reconstructed = await reconstructor.reconstruct_state_at(run_id, wave_number=2)

    assert "node" in reconstructed
    node_state = reconstructed["node"]
    assert isinstance(node_state, NodeState)
    assert node_state.value == 3  # Wave 2 has value=3
