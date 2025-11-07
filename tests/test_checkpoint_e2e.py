"""End-to-end tests for checkpoint v2 with real flow execution.

Tests the full integration of checkpoint v2 with StepperEngine.
"""

from collections.abc import AsyncIterator

from pydantic import BaseModel
import pytest

from pydantic_flow.checkpoints import CheckpointConfig
from pydantic_flow.checkpoints import SQLiteCheckpointBackend
from pydantic_flow.checkpoints import SQLiteCheckpointConfig
from pydantic_flow.checkpoints.reconstructor import StateReconstructor
from pydantic_flow.core.run_config import RunConfig
from pydantic_flow.flow.flow import Flow
from pydantic_flow.nodes.base import BaseNode
from pydantic_flow.streaming import ProgressItem
from pydantic_flow.streaming import StreamEnd
from pydantic_flow.streaming import StreamStart
from pydantic_flow.streaming.tool_events import ToolResult
from tests.conftest import extract_result_from_stream


class SimpleInput(BaseModel):
    """Test input."""

    value: int


# SimpleOutput has same structure for node chaining
SimpleState = SimpleInput


class SimpleOutput(BaseModel):
    """Output with multiple node results."""

    a: SimpleState
    b: SimpleState
    c: SimpleState


class IncrementNode(BaseNode[SimpleInput, SimpleInput]):
    """Node that increments value."""

    async def astream(self, input_data: SimpleInput) -> AsyncIterator[ProgressItem]:
        """Increment the value."""
        yield StreamStart(run_id=self.run_id or "", node_id=self.name)
        result = SimpleInput(value=input_data.value + 1)
        yield ToolResult(result=result)
        yield StreamEnd(
            run_id=self.run_id or "",
            node_id=self.name,
            result=result.model_dump(),
        )


@pytest.fixture
async def temp_checkpoint_backend(tmp_path):
    """Create temporary checkpoint v2 backend."""
    db_path = tmp_path / "test_e2e_checkpoint.db"
    config = SQLiteCheckpointConfig(db_path=db_path)
    backend = SQLiteCheckpointBackend(config)
    await backend.initialize()
    try:
        yield backend
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_flow_execution_with_checkpoints(temp_checkpoint_backend):
    """Test full flow execution with checkpoint v2 enabled."""
    # Create flow with 3 chained nodes
    flow = Flow(input_type=SimpleInput, output_type=SimpleOutput)

    node_a = IncrementNode(name="a")
    node_b = IncrementNode(name="b", inputs=(node_a.output,))
    node_c = IncrementNode(name="c", inputs=(node_b.output,))

    flow.add_nodes(node_a, node_b, node_c)

    # Compile - should auto-select DAG engine (now has checkpoint v2)
    # Flows execute directly - no compilation needed

    # Configure checkpoint v2
    checkpoint_config = CheckpointConfig(
        trace_sample_rate=0.0,  # No traces for now
        save_full_snapshot_every=10,
    )

    run_config = RunConfig(
        checkpoint_backend=temp_checkpoint_backend,
        checkpoint_config=checkpoint_config,
        run_id="test_e2e_run",
    )

    # Execute flow
    result = await extract_result_from_stream(
        flow.astream(SimpleInput(value=1), run_config)
    )

    # Verify result - each node increments by 1
    assert result.a.value == 2  # 1 + 1
    assert result.b.value == 3  # 2 + 1
    assert result.c.value == 4  # 3 + 1

    # Verify checkpoints were saved
    from pydantic_flow.checkpoints.types import RunId

    run_id = RunId("test_e2e_run")

    # Get wave 0 snapshot (after node a)
    snapshot_0 = await temp_checkpoint_backend.get_state_snapshot(run_id, wave_number=0)
    assert snapshot_0 is not None
    assert snapshot_0.wave_number == 0
    assert snapshot_0.full_state is not None
    assert "a" in snapshot_0.full_state

    # Get wave 1 snapshot (after node b, should be delta)
    snapshot_1 = await temp_checkpoint_backend.get_state_snapshot(run_id, wave_number=1)
    assert snapshot_1 is not None
    assert snapshot_1.wave_number == 1
    assert snapshot_1.forward_delta is not None

    # Get wave 2 snapshot (after node c, should be delta)
    snapshot_2 = await temp_checkpoint_backend.get_state_snapshot(run_id, wave_number=2)
    assert snapshot_2 is not None
    assert snapshot_2.wave_number == 2

    # Verify run metadata
    metadata = await temp_checkpoint_backend.get_run_metadata(run_id)
    assert metadata is not None
    assert metadata.status.value == "completed"
    assert metadata.total_waves == 3  # 3 waves executed


@pytest.mark.asyncio
async def test_state_reconstruction_from_real_execution(temp_checkpoint_backend):
    """Test state reconstruction from actual flow execution."""
    # Create and execute flow
    flow = Flow(input_type=SimpleInput, output_type=SimpleOutput)

    node_a = IncrementNode(name="a")
    node_b = IncrementNode(name="b", inputs=(node_a.output,))
    node_c = IncrementNode(name="c", inputs=(node_b.output,))

    flow.add_nodes(node_a, node_b, node_c)

    # Flows execute directly - no compilation needed

    checkpoint_config = CheckpointConfig(
        trace_sample_rate=0.0,
        save_full_snapshot_every=10,
    )

    run_config = RunConfig(
        checkpoint_backend=temp_checkpoint_backend,
        checkpoint_config=checkpoint_config,
        run_id="test_reconstruction_run",
    )

    await extract_result_from_stream(flow.astream(SimpleInput(value=10), run_config))

    # Reconstruct state at wave 1 (after nodes a and b)
    from pydantic_flow.checkpoints.types import RunId

    run_id = RunId("test_reconstruction_run")
    reconstructor = StateReconstructor(backend=temp_checkpoint_backend)

    state_at_wave_1 = await reconstructor.reconstruct_state_at(run_id, wave_number=1)
    assert "a" in state_at_wave_1
    assert "b" in state_at_wave_1
    a_state = state_at_wave_1["a"]
    b_state = state_at_wave_1["b"]
    assert isinstance(a_state, SimpleState)
    assert isinstance(b_state, SimpleState)
    assert a_state.value == 11  # 10 + 1
    assert b_state.value == 12  # 11 + 1

    # Reconstruct at wave 2 (after node c)
    state_at_wave_2 = await reconstructor.reconstruct_state_at(run_id, wave_number=2)
    assert "c" in state_at_wave_2
    c_state = state_at_wave_2["c"]
    assert isinstance(c_state, SimpleState)
    assert c_state.value == 13  # 12 + 1


@pytest.mark.asyncio
async def test_full_snapshot_every_nth_wave_with_real_flow(temp_checkpoint_backend):
    """Test full snapshot configuration with real execution."""
    flow = Flow(input_type=SimpleInput, output_type=SimpleOutput)

    node_a = IncrementNode(name="a")
    node_b = IncrementNode(name="b", inputs=(node_a.output,))
    node_c = IncrementNode(name="c", inputs=(node_b.output,))

    flow.add_nodes(node_a, node_b, node_c)

    # Flows execute directly - no compilation needed

    # Configure full snapshot every 2nd wave
    checkpoint_config = CheckpointConfig(
        trace_sample_rate=0.0,
        save_full_snapshot_every=2,
    )

    run_config = RunConfig(
        checkpoint_backend=temp_checkpoint_backend,
        checkpoint_config=checkpoint_config,
        run_id="test_full_snapshot_run",
    )

    await extract_result_from_stream(flow.astream(SimpleInput(value=1), run_config))

    # Verify snapshot types
    from pydantic_flow.checkpoints.types import RunId

    run_id = RunId("test_full_snapshot_run")

    # Wave 0: Always full
    snap_0 = await temp_checkpoint_backend.get_state_snapshot(run_id, wave_number=0)
    assert snap_0.full_state is not None

    # Wave 1: Delta
    snap_1 = await temp_checkpoint_backend.get_state_snapshot(run_id, wave_number=1)
    assert snap_1.forward_delta is not None
    assert snap_1.full_state is None

    # Wave 2: Full (every 2nd wave)
    snap_2 = await temp_checkpoint_backend.get_state_snapshot(run_id, wave_number=2)
    assert snap_2.full_state is not None
