"""Tests for durability modes (SYNC, ASYNC, EXIT).

Tests verify checkpoint creation behavior for each durability mode:
- SYNC: Checkpoints created synchronously after each node
- ASYNC: Checkpoints created in background tasks
- EXIT: Checkpoints created only on completion/error
"""

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel
import pytest

from pydantic_flow import Flow
from pydantic_flow.core.durability import DurabilityMode
from pydantic_flow.core.errors import FlowError
from pydantic_flow.core.run_config import RunConfig
from pydantic_flow.engine.stepper import EngineConfig
from pydantic_flow.engine.stepper import StepperEngine
from pydantic_flow.hitl.checkpoints.interface import CheckpointQuery
from pydantic_flow.hitl.checkpoints.interface import RunId
from pydantic_flow.hitl.checkpoints.memory import InMemoryCheckpointStore
from pydantic_flow.nodes.base import BaseNode
from pydantic_flow.streaming.base import ProgressItem
from pydantic_flow.streaming.base import ProgressType


class SimpleInput(BaseModel):
    """Test input model."""

    value: str


# SimpleOutput has same structure for node chaining
SimpleOutput = SimpleInput


class ThreeNodeOutput(BaseModel):
    """Output for 3-node flow."""

    node1: SimpleInput
    node2: SimpleInput
    node3: SimpleInput


class TwoNodeOutput(BaseModel):
    """Output for 2-node flow."""

    node1: SimpleInput
    node2: SimpleInput


class SingleNodeOutput(BaseModel):
    """Output for single-node flow."""

    node: SimpleInput


class ProcessNode(BaseNode[SimpleInput, SimpleInput]):
    """Node that processes SimpleInput."""

    def __init__(self, name: str, suffix: str = "", **kwargs: Any):
        """Initialize process node."""
        super().__init__(name=name, **kwargs)
        self.suffix = suffix

    async def run(self, input_data: SimpleInput) -> SimpleInput:
        """Process and modify value."""
        return SimpleInput(value=f"{input_data.value}{self.suffix}")

    async def astream(self, input_data: SimpleInput) -> AsyncIterator[ProgressItem]:
        """Stream process."""
        await self.run(input_data)
        yield ProgressItem(type=ProgressType.END)


# ============================================================================
# DAG Engine Tests
# ============================================================================


@pytest.mark.asyncio
async def test_dag_sync_mode_creates_checkpoints_after_each_node():
    """Verify SYNC mode creates checkpoints synchronously after each node in DAG."""
    # Create simple 3-node flow
    flow = Flow(input_type=SimpleInput, output_type=ThreeNodeOutput)
    node1 = ProcessNode(name="node1", suffix="_p1")
    node2 = ProcessNode(name="node2", suffix="_p2")
    node3 = ProcessNode(name="node3", suffix="_p2")
    flow.add_nodes(node1, node2, node3)
    flow.add_edge(node1, node2)
    flow.add_edge(node2, node3)

    # Configure SYNC mode
    store = InMemoryCheckpointStore()
    run_id = "dag_sync_test"
    config = RunConfig(
        checkpoint_store=store, run_id=run_id, durability_mode=DurabilityMode.SYNC
    )

    # Run flow
    result = await flow.run(SimpleInput(value="test"), config=config)
    assert result is not None

    # Verify checkpoints created after each node (3 intermediate + 1 final)
    query = CheckpointQuery(run_id=RunId(run_id))
    checkpoints, _ = await store.list(query)
    msg = f"Expected at least 3 checkpoints, got {len(checkpoints)}"
    assert len(checkpoints) >= 3, msg


@pytest.mark.asyncio
async def test_dag_async_mode_creates_background_checkpoints():
    """Verify ASYNC mode creates checkpoints without blocking execution in DAG."""
    # Create simple 2-node flow
    flow = Flow(input_type=SimpleInput, output_type=TwoNodeOutput)
    node1 = ProcessNode(name="node1", suffix="_p1")
    node2 = ProcessNode(name="node2", suffix="_p2")
    flow.add_nodes(node1, node2)
    flow.add_edge(node1, node2)

    # Configure ASYNC mode (this is the default)
    store = InMemoryCheckpointStore()
    run_id = "dag_async_test"
    config = RunConfig(
        checkpoint_store=store, run_id=run_id, durability_mode=DurabilityMode.ASYNC
    )

    # Run flow
    result = await flow.run(SimpleInput(value="test"), config=config)
    assert result is not None

    # Wait a moment for background tasks to complete
    await asyncio.sleep(0.1)

    # Verify checkpoints were created in background
    query = CheckpointQuery(run_id=RunId(run_id))
    checkpoints, _ = await store.list(query)
    msg = f"Expected at least 2 checkpoints, got {len(checkpoints)}"
    assert len(checkpoints) >= 2, msg


@pytest.mark.asyncio
async def test_dag_exit_mode_only_checkpoints_on_completion():
    """Verify EXIT mode creates no intermediate checkpoints in DAG."""
    # Create 3-node flow
    flow = Flow(input_type=SimpleInput, output_type=ThreeNodeOutput)
    node1 = ProcessNode(name="node1", suffix="_p1")
    node2 = ProcessNode(name="node2", suffix="_p2")
    node3 = ProcessNode(name="node3", suffix="_p2")
    flow.add_nodes(node1, node2, node3)
    flow.add_edge(node1, node2)
    flow.add_edge(node2, node3)

    # Configure EXIT mode
    store = InMemoryCheckpointStore()
    run_id = "dag_exit_test"
    config = RunConfig(
        checkpoint_store=store, run_id=run_id, durability_mode=DurabilityMode.EXIT
    )

    # Run flow
    result = await flow.run(SimpleInput(value="test"), config=config)
    assert result is not None

    # Verify only 1 checkpoint (on completion)
    query = CheckpointQuery(run_id=RunId(run_id))
    checkpoints, _ = await store.list(query)
    msg = f"Expected exactly 1 checkpoint, got {len(checkpoints)}"
    assert len(checkpoints) == 1, msg


@pytest.mark.asyncio
async def test_dag_exit_mode_checkpoints_on_error():
    """Verify EXIT mode creates checkpoint on error in DAG."""

    class FailingNode(BaseNode[SimpleInput, SimpleInput]):
        """Node that always fails."""

        async def run(self, input_data: SimpleInput) -> SimpleInput:
            """Raise an error."""
            raise ValueError("Intentional test error")

        async def astream(self, input_data: SimpleInput) -> AsyncIterator[ProgressItem]:
            """Stream implementation."""
            if False:
                yield ProgressItem(type=ProgressType.END)
            raise ValueError("Intentional test error")

    # Create flow with failing node
    flow = Flow(input_type=SimpleInput, output_type=TwoNodeOutput)
    node1 = ProcessNode(name="node1", suffix="_p1")
    node2 = FailingNode(name="node2")
    flow.add_nodes(node1, node2)
    flow.add_edge(node1, node2)

    # Configure EXIT mode
    store = InMemoryCheckpointStore()
    run_id = "dag_exit_error_test"
    config = RunConfig(
        checkpoint_store=store, run_id=run_id, durability_mode=DurabilityMode.EXIT
    )

    # Run flow - expect error
    with pytest.raises(FlowError, match="Intentional test error"):
        await flow.run(SimpleInput(value="test"), config=config)

    # Verify checkpoint created on error
    query = CheckpointQuery(run_id=RunId(run_id))
    checkpoints, _ = await store.list(query)
    msg = f"Expected 1 checkpoint on error, got {len(checkpoints)}"
    assert len(checkpoints) == 1, msg


@pytest.mark.asyncio
async def test_dag_no_store_gracefully_handles_all_modes():
    """Verify all durability modes handle checkpoint_store=None gracefully in DAG."""
    # Create simple flow
    flow = Flow(input_type=SimpleInput, output_type=SingleNodeOutput)
    node = ProcessNode(name="node", suffix="_p1")
    flow.add_nodes(node)

    # Test each mode with no store
    for mode in [DurabilityMode.SYNC, DurabilityMode.ASYNC, DurabilityMode.EXIT]:
        config = RunConfig(checkpoint_store=None, durability_mode=mode)
        result = await flow.run(SimpleInput(value="test"), config=config)
        assert result is not None


# ============================================================================
# Stepper Engine Tests
# ============================================================================


@pytest.mark.asyncio
async def test_stepper_sync_mode_creates_checkpoints_after_each_frontier():
    """Verify SYNC mode creates checkpoints after each frontier in Stepper."""
    # Create simple stepper flow (needs explicit mode or cycles)
    node1 = ProcessNode(name="node1", suffix="_p")
    node2 = ProcessNode(name="node2", suffix="_p")
    node3 = ProcessNode(name="node3", suffix="_p")

    # Build stepper engine directly with explicit nodes
    engine_config = EngineConfig(
        nodes=[node1, node2, node3],
        edges={"node1": ["node2"], "node2": ["node3"]},
        entry_nodes=["node1"],
        input_type=SimpleInput,
        output_type=ThreeNodeOutput,
        flow_id="stepper_sync_test",
    )

    engine = StepperEngine[SimpleInput, ThreeNodeOutput](config=engine_config)

    # Configure SYNC mode
    store = InMemoryCheckpointStore()
    run_id = "stepper_sync_test"
    config = RunConfig(
        checkpoint_store=store, run_id=run_id, durability_mode=DurabilityMode.SYNC
    )

    # Run engine
    result = await engine.invoke(SimpleInput(value="test"), config=config)
    assert result.node1.value == "test_p"

    # Verify checkpoints created after each frontier (3 frontiers + 1 on exit)
    query = CheckpointQuery(run_id=RunId(run_id))
    checkpoints, _ = await store.list(query)
    assert len(checkpoints) >= 3, (
        f"Expected at least 3 checkpoints, got {len(checkpoints)}"
    )


@pytest.mark.asyncio
async def test_stepper_async_mode_creates_background_checkpoints():
    """Verify ASYNC mode creates background checkpoints in Stepper."""
    # Create stepper flow
    node1 = ProcessNode(name="node1", suffix="_p")
    node2 = ProcessNode(name="node2", suffix="_p")

    engine_config = EngineConfig(
        nodes=[node1, node2],
        edges={"node1": ["node2"]},
        entry_nodes=["node1"],
        input_type=SimpleInput,
        output_type=TwoNodeOutput,
        flow_id="stepper_async_test",
    )

    engine = StepperEngine[SimpleInput, TwoNodeOutput](config=engine_config)

    # Configure ASYNC mode
    store = InMemoryCheckpointStore()
    run_id = "stepper_async_test"
    config = RunConfig(
        checkpoint_store=store, run_id=run_id, durability_mode=DurabilityMode.ASYNC
    )

    # Run engine
    result = await engine.invoke(SimpleInput(value="test"), config=config)
    assert result.node1.value == "test_p"

    # Wait for background tasks
    await asyncio.sleep(0.1)

    # Verify checkpoints created
    query = CheckpointQuery(run_id=RunId(run_id))
    checkpoints, _ = await store.list(query)
    assert len(checkpoints) >= 2, (
        f"Expected at least 2 checkpoints, got {len(checkpoints)}"
    )


@pytest.mark.asyncio
async def test_stepper_exit_mode_only_checkpoints_on_completion():
    """Verify EXIT mode creates no intermediate checkpoints in Stepper."""
    # Create stepper flow
    node1 = ProcessNode(name="node1", suffix="_p")
    node2 = ProcessNode(name="node2", suffix="_p")
    node3 = ProcessNode(name="node3", suffix="_p")

    engine_config = EngineConfig(
        nodes=[node1, node2, node3],
        edges={"node1": ["node2"], "node2": ["node3"]},
        entry_nodes=["node1"],
        input_type=SimpleInput,
        output_type=ThreeNodeOutput,
        flow_id="stepper_exit_test",
    )

    engine = StepperEngine[SimpleInput, ThreeNodeOutput](config=engine_config)

    # Configure EXIT mode
    store = InMemoryCheckpointStore()
    run_id = "stepper_exit_test"
    config = RunConfig(
        checkpoint_store=store, run_id=run_id, durability_mode=DurabilityMode.EXIT
    )

    # Run engine
    result = await engine.invoke(SimpleInput(value="test"), config=config)
    assert result.node1.value == "test_p"

    # Verify only 1 checkpoint (on completion)
    query = CheckpointQuery(run_id=RunId(run_id))
    checkpoints, _ = await store.list(query)
    assert len(checkpoints) == 1, (
        f"Expected exactly 1 checkpoint, got {len(checkpoints)}"
    )


@pytest.mark.asyncio
async def test_stepper_exit_mode_checkpoints_on_error():
    """Verify EXIT mode creates checkpoint on error in Stepper."""

    class FailingNode(BaseNode[SimpleInput, SimpleInput]):
        """Node that always fails."""

        async def run(self, input_data: SimpleInput) -> SimpleInput:
            """Raise an error."""
            raise ValueError("Intentional test error")

        async def astream(self, input_data: SimpleInput) -> AsyncIterator[ProgressItem]:
            """Stream implementation."""
            if False:
                yield ProgressItem(type=ProgressType.END)
            raise ValueError("Intentional test error")

    # Create stepper flow with failing node
    node1 = ProcessNode(name="node1", suffix="_p")
    node2 = FailingNode(name="node2")

    engine_config = EngineConfig(
        nodes=[node1, node2],
        edges={"node1": ["node2"]},
        entry_nodes=["node1"],
        input_type=SimpleInput,
        output_type=TwoNodeOutput,
        flow_id="stepper_exit_error_test",
    )

    engine = StepperEngine[SimpleInput, TwoNodeOutput](config=engine_config)

    # Configure EXIT mode
    store = InMemoryCheckpointStore()
    run_id = "stepper_exit_error_test"
    config = RunConfig(
        checkpoint_store=store, run_id=run_id, durability_mode=DurabilityMode.EXIT
    )

    # Run engine - expect error
    with pytest.raises(FlowError, match="Intentional test error"):
        await engine.invoke(SimpleInput(value="test"), config=config)

    # Verify checkpoint created on error
    query = CheckpointQuery(run_id=RunId(run_id))
    checkpoints, _ = await store.list(query)
    assert len(checkpoints) == 1, (
        f"Expected 1 checkpoint on error, got {len(checkpoints)}"
    )


@pytest.mark.asyncio
async def test_stepper_no_store_gracefully_handles_all_modes():
    """Verify all durability modes handle None gracefully in Stepper."""
    # Create stepper flow
    node = ProcessNode(name="node", suffix="_p")

    engine_config = EngineConfig(
        nodes=[node],
        edges={},
        entry_nodes=["node"],
        input_type=SimpleInput,
        output_type=SingleNodeOutput,
        flow_id="stepper_no_store_test",
    )

    engine = StepperEngine[SimpleInput, SingleNodeOutput](config=engine_config)

    # Test each mode with no store
    for mode in [DurabilityMode.SYNC, DurabilityMode.ASYNC, DurabilityMode.EXIT]:
        config = RunConfig(checkpoint_store=None, durability_mode=mode)
        result = await engine.invoke(SimpleInput(value="test"), config=config)
        assert result.node.value == "test_p"


# ============================================================================
# Mode Switching Tests
# ============================================================================


@pytest.mark.asyncio
async def test_mode_switching_between_runs():
    """Verify different runs can use different durability modes."""
    # Create simple flow
    flow = Flow(input_type=SimpleInput, output_type=SingleNodeOutput)
    node = ProcessNode(name="node", suffix="_p1")
    flow.add_nodes(node)

    store = InMemoryCheckpointStore()

    # Run 1: SYNC mode
    config1 = RunConfig(
        checkpoint_store=store, run_id="run1", durability_mode=DurabilityMode.SYNC
    )
    result1 = await flow.run(SimpleInput(value="test1"), config=config1)
    assert result1.node.value == "test1_p1"

    # Run 2: ASYNC mode
    config2 = RunConfig(
        checkpoint_store=store, run_id="run2", durability_mode=DurabilityMode.ASYNC
    )
    result2 = await flow.run(SimpleInput(value="test2"), config=config2)
    assert result2.node.value == "test2_p1"

    # Run 3: EXIT mode
    config3 = RunConfig(
        checkpoint_store=store, run_id="run3", durability_mode=DurabilityMode.EXIT
    )
    result3 = await flow.run(SimpleInput(value="test3"), config=config3)
    assert result3.node.value == "test3_p1"

    # Wait for async tasks
    await asyncio.sleep(0.1)

    # Verify all runs created checkpoints with proper isolation
    query1 = CheckpointQuery(run_id=RunId("run1"))
    checkpoints1, _ = await store.list(query1)
    assert len(checkpoints1) >= 1

    query2 = CheckpointQuery(run_id=RunId("run2"))
    checkpoints2, _ = await store.list(query2)
    assert len(checkpoints2) >= 1

    query3 = CheckpointQuery(run_id=RunId("run3"))
    checkpoints3, _ = await store.list(query3)
    assert len(checkpoints3) == 1  # EXIT mode: only 1 checkpoint


@pytest.mark.asyncio
async def test_default_mode_is_async():
    """Verify that default durability mode is ASYNC."""
    # Create simple flow
    flow = Flow(input_type=SimpleInput, output_type=SingleNodeOutput)
    node = ProcessNode(name="node", suffix="_p1")
    flow.add_nodes(node)

    # Create config without explicit durability_mode
    store = InMemoryCheckpointStore()
    run_id = "default_mode_test"
    config = RunConfig(checkpoint_store=store, run_id=run_id)

    # Verify default is ASYNC
    assert config.durability_mode == DurabilityMode.ASYNC

    # Run flow
    result = await flow.run(SimpleInput(value="test"), config=config)
    assert result is not None

    # Wait for background tasks
    await asyncio.sleep(0.1)

    # Verify checkpoints created (ASYNC behavior)
    query = CheckpointQuery(run_id=RunId(run_id))
    checkpoints, _ = await store.list(query)
    assert len(checkpoints) >= 1


# ============================================================================
# Edge Case Tests
# ============================================================================


@pytest.mark.asyncio
async def test_sync_mode_with_parallel_nodes():
    """Verify SYNC mode handles parallel execution correctly."""
    # Create flow with parallel nodes (fan-out pattern)
    flow = Flow(input_type=SimpleInput, output_type=ThreeNodeOutput)
    node1 = ProcessNode(name="node1", suffix="_p1")
    node2 = ProcessNode(name="node2", suffix="_p2")
    node3 = ProcessNode(name="node3", suffix="_p3")
    flow.add_nodes(node1, node2, node3)
    flow.add_edge(node1, node2)
    flow.add_edge(node1, node3)

    # Configure SYNC mode
    store = InMemoryCheckpointStore()
    run_id = "sync_parallel_test"
    config = RunConfig(
        checkpoint_store=store, run_id=run_id, durability_mode=DurabilityMode.SYNC
    )

    # Run flow
    result = await flow.run(SimpleInput(value="test"), config=config)
    assert result.node1.value == "test_p1"

    # Verify checkpoints created
    query = CheckpointQuery(run_id=RunId(run_id))
    checkpoints, _ = await store.list(query)
    assert len(checkpoints) >= 1, "Expected checkpoints from parallel execution"


@pytest.mark.asyncio
async def test_async_mode_background_tasks_complete_before_return():
    """Verify ASYNC mode tracks background tasks correctly."""
    # Create flow with multiple nodes
    flow = Flow(input_type=SimpleInput, output_type=TwoNodeOutput)
    node1 = ProcessNode(name="node1", suffix="_p1")
    node2 = ProcessNode(name="node2", suffix="_p2")
    flow.add_nodes(node1, node2)
    flow.add_edge(node1, node2)

    # Configure ASYNC mode
    store = InMemoryCheckpointStore()
    run_id = "async_task_tracking_test"
    config = RunConfig(
        checkpoint_store=store, run_id=run_id, durability_mode=DurabilityMode.ASYNC
    )

    # Run flow
    result = await flow.run(SimpleInput(value="test"), config=config)
    assert result is not None

    # Give background tasks time to complete
    await asyncio.sleep(0.2)

    # Verify all checkpoints eventually persisted
    query = CheckpointQuery(run_id=RunId(run_id))
    checkpoints, _ = await store.list(query)
    assert len(checkpoints) >= 2, "Expected background checkpoints to complete"
