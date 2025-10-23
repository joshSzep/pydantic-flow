"""Tests for ExecutionMode and flow compilation options."""

from pydantic import BaseModel
import pytest

from pydantic_flow import ExecutionMode
from pydantic_flow import Flow
from pydantic_flow import Route
from pydantic_flow.core.errors import FlowError
from pydantic_flow.core.routing import T_Route
from pydantic_flow.nodes import BaseNode
from pydantic_flow.nodes import MergeToolNode


class SimpleState(BaseModel):
    """Simple state for testing."""

    value: int


class SimpleOutput(BaseModel):
    """Output with simple node result."""

    simple: SimpleState


class NodeOutput(BaseModel):
    """Output with node field."""

    node: SimpleState


class MergeOutput(BaseModel):
    """Output with merge node result."""

    node1: SimpleState
    node2: SimpleState
    merge: SimpleState


class SimpleNode(BaseNode[SimpleState, SimpleState]):
    """Simple node for testing."""

    async def run(self, input_data: SimpleState) -> SimpleState:
        """Pass through."""
        return input_data


class TestExecutionMode:
    """Test ExecutionMode selection and flow compilation."""

    @pytest.mark.asyncio
    async def test_explicit_dag_mode_with_simple_flow(self) -> None:
        """Test explicitly forcing DAG mode on a simple flow."""
        flow = Flow(input_type=SimpleState, output_type=SimpleOutput)
        node = SimpleNode(name="simple")
        flow.add_nodes(node)

        # Should work fine with DAG mode
        compiled = flow.compile(mode=ExecutionMode.DAG)
        result = await compiled.invoke(SimpleState(value=42))

        assert result.simple.value == 42

    @pytest.mark.asyncio
    async def test_explicit_stepper_mode_with_simple_flow(self) -> None:
        """Test explicitly forcing stepper mode on a simple flow."""
        flow = Flow(input_type=SimpleState, output_type=SimpleOutput)
        node = SimpleNode(name="simple")
        flow.add_nodes(node)
        flow.set_entry_nodes("simple")

        # Should work fine with stepper mode
        compiled = flow.compile(mode=ExecutionMode.STEPPER)
        result = await compiled.invoke(SimpleState(value=42))

        assert result.simple.value == 42

    @pytest.mark.asyncio
    async def test_dag_mode_with_conditional_edges_raises_error(self) -> None:
        """Test that DAG mode raises error if conditional edges exist."""
        flow = Flow(input_type=SimpleState, output_type=SimpleState)
        node = SimpleNode(name="simple")
        flow.add_nodes(node)
        flow.set_entry_nodes("simple")

        def router(state: BaseModel) -> T_Route:
            return Route.END

        flow.add_conditional_edges("simple", router)

        # Should raise FlowError
        with pytest.raises(
            FlowError, match="Cannot use DAG mode with conditional edges"
        ):
            flow.compile(mode=ExecutionMode.DAG)

    @pytest.mark.asyncio
    async def test_dag_mode_with_cycles_raises_error(self) -> None:
        """Test that DAG mode raises error if cycles exist."""
        flow = Flow(input_type=SimpleState, output_type=SimpleState)
        node1 = SimpleNode(name="node1")
        node2 = SimpleNode(name="node2")
        flow.add_nodes(node1, node2)

        # Create a cycle
        flow.add_edge("node1", "node2")
        flow.add_edge("node2", "node1")
        flow.set_entry_nodes("node1")

        # Should raise FlowError
        with pytest.raises(
            FlowError, match="Cannot use DAG mode with cyclic dependencies"
        ):
            flow.compile(mode=ExecutionMode.DAG)

    @pytest.mark.asyncio
    async def test_auto_mode_selects_stepper_for_conditional_edges(self) -> None:
        """Test that AUTO mode selects stepper when conditional edges exist."""
        flow = Flow(input_type=SimpleState, output_type=SimpleOutput)
        node = SimpleNode(name="simple")
        flow.add_nodes(node)
        flow.set_entry_nodes("simple")

        def router(state: BaseModel) -> T_Route:
            return Route.END

        flow.add_conditional_edges("simple", router)

        # AUTO mode should select stepper (no error)
        compiled = flow.compile(mode=ExecutionMode.AUTO)
        result = await compiled.invoke(SimpleState(value=42))

        assert result.simple.value == 42

    @pytest.mark.asyncio
    async def test_auto_mode_selects_stepper_for_cycles(self) -> None:
        """Test that AUTO mode selects stepper when cycles exist."""
        flow = Flow(input_type=SimpleState, output_type=NodeOutput)
        node = SimpleNode(name="node")
        flow.add_nodes(node)
        flow.set_entry_nodes("node")

        def router(state: BaseModel) -> T_Route:
            return Route.END

        # Create a self-loop via conditional edge
        flow.add_conditional_edges("node", router)

        # AUTO mode should select stepper (no error)
        compiled = flow.compile(mode=ExecutionMode.AUTO)
        result = await compiled.invoke(SimpleState(value=42))

        assert result.node.value == 42

    @pytest.mark.asyncio
    async def test_detect_cycles_efficiently_with_multi_input_nodes(self) -> None:
        """Test cycle detection handles multi-input nodes correctly."""
        flow = Flow(input_type=SimpleState, output_type=MergeOutput)
        node1 = SimpleNode(name="node1")
        node2 = SimpleNode(name="node2")

        # MergeToolNode with inputs from both nodes
        def merge_func(a: SimpleState, b: SimpleState) -> SimpleState:
            return SimpleState(value=a.value + b.value)

        merge = MergeToolNode[SimpleState, SimpleState, SimpleState](
            tool_func=merge_func,
            inputs=(node1.output, node2.output),
            name="merge",
        )
        flow.add_nodes(node1, node2, merge)

        # No cycles - should compile in DAG mode
        compiled = flow.compile(mode=ExecutionMode.DAG)
        result = await compiled.invoke(SimpleState(value=10))

        assert result.merge.value == 20  # 10 + 10
