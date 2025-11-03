"""Tests for flow compilation and execution using stepper engine."""

from collections.abc import AsyncIterator

from pydantic import BaseModel
import pytest

from pydantic_flow import Flow
from pydantic_flow import Route
from pydantic_flow.nodes import BaseNode
from pydantic_flow.nodes import MergeToolNode
from pydantic_flow.streaming.base import ProgressItem
from pydantic_flow.streaming.core_events import StreamEnd
from pydantic_flow.streaming.core_events import StreamStart
from pydantic_flow.streaming.tool_events import ToolResult
from tests.conftest import extract_result_from_stream


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

    async def astream(self, input_data: SimpleState) -> AsyncIterator[ProgressItem]:
        """Stream while passing through."""
        yield StreamStart(run_id=self.run_id or "", node_id=self.name)
        yield ToolResult(result=input_data)
        yield StreamEnd(
            run_id=self.run_id or "",
            node_id=self.name,
            result=input_data.model_dump(),
        )


class TestFlowCompilation:
    """Test flow compilation and execution with stepper engine."""

    @pytest.mark.asyncio
    async def test_compile_and_run_simple_flow(self) -> None:
        """Test compiling and running a simple flow."""
        flow = Flow(input_type=SimpleState, output_type=SimpleOutput)
        node = SimpleNode(name="simple")
        flow.add_nodes(node)

        # Compile and run
        compiled = flow.compile()
        result = await extract_result_from_stream(
            compiled.astream(SimpleState(value=42))
        )

        assert result.simple.value == 42

    @pytest.mark.asyncio
    async def test_compile_flow_with_entry_nodes(self) -> None:
        """Test compiling flow with explicit entry nodes."""
        flow = Flow(input_type=SimpleState, output_type=SimpleOutput)
        node = SimpleNode(name="simple")
        flow.add_nodes(node)
        flow.set_entry_nodes("simple")

        # Compile and run
        compiled = flow.compile()
        result = await extract_result_from_stream(
            compiled.astream(SimpleState(value=42))
        )

        assert result.simple.value == 42

    @pytest.mark.asyncio
    async def test_flow_with_conditional_edges(self) -> None:
        """Test flow with conditional edges compiles and runs."""
        flow = Flow(input_type=SimpleState, output_type=SimpleOutput)
        node = SimpleNode(name="simple")
        flow.add_nodes(node)
        flow.set_entry_nodes("simple")

        def router(state: BaseModel) -> Route | list[Route]:
            return Route.END

        flow.add_conditional_edges("simple", router)

        # Stepper handles conditional edges
        compiled = flow.compile()
        result = await extract_result_from_stream(
            compiled.astream(SimpleState(value=42))
        )

        assert result.simple.value == 42

    @pytest.mark.asyncio
    async def test_flow_with_cycles(self) -> None:
        """Test flow with cycles compiles and runs."""

        class TwoNodeOutput(BaseModel):
            node1: SimpleState
            node2: SimpleState

        flow = Flow(input_type=SimpleState, output_type=TwoNodeOutput)
        node1 = SimpleNode(name="node1")
        node2 = SimpleNode(name="node2")
        flow.add_nodes(node1, node2)

        # Create a cycle that terminates via conditional routing
        flow.add_edge("node1", "node2")
        flow.add_edge("node2", "node1")
        flow.set_entry_nodes("node1")

        def router(state: BaseModel) -> Route | list[Route]:
            return Route.END

        flow.add_conditional_edges("node2", router)

        # Stepper handles cycles
        compiled = flow.compile()
        result = await extract_result_from_stream(
            compiled.astream(SimpleState(value=42))
        )

        assert result.node1.value == 42
        assert result.node2.value == 42

    @pytest.mark.asyncio
    async def test_flow_with_conditional_routing(self) -> None:
        """Test flow with conditional routing."""
        flow = Flow(input_type=SimpleState, output_type=SimpleOutput)
        node = SimpleNode(name="simple")
        flow.add_nodes(node)
        flow.set_entry_nodes("simple")

        def router(state: BaseModel) -> Route | list[Route]:
            return Route.END

        flow.add_conditional_edges("simple", router)

        # Compile and run
        compiled = flow.compile()
        result = await extract_result_from_stream(
            compiled.astream(SimpleState(value=42))
        )

        assert result.simple.value == 42

    @pytest.mark.asyncio
    async def test_flow_with_self_loop(self) -> None:
        """Test flow with self-loop via conditional edge."""
        flow = Flow(input_type=SimpleState, output_type=NodeOutput)
        node = SimpleNode(name="node")
        flow.add_nodes(node)
        flow.set_entry_nodes("node")

        def router(state: BaseModel) -> Route | list[Route]:
            return Route.END

        # Create a self-loop via conditional edge
        flow.add_conditional_edges("node", router)

        # Compile and run
        compiled = flow.compile()
        result = await extract_result_from_stream(
            compiled.astream(SimpleState(value=42))
        )

        assert result.node.value == 42

    @pytest.mark.asyncio
    async def test_flow_with_multi_input_nodes(self) -> None:
        """Test flow with multi-input merge nodes."""
        flow = Flow(input_type=SimpleState, output_type=MergeOutput)
        node1 = SimpleNode(name="node1")
        node2 = SimpleNode(name="node2")

        # MergeToolNode with inputs from both nodes
        async def merge_func(a: SimpleState, b: SimpleState) -> SimpleState:
            return SimpleState(value=a.value + b.value)

        merge = MergeToolNode[SimpleState, SimpleState, SimpleState](
            tool_func=merge_func,
            inputs=(node1.output, node2.output),
            name="merge",
        )
        flow.add_nodes(node1, node2, merge)

        # Compile and run
        compiled = flow.compile()
        result = await extract_result_from_stream(
            compiled.astream(SimpleState(value=10))
        )

        assert result.merge.value == 20  # 10 + 10
