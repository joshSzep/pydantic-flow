"""Tests for comprehensive coverage of stepper engine."""

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel
import pytest

from pydantic_flow import Flow
from pydantic_flow.core.errors import FlowError
from pydantic_flow.core.errors import FlowTimeoutError
from pydantic_flow.core.errors import RecursionLimitError
from pydantic_flow.core.errors import RoutingError
from pydantic_flow.core.routing import Route
from pydantic_flow.core.routing import T_Route
from pydantic_flow.core.run_config import RunConfig
from pydantic_flow.engine.stepper import ConditionalEdge
from pydantic_flow.engine.stepper import EngineConfig
from pydantic_flow.engine.stepper import IterationEvent
from pydantic_flow.engine.stepper import StepperEngine
from pydantic_flow.nodes import BaseNode
from pydantic_flow.streaming import ProgressItem
from pydantic_flow.streaming import StreamEnd
from pydantic_flow.streaming import StreamStart


class SimpleState(BaseModel):
    """Simple state for testing."""

    value: int


class MultiState(BaseModel):
    """State with multiple fields."""

    a: SimpleState
    b: SimpleState


class LoopState(BaseModel):
    """Output for loop test."""

    loop: SimpleState


class SimpleNode(BaseNode[SimpleState, SimpleState]):
    """Simple node for testing."""

    async def astream(self, input_data: SimpleState) -> AsyncIterator[ProgressItem]:
        """Increment value."""
        yield StreamStart(run_id=self.run_id or "", node_id=self.name)
        result = SimpleState(value=input_data.value + 1)
        yield StreamEnd(
            run_id=self.run_id or "",
            node_id=self.name,
            result_preview=result.model_dump(),
        )


class SlowNode(BaseNode[SimpleState, SimpleState]):
    """Node that takes time to execute."""

    async def astream(self, input_data: SimpleState) -> AsyncIterator[ProgressItem]:
        """Slow execution."""
        yield StreamStart(run_id=self.run_id or "", node_id=self.name)
        await asyncio.sleep(2.0)  # Sleep longer than timeout
        result = SimpleState(value=input_data.value + 1)
        yield StreamEnd(
            run_id=self.run_id or "",
            node_id=self.name,
            result_preview=result.model_dump(),
        )


class ErrorNode(BaseNode[SimpleState, SimpleState]):
    """Node that raises an error."""

    async def astream(self, input_data: SimpleState) -> AsyncIterator[ProgressItem]:
        """Raise error."""
        yield StreamStart(run_id=self.run_id or "", node_id=self.name)
        raise ValueError("Test error")


class TestInputValidation:
    """Tests for input type validation."""

    @pytest.mark.asyncio
    async def test_input_type_mismatch(self) -> None:
        """Test that wrong input type raises TypeError."""
        node = SimpleNode(name="node")
        config = EngineConfig(
            nodes=[node],
            edges={},
            conditional_edges=[],
            entry_nodes=["node"],
            input_type=SimpleState,
            output_type=SimpleState,
        )
        engine = StepperEngine(config)

        # Pass wrong type
        class WrongState(BaseModel):
            wrong_field: str

        wrong_input: Any = WrongState(wrong_field="test")
        with pytest.raises(TypeError) as exc_info:
            await engine.invoke(wrong_input, RunConfig())

        assert "Input type mismatch" in str(exc_info.value)
        assert "expected SimpleState" in str(exc_info.value)
        assert "got WrongState" in str(exc_info.value)


class TestRecursionLimit:
    """Tests for recursion limit handling."""

    @pytest.mark.asyncio
    async def test_recursion_limit_reached(self) -> None:
        """Test that exceeding max_steps raises RecursionLimitError."""
        flow = Flow(input_type=SimpleState, output_type=SimpleState)
        node = SimpleNode(name="loop")
        flow.add_nodes(node)
        flow.set_entry_nodes("loop")

        # Create a loop
        def router(state: BaseModel) -> T_Route:
            return "loop"  # Always loop back

        flow.add_conditional_edges("loop", router)

        compiled = flow.compile()

        with pytest.raises(RecursionLimitError) as exc_info:
            await compiled.invoke(SimpleState(value=0), RunConfig(max_steps=3))

        assert "Exceeded max_steps=3" in str(exc_info.value)
        assert "at iteration 3" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_recursion_limit_with_trace(self) -> None:
        """Test that RecursionLimitError includes recent events when tracing."""
        flow = Flow(input_type=SimpleState, output_type=SimpleState)
        node = SimpleNode(name="loop")
        flow.add_nodes(node)
        flow.set_entry_nodes("loop")

        def router(state: BaseModel) -> T_Route:
            return "loop"

        flow.add_conditional_edges("loop", router)

        compiled = flow.compile()

        with pytest.raises(RecursionLimitError) as exc_info:
            await compiled.invoke(
                SimpleState(value=0),
                RunConfig(max_steps=3, trace_iterations=True, recent_events_count=2),
            )

        error_msg = str(exc_info.value)
        assert "Exceeded max_steps=3" in error_msg
        assert "Recent iterations:" in error_msg


class SlowNodeOutput(BaseModel):
    """Output for slow node test."""

    slow: SimpleState


class TestTimeout:
    """Tests for timeout handling."""

    @pytest.mark.asyncio
    async def test_timeout_exceeded(self) -> None:
        """Test that exceeding timeout raises FlowTimeoutError."""
        flow = Flow(input_type=SimpleState, output_type=LoopState)
        node = SlowNode(name="loop")
        flow.add_nodes(node)
        flow.set_entry_nodes("loop")

        # Create a loop that runs multiple times slowly
        loop_count = 0

        def router(state: BaseModel) -> T_Route:
            nonlocal loop_count
            loop_count += 1
            if loop_count < 5:  # Loop 5 times with 2s each = 10s total
                return "loop"
            return Route.END

        flow.add_conditional_edges("loop", router)

        compiled = flow.compile()

        with pytest.raises(FlowTimeoutError) as exc_info:
            # Timeout after 1 second, but loop will take ~10 seconds
            await compiled.invoke(SimpleState(value=0), RunConfig(timeout_seconds=1))

        assert "Exceeded timeout of 1s" in str(exc_info.value)


class TestNodeInputRetrieval:
    """Tests for _get_node_input method."""

    @pytest.mark.asyncio
    async def test_node_uses_previous_result_in_loop(self) -> None:
        """Test that node with no dependencies uses previous result in loop."""
        flow = Flow(input_type=SimpleState, output_type=LoopState)
        node = SimpleNode(name="loop")
        flow.add_nodes(node)
        flow.set_entry_nodes("loop")

        # Create a finite loop: loop 2 times then end
        loop_count = 0

        def router(state: BaseModel) -> T_Route:
            nonlocal loop_count
            loop_count += 1
            if loop_count < 2:
                return "loop"
            return Route.END

        flow.add_conditional_edges("loop", router)

        compiled = flow.compile()
        result = await compiled.invoke(SimpleState(value=0), RunConfig())

        # First iteration: 0 + 1 = 1
        # Second iteration: 1 + 1 = 2
        assert result.loop.value == 2


class TestConditionalRouting:
    """Tests for conditional routing edge cases."""

    @pytest.mark.asyncio
    async def test_router_outcome_not_in_mapping(self) -> None:
        """Test that unmapped router outcome raises RoutingError."""
        flow = Flow(input_type=SimpleState, output_type=SimpleState)
        node = SimpleNode(name="start")
        flow.add_nodes(node)
        flow.set_entry_nodes("start")

        # Create mapping but return unmapped value
        def router(state: BaseModel) -> T_Route:
            return "unmapped"

        mapping = {"option_a": "start", "option_b": Route.END}
        flow.add_conditional_edges("start", router, mapping=mapping)

        compiled = flow.compile()

        with pytest.raises(RoutingError) as exc_info:
            await compiled.invoke(SimpleState(value=0), RunConfig())

        assert "Router outcome 'unmapped' not in mapping" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_router_returns_invalid_node_name(self) -> None:
        """Test that invalid node name raises RoutingError."""
        flow = Flow(input_type=SimpleState, output_type=SimpleState)
        node = SimpleNode(name="start")
        flow.add_nodes(node)
        flow.set_entry_nodes("start")

        def router(state: BaseModel) -> T_Route:
            return "nonexistent_node"

        flow.add_conditional_edges("start", router)

        compiled = flow.compile()

        with pytest.raises(RoutingError) as exc_info:
            await compiled.invoke(SimpleState(value=0), RunConfig())

        assert "Router target 'nonexistent_node' is not a valid node name" in str(
            exc_info.value
        )

    @pytest.mark.asyncio
    async def test_router_returns_invalid_type(self) -> None:
        """Test that invalid routing target type raises RoutingError."""
        flow = Flow(input_type=SimpleState, output_type=SimpleState)
        node = SimpleNode(name="start")
        flow.add_nodes(node)
        flow.set_entry_nodes("start")

        def router(state: BaseModel) -> Any:
            return 123  # Invalid type

        flow.add_conditional_edges("start", router)

        compiled = flow.compile()

        with pytest.raises(RoutingError) as exc_info:
            await compiled.invoke(SimpleState(value=0), RunConfig())

        assert "Invalid routing target: 123" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_router_returns_list_of_routes(self) -> None:
        """Test that router can return list of routes with static edge."""
        flow = Flow(input_type=SimpleState, output_type=MultiState)
        node_a = SimpleNode(name="a")
        node_b = SimpleNode(name="b")
        flow.add_nodes(node_a, node_b)
        flow.set_entry_nodes("a")

        # Router from a returns list with b
        def router_a(state: BaseModel) -> list[T_Route]:
            return ["b"]

        flow.add_conditional_edges("a", router_a)
        flow.add_conditional_edges("b", lambda s: Route.END)

        compiled = flow.compile()
        result = await compiled.invoke(SimpleState(value=0), RunConfig())

        # Both nodes should be executed, both use flow input
        assert result.a.value == 1
        # b also uses flow input since no explicit dependency
        assert result.b.value == 1

    @pytest.mark.asyncio
    async def test_conditional_edge_with_mapping(self) -> None:
        """Test conditional edge with mapping."""
        flow = Flow(input_type=SimpleState, output_type=MultiState)
        node_a = SimpleNode(name="a")
        node_b = SimpleNode(name="b")
        flow.add_nodes(node_a, node_b)
        flow.set_entry_nodes("a")

        def router(state: BaseModel) -> T_Route:
            return "go_to_b"

        mapping = {"go_to_b": "b", "done": Route.END}
        flow.add_conditional_edges("a", router, mapping=mapping)
        flow.add_conditional_edges("b", lambda s: Route.END)

        compiled = flow.compile()
        result = await compiled.invoke(SimpleState(value=0), RunConfig())

        assert result.a.value == 1
        assert result.b.value == 1  # b also uses flow input


class SingleNodeOutput(BaseModel):
    """Output for single node test."""

    a: SimpleState


class TestIterationTracing:
    """Tests for iteration event tracing."""

    @pytest.mark.asyncio
    async def test_trace_iterations_enabled(self) -> None:
        """Test that trace_iterations captures events."""
        flow = Flow(input_type=SimpleState, output_type=SingleNodeOutput)
        node_a = SimpleNode(name="a")
        flow.add_nodes(node_a)
        flow.set_entry_nodes("a")
        flow.add_conditional_edges("a", lambda state: Route.END)

        compiled = flow.compile()
        result = await compiled.invoke(
            SimpleState(value=0), RunConfig(trace_iterations=True)
        )

        assert result.a.value == 1


class TestErrorHandling:
    """Tests for general error handling."""

    @pytest.mark.asyncio
    async def test_node_error_wrapped_in_flow_error(self) -> None:
        """Test that node errors are wrapped in FlowError."""
        flow = Flow(input_type=SimpleState, output_type=SimpleState)
        node = ErrorNode(name="error")
        flow.add_nodes(node)
        flow.set_entry_nodes("error")
        flow.add_conditional_edges("error", lambda s: Route.END)

        compiled = flow.compile()

        with pytest.raises(FlowError) as exc_info:
            await compiled.invoke(SimpleState(value=0), RunConfig())

        assert "Flow execution failed" in str(exc_info.value)


class TestConditionalEdgeConstruction:
    """Tests for ConditionalEdge class."""

    def test_conditional_edge_init(self) -> None:
        """Test ConditionalEdge initialization."""

        def router(state: SimpleState) -> T_Route:
            return "target"

        edge = ConditionalEdge(from_node="source", router=router, mapping=None)

        assert edge.from_node == "source"
        assert edge.router == router
        assert edge.mapping is None

    def test_conditional_edge_with_mapping(self) -> None:
        """Test ConditionalEdge with mapping."""

        def router(state: SimpleState) -> T_Route:
            return "key"

        mapping = {"key": "target"}
        edge = ConditionalEdge(from_node="source", router=router, mapping=mapping)

        assert edge.mapping == mapping


class TestIterationEvent:
    """Tests for IterationEvent model."""

    def test_iteration_event_creation(self) -> None:
        """Test IterationEvent creation and serialization."""
        event = IterationEvent(
            iteration=0,
            frontier=["node_a"],
            routed_to=["node_b"],
            ended=False,
            elapsed_ms=123.45,
        )

        assert event.iteration == 0
        assert event.frontier == ["node_a"]
        assert event.routed_to == ["node_b"]
        assert event.ended is False
        assert event.elapsed_ms == 123.45

        # Test serialization
        dumped = event.model_dump()
        assert dumped["iteration"] == 0
        assert dumped["ended"] is False
