"""Tests for loop support with conditional routing."""

import asyncio

from pydantic import BaseModel
import pytest

from pydantic_flow import Flow
from pydantic_flow import FlowTimeoutError
from pydantic_flow import RecursionLimitError
from pydantic_flow import Route
from pydantic_flow import RoutingError
from pydantic_flow import RunConfig
from pydantic_flow.core.errors import FlowError
from pydantic_flow.core.routing import T_Route
from pydantic_flow.engine.stepper import EngineConfig
from pydantic_flow.nodes import BaseNode
from pydantic_flow.nodes import NodeWithInput


class CounterState(BaseModel):
    """State for counter tests."""

    n: int


class IncrementNode(BaseNode[CounterState, CounterState]):
    """Node that increments a counter."""

    async def run(self, input_data: CounterState) -> CounterState:
        """Increment the counter."""
        return CounterState(n=input_data.n + 1)


class StartNode(BaseNode[CounterState, CounterState]):
    """Start node that passes through input."""

    async def run(self, input_data: CounterState) -> CounterState:
        """Pass through the input."""
        return input_data


class ExecuteNode(BaseNode[CounterState, CounterState]):
    """Execute node for two-node loop."""

    async def run(self, input_data: CounterState) -> CounterState:
        """Increment by 2."""
        return CounterState(n=input_data.n + 2)


class OutputState(BaseModel):
    """Output state with tick node result."""

    tick: CounterState


class TwoNodeOutput(BaseModel):
    """Output for two-node loop."""

    plan: CounterState
    execute: CounterState


class DependentNodeOutput(BaseModel):
    """Output for dependent node test."""

    start: CounterState
    dependent: CounterState


class BooleanRouterOutput(BaseModel):
    """Output for boolean router test."""

    a: CounterState | None = None
    b: CounterState | None = None


class TestLoops:
    """Test loop functionality with conditional routing."""

    @pytest.mark.asyncio
    async def test_self_loop_until_end(self) -> None:
        """Test self-loop that terminates with Route.END."""
        flow = Flow(input_type=CounterState, output_type=OutputState)
        tick_node = IncrementNode(name="tick")
        flow.add_nodes(tick_node)
        flow.set_entry_nodes("tick")

        def router(state: BaseModel) -> T_Route:
            tick_state = getattr(state, "tick", None)
            if tick_state and tick_state.n >= 5:
                return Route.END
            return "tick"

        flow.add_conditional_edges("tick", router)

        compiled = flow.compile()
        result = await compiled.invoke(CounterState(n=0))

        assert result.tick.n == 5

    @pytest.mark.asyncio
    async def test_two_node_loop(self) -> None:
        """Test two-node loop with conditional routing back."""
        flow = Flow(input_type=CounterState, output_type=TwoNodeOutput)
        plan_node = StartNode(name="plan")
        execute_node = ExecuteNode(name="execute")
        flow.add_nodes(plan_node, execute_node)
        flow.set_entry_nodes("plan")
        flow.add_edge("plan", "execute")

        def router(state: BaseModel) -> T_Route:
            execute_state = getattr(state, "execute", None)
            if execute_state and execute_state.n >= 10:
                return Route.END
            return "plan"

        flow.add_conditional_edges("execute", router)

        compiled = flow.compile()
        result = await compiled.invoke(CounterState(n=0))

        assert result.execute.n >= 10

    @pytest.mark.asyncio
    async def test_recursion_limit_triggered(self) -> None:
        """Test that RecursionLimitError is raised when limit exceeded."""
        flow = Flow(input_type=CounterState, output_type=OutputState)
        tick_node = IncrementNode(name="tick")
        flow.add_nodes(tick_node)
        flow.set_entry_nodes("tick")

        def router(state: BaseModel) -> str:
            return "tick"

        flow.add_conditional_edges("tick", router)

        compiled = flow.compile()
        config = RunConfig(max_steps=10)

        with pytest.raises(RecursionLimitError) as exc_info:
            await compiled.invoke(CounterState(n=0), config)

        assert "Exceeded max_steps=10" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_router_mapping_dict(self) -> None:
        """Test router with mapping dict."""
        flow = Flow(input_type=CounterState, output_type=BooleanRouterOutput)
        a_node = IncrementNode(name="a")
        b_node = IncrementNode(name="b")
        flow.add_nodes(a_node, b_node)
        flow.set_entry_nodes("a")

        def router(state: BaseModel) -> str:
            a_state = getattr(state, "a", None)
            if a_state is not None and a_state.n >= 3:
                return "true"
            return "false"

        flow.add_conditional_edges("a", router, mapping={"true": "b", "false": "a"})
        flow.add_conditional_edges("b", lambda s: Route.END)

        compiled = flow.compile()
        result = await compiled.invoke(CounterState(n=0))

        assert result.a is not None
        assert result.a.n >= 3
        assert result.b is not None

    @pytest.mark.asyncio
    async def test_unknown_target_raises(self) -> None:
        """Test that unknown routing target raises RoutingError."""
        flow = Flow(input_type=CounterState, output_type=OutputState)
        tick_node = IncrementNode(name="tick")
        flow.add_nodes(tick_node)
        flow.set_entry_nodes("tick")

        def router(state: BaseModel) -> str:
            return "unknown_node"

        flow.add_conditional_edges("tick", router)

        compiled = flow.compile()

        with pytest.raises(RoutingError) as exc_info:
            await compiled.invoke(CounterState(n=0))

        assert "not a valid node name" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_observability_trace(self) -> None:
        """Test that trace_iterations produces iteration events."""
        flow = Flow(input_type=CounterState, output_type=OutputState)
        tick_node = IncrementNode(name="tick")
        flow.add_nodes(tick_node)
        flow.set_entry_nodes("tick")

        iterations_seen = []

        def router(state: BaseModel) -> T_Route:
            tick_state = getattr(state, "tick", None)
            if tick_state:
                iterations_seen.append(tick_state.n)
            if tick_state and tick_state.n >= 3:
                return Route.END
            return "tick"

        flow.add_conditional_edges("tick", router)

        compiled = flow.compile()
        config = RunConfig(trace_iterations=True)
        result = await compiled.invoke(CounterState(n=0), config)

        assert result.tick.n == 3
        assert len(iterations_seen) > 0

    @pytest.mark.asyncio
    async def test_mapping_with_unknown_outcome_raises(self) -> None:
        """Test that unmapped outcome raises RoutingError."""
        flow = Flow(input_type=CounterState, output_type=OutputState)
        tick_node = IncrementNode(name="tick")
        flow.add_nodes(tick_node)
        flow.set_entry_nodes("tick")

        def router(state: BaseModel) -> str:
            return "unknown_outcome"

        flow.add_conditional_edges("tick", router, mapping={"known": "tick"})

        compiled = flow.compile()

        with pytest.raises(RoutingError) as exc_info:
            await compiled.invoke(CounterState(n=0))

        assert "not in mapping" in str(exc_info.value)

    def test_set_entry_nodes_with_unknown_node_raises(self) -> None:
        """Test that set_entry_nodes validates node names."""
        flow = Flow(input_type=CounterState, output_type=OutputState)
        tick_node = IncrementNode(name="tick")
        flow.add_nodes(tick_node)

        with pytest.raises(ValueError) as exc_info:
            flow.set_entry_nodes("nonexistent")

        assert "Unknown entry nodes" in str(exc_info.value)
        assert "nonexistent" in str(exc_info.value)

    def test_set_entry_nodes_with_no_nodes_raises(self) -> None:
        """Test that set_entry_nodes requires at least one node."""
        flow = Flow(input_type=CounterState, output_type=OutputState)

        with pytest.raises(ValueError) as exc_info:
            flow.set_entry_nodes()

        assert "Must specify at least one entry node" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_timeout_during_execution(self) -> None:
        """Test that FlowTimeoutError is raised when timeout exceeds limit."""
        flow = Flow(input_type=CounterState, output_type=OutputState)

        class SlowNode(BaseNode[CounterState, CounterState]):
            """Node that sleeps to trigger timeout."""

            async def run(self, input_data: CounterState) -> CounterState:
                """Sleep to accumulate time over multiple iterations."""
                await asyncio.sleep(0.3)
                return CounterState(n=input_data.n + 1)

        slow_node = SlowNode(name="tick")
        flow.add_nodes(slow_node)
        flow.set_entry_nodes("tick")

        def router(state: BaseModel) -> T_Route:
            tick_state = getattr(state, "tick", None)
            if tick_state and tick_state.n >= 10:
                return Route.END
            return "tick"

        flow.add_conditional_edges("tick", router)

        compiled = flow.compile()
        config = RunConfig(timeout_seconds=1, max_steps=50)

        with pytest.raises(FlowTimeoutError) as exc_info:
            await compiled.invoke(CounterState(n=0), config)

        assert "timeout" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_invalid_input_type_raises(self) -> None:
        """Test that wrong input type raises TypeError."""
        flow = Flow(input_type=CounterState, output_type=OutputState)
        tick_node = IncrementNode(name="tick")
        flow.add_nodes(tick_node)
        flow.set_entry_nodes("tick")

        def router(state: BaseModel) -> T_Route:
            return Route.END

        flow.add_conditional_edges("tick", router)

        compiled = flow.compile()

        class WrongState(BaseModel):
            """Wrong state type."""

            x: str

        with pytest.raises(TypeError) as exc_info:
            await compiled.invoke(WrongState(x="wrong"))  # type: ignore[arg-type]

        assert "Input type mismatch" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_node_exception_wrapped_in_flow_error(self) -> None:
        """Test that node exceptions are wrapped in FlowError."""
        flow = Flow(input_type=CounterState, output_type=OutputState)

        class BrokenNode(BaseNode[CounterState, CounterState]):
            """Node that raises an exception."""

            async def run(self, input_data: CounterState) -> CounterState:
                """Raise a generic exception."""
                raise RuntimeError("Node is broken!")

        broken_node = BrokenNode(name="tick")
        flow.add_nodes(broken_node)
        flow.set_entry_nodes("tick")

        def router(state: BaseModel) -> T_Route:
            return Route.END

        flow.add_conditional_edges("tick", router)

        compiled = flow.compile()

        with pytest.raises(FlowError) as exc_info:
            await compiled.invoke(CounterState(n=0))

        assert "Flow execution failed" in str(exc_info.value)

    def test_engine_config_validates_entry_nodes(self) -> None:
        """Test that EngineConfig validates entry nodes exist."""
        tick_node = IncrementNode(name="tick")

        with pytest.raises(ValueError) as exc_info:
            EngineConfig(
                nodes=[tick_node],
                edges={},
                conditional_edges=[],
                entry_nodes=["nonexistent"],
                input_type=CounterState,
                output_type=OutputState,
            )

        assert "Unknown entry nodes" in str(exc_info.value)

    def test_engine_config_validates_edge_targets(self) -> None:
        """Test that EngineConfig validates edge targets exist."""
        tick_node = IncrementNode(name="tick")

        with pytest.raises(ValueError) as exc_info:
            EngineConfig(
                nodes=[tick_node],
                edges={"tick": ["nonexistent"]},
                conditional_edges=[],
                entry_nodes=["tick"],
                input_type=CounterState,
                output_type=OutputState,
            )

        assert "Unknown edge targets" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_node_with_explicit_input_dependency(self) -> None:
        """Test node with explicit input dependency in stepper."""
        flow = Flow(input_type=CounterState, output_type=DependentNodeOutput)
        start_node = StartNode(name="start")

        class DependentNode(NodeWithInput[CounterState, CounterState]):
            """Node with explicit input dependency."""

            async def run(self, input_data: CounterState) -> CounterState:
                """Pass through with increment."""
                return CounterState(n=input_data.n + 10)

        dependent_node = DependentNode(input=start_node.output, name="dependent")
        flow.add_nodes(start_node, dependent_node)
        flow.set_entry_nodes("start")
        flow.add_edge("start", "dependent")

        def router(state: BaseModel) -> T_Route:
            return Route.END

        flow.add_conditional_edges("dependent", router)

        compiled = flow.compile()
        result = await compiled.invoke(CounterState(n=5), RunConfig())

        assert result.start.n == 5
        assert result.dependent.n == 15
