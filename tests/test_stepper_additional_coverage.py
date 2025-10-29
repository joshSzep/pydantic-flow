"""Additional tests for engine/stepper.py to achieve full coverage."""

import asyncio
from collections.abc import AsyncIterator

from pydantic import BaseModel
import pytest

from pydantic_flow.cache.base import CachePolicy
from pydantic_flow.core.errors import FlowError
from pydantic_flow.core.errors import RoutingError
from pydantic_flow.core.routing import T_Route
from pydantic_flow.engine.stepper import ConditionalEdge
from pydantic_flow.engine.stepper import EngineConfig
from pydantic_flow.engine.stepper import StepperEngine
from pydantic_flow.nodes.base import BaseNode
from pydantic_flow.streaming.events import ProgressItem
from pydantic_flow.streaming.events import StreamEnd
from pydantic_flow.streaming.events import StreamStart


class SimpleState(BaseModel):
    """Simple state model."""

    value: int


class IncrementNode(BaseNode[SimpleState, SimpleState]):
    """Node that increments input value."""

    async def astream(self, input_data: SimpleState) -> AsyncIterator[ProgressItem]:
        """Increment the value."""
        yield StreamStart(run_id=self.run_id or "", node_id=self.name)
        result = SimpleState(value=input_data.value + 1)
        yield StreamEnd(
            run_id=self.run_id or "",
            node_id=self.name,
            result_preview=result.model_dump(),
        )


class ErrorNode(BaseNode[SimpleState, SimpleState]):
    """Node that raises an error."""

    async def astream(self, input_data: SimpleState) -> AsyncIterator[ProgressItem]:
        """Raise an error."""
        yield StreamStart(run_id=self.run_id or "", node_id=self.name)
        raise RuntimeError("Node execution failed")


class SlowNode(BaseNode[SimpleState, SimpleState]):
    """Node that takes time to execute."""

    async def astream(self, input_data: SimpleState) -> AsyncIterator[ProgressItem]:
        """Sleep then return result."""
        yield StreamStart(run_id=self.run_id or "", node_id=self.name)
        await asyncio.sleep(2)
        result = SimpleState(value=input_data.value)
        yield StreamEnd(
            run_id=self.run_id or "",
            node_id=self.name,
            result_preview=result.model_dump(),
        )


class CacheableNode(BaseNode[SimpleState, SimpleState]):
    """Node with cache policy."""

    def __init__(self, name: str, cache_policy: CachePolicy) -> None:
        """Initialize with cache policy."""
        super().__init__(name=name)
        self.cache_policy = cache_policy

    async def astream(self, input_data: SimpleState) -> AsyncIterator[ProgressItem]:
        """Return incremented value."""
        yield StreamStart(run_id=self.run_id or "", node_id=self.name)
        result = SimpleState(value=input_data.value + 10)
        yield StreamEnd(
            run_id=self.run_id or "",
            node_id=self.name,
            result_preview=result.model_dump(),
        )


@pytest.mark.asyncio
class TestStepperEngineAdditionalCoverage:
    """Tests for uncovered paths in StepperEngine."""

    async def test_config_validation_unknown_entry_nodes(self) -> None:
        """Test that unknown entry nodes raise ValueError."""
        node1 = IncrementNode(name="node1")

        with pytest.raises(
            ValueError, match="Unknown entry nodes: \\['nonexistent'\\]"
        ):
            EngineConfig[SimpleState, SimpleState](
                nodes=[node1],
                entry_nodes=["nonexistent"],
                input_type=SimpleState,
                output_type=SimpleState,
            )

    async def test_config_validation_unknown_edge_targets(self) -> None:
        """Test that unknown edge targets raise ValueError."""
        node1 = IncrementNode(name="node1")

        with pytest.raises(
            ValueError,
            match="Unknown edge targets from 'node1': \\['nonexistent'\\]",
        ):
            EngineConfig[SimpleState, SimpleState](
                nodes=[node1],
                edges={"node1": ["nonexistent"]},
                entry_nodes=["node1"],
                input_type=SimpleState,
                output_type=SimpleState,
            )

    async def test_conditional_edge_outcome_not_in_mapping_raises(
        self,
    ) -> None:
        """Test that unmapped router outcome raises RoutingError."""
        node1 = IncrementNode(name="node1")
        node2 = IncrementNode(name="node2")

        def router(_state: BaseModel) -> str:
            return "unknown_key"

        cond_edge = ConditionalEdge[BaseModel](
            from_node="node1",
            router=router,
            mapping={"valid_key": "node2"},
        )

        config = EngineConfig[SimpleState, SimpleState](
            nodes=[node1, node2],
            entry_nodes=["node1"],
            conditional_edges=[cond_edge],
            input_type=SimpleState,
            output_type=SimpleState,
        )

        engine = StepperEngine(config)

        with pytest.raises(RoutingError, match="not in mapping"):
            await engine.invoke(SimpleState(value=50))

    async def test_conditional_edge_invalid_node_target_raises(self) -> None:
        """Test that routing to non-existent node raises RoutingError."""
        node1 = IncrementNode(name="node1")

        def router(_state: BaseModel) -> str:
            return "nonexistent_node"

        cond_edge = ConditionalEdge[BaseModel](
            from_node="node1",
            router=router,
        )

        config = EngineConfig[SimpleState, SimpleState](
            nodes=[node1],
            entry_nodes=["node1"],
            conditional_edges=[cond_edge],
            input_type=SimpleState,
            output_type=SimpleState,
        )

        engine = StepperEngine(config)

        with pytest.raises(RoutingError, match="not a valid node name"):
            await engine.invoke(SimpleState(value=60))

    async def test_conditional_edge_invalid_type_raises(self) -> None:
        """Test that routing with invalid type raises RoutingError."""
        node1 = IncrementNode(name="node1")

        def router(_state: BaseModel) -> T_Route:
            return 123  # type: ignore[return-value]

        cond_edge = ConditionalEdge[BaseModel](
            from_node="node1",
            router=router,
        )

        config = EngineConfig[SimpleState, SimpleState](
            nodes=[node1],
            entry_nodes=["node1"],
            conditional_edges=[cond_edge],
            input_type=SimpleState,
            output_type=SimpleState,
        )

        engine = StepperEngine(config)

        with pytest.raises(RoutingError, match="Invalid routing target"):
            await engine.invoke(SimpleState(value=70))

    async def test_node_execution_error_wrapped_in_flow_error(self) -> None:
        """Test that node execution errors are wrapped in FlowError."""
        error_node = ErrorNode(name="value")

        config = EngineConfig[SimpleState, SimpleState](
            nodes=[error_node],
            entry_nodes=["value"],
            input_type=SimpleState,
            output_type=SimpleState,
        )

        engine = StepperEngine(config)

        with pytest.raises(FlowError, match="Flow execution failed"):
            await engine.invoke(SimpleState(value=80))
