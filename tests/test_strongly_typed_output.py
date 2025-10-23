"""Tests for strongly typed Flow output functionality."""

from pydantic import BaseModel
import pytest

from pydantic_flow import Flow
from pydantic_flow import ToolNode

# Test constants
EXPECTED_RESULT_10 = 10
EXPECTED_RESULT_6 = 6
EXPECTED_RESULT_14 = 14


class InputModel(BaseModel):
    """Test input model."""

    value: int


class OutputModel(BaseModel):
    """Test output model."""

    result: int


class FlowResults(BaseModel):
    """Strongly typed results structure as BaseModel."""

    multiply_by_two: OutputModel


class SingleNodeResults(BaseModel):
    """Results structure for single node flows."""

    test_node: OutputModel


def multiply_by_two(input_data: InputModel) -> OutputModel:
    """Multiply input value by two."""
    return OutputModel(result=input_data.value * 2)


class TestStronglyTypedFlow:
    """Test strongly typed Flow functionality."""

    @pytest.mark.asyncio
    async def test_strongly_typed_flow_execution(self):
        """Test that strongly typed flow returns correctly typed results."""
        # Create a flow with specific input and output types
        flow: Flow[InputModel, FlowResults] = Flow(
            input_type=InputModel, output_type=FlowResults
        )

        node = ToolNode[InputModel, OutputModel](
            tool_func=multiply_by_two,
            name="multiply_by_two",
        )

        flow.add_nodes(node)

        # Execute the flow
        input_data = InputModel(value=5)
        results = await flow.run(input_data)

        # Results should be typed as FlowResults (BaseModel)
        # In a type-aware IDE, this would provide auto-completion
        assert isinstance(results, FlowResults)
        assert isinstance(results.multiply_by_two, OutputModel)
        assert results.multiply_by_two.result == EXPECTED_RESULT_10

    @pytest.mark.asyncio
    async def test_flow_with_basemodel_output_type(self):
        """Test flow with BaseModel output type."""
        # Create a flow with BaseModel output
        flow: Flow[InputModel, SingleNodeResults] = Flow(
            input_type=InputModel, output_type=SingleNodeResults
        )

        node = ToolNode[InputModel, OutputModel](
            tool_func=multiply_by_two,
            name="test_node",
        )

        flow.add_nodes(node)

        # Execute the flow
        input_data = InputModel(value=3)
        results = await flow.run(input_data)

        # Results should be a BaseModel instance
        assert isinstance(results, SingleNodeResults)
        assert isinstance(results.test_node, OutputModel)
        assert results.test_node.result == EXPECTED_RESULT_6

    def test_flow_type_annotations(self):
        """Test that Flow properly accepts type parameters."""
        # These should all compile without type errors

        # Strongly typed flow
        typed_flow: Flow[InputModel, FlowResults] = Flow(
            input_type=InputModel, output_type=FlowResults
        )
        assert len(typed_flow.nodes) == 0

        # Another BaseModel output type
        single_flow: Flow[InputModel, SingleNodeResults] = Flow(
            input_type=InputModel, output_type=SingleNodeResults
        )
        assert len(single_flow.nodes) == 0

        # Both flows should be Flow instances
        assert isinstance(typed_flow, Flow)
        assert isinstance(single_flow, Flow)

    @pytest.mark.asyncio
    async def test_backward_compatibility(self):
        """Test flow with explicit output type specification."""
        # Create a flow with explicit output type
        flow: Flow[InputModel, SingleNodeResults] = Flow(
            input_type=InputModel, output_type=SingleNodeResults
        )

        node = ToolNode[InputModel, OutputModel](
            tool_func=multiply_by_two,
            name="test_node",
        )

        flow.add_nodes(node)

        input_data = InputModel(value=7)
        results = await flow.run(input_data)

        # Should return a BaseModel instance
        assert isinstance(results, SingleNodeResults)
        assert isinstance(results.test_node, OutputModel)
        assert results.test_node.result == EXPECTED_RESULT_14
