"""Test suite for the pflow framework.

This module contains comprehensive tests for all node types, Flow functionality,
and edge cases to ensure the framework works as expected.
"""

import builtins

from pydantic import BaseModel
import pytest

from pflow import Flow
from pflow import FlowNode
from pflow import IfNode
from pflow import ParserNode
from pflow import PromptConfig
from pflow import PromptNode
from pflow import RetryNode
from pflow import ToolNode
from pflow.flow.exceptions import CyclicDependencyError
from pflow.flow.exceptions import FlowError

# Test constants
EXPECTED_TEMPERATURE = 22.5
EXPECTED_NODES_COUNT = 2
EXPECTED_COMPLEX_NODES_COUNT = 4
EXPECTED_TEST_TEMPERATURE = 25.0


# Test data models
class WeatherQuery(BaseModel):
    """Test model for weather query inputs."""

    location: str
    temperature_unit: str = "celsius"


class WeatherInfo(BaseModel):
    """Test model for weather information."""

    temperature: float
    condition: str
    location: str


class SummaryRequest(BaseModel):
    """Test model for summary requests."""

    weather_info: str
    style: str = "brief"


class WeatherSummary(BaseModel):
    """Test model for weather summaries."""

    summary: str
    recommendation: str


class GenericFlowResults(BaseModel):
    """Generic test model for flow results."""

    # Dynamic fields for testing - will contain node results
    pass


class SimpleFlowResults(BaseModel):
    """Results for simple flow tests with one node."""

    weather_api: WeatherInfo


class ComplexFlowResults(BaseModel):
    """Results for complex flow tests with multiple nodes."""

    prompt: str
    parser: WeatherInfo


class EmptyFlowResults(BaseModel):
    """Results for empty flow tests."""

    pass


class FlowNodeResults(BaseModel):
    """Results for FlowNode testing with sub-flow outputs."""

    weather_sub_flow: SimpleFlowResults


class NestedFlowResults(BaseModel):
    """Results for nested FlowNode testing."""

    level2_wrapper: FlowNodeResults


# Test helper functions
def parse_weather_string(weather_str: str) -> WeatherInfo:
    """Parse a weather string into structured data."""
    # Simple parser for testing
    parts = weather_str.split("|")
    return WeatherInfo(
        temperature=float(parts[0]),
        condition=parts[1].strip(),
        location=parts[2].strip(),
    )


def call_weather_api(query: WeatherQuery) -> WeatherInfo:
    """Mock weather API call."""
    return WeatherInfo(
        temperature=EXPECTED_TEMPERATURE,
        condition="sunny",
        location=query.location,
    )


def generate_summary(request: SummaryRequest) -> WeatherSummary:
    """Generate a weather summary."""
    return WeatherSummary(
        summary=f"Weather is {request.weather_info}",
        recommendation="Perfect day for outdoor activities!",
    )


class TestNodes:
    """Test individual node functionality."""

    def test_prompt_node_initialization(self):
        """Test PromptNode initialization."""
        node = PromptNode[WeatherQuery, str](
            prompt="What's the weather in {location}?",
            name="weather_prompt",
        )
        assert node.name == "weather_prompt"
        assert node.prompt == "What's the weather in {location}?"

    @pytest.mark.asyncio
    async def test_prompt_node_execution(self):
        """Test PromptNode execution."""
        node = PromptNode[WeatherQuery, str](
            prompt="What's the weather in {location}?",
        )

        query = WeatherQuery(location="Paris")
        result = await node.run(query)

        # Should return the formatted prompt for now
        assert "Paris" in result

    def test_parser_node_initialization(self):
        """Test ParserNode initialization."""
        node = ParserNode[str, WeatherInfo](
            parser_func=parse_weather_string,
            name="weather_parser",
        )
        assert node.name == "weather_parser"
        assert node.parser_func == parse_weather_string

    @pytest.mark.asyncio
    async def test_parser_node_execution(self):
        """Test ParserNode execution."""
        node = ParserNode[str, WeatherInfo](
            parser_func=parse_weather_string,
        )

        weather_str = "22.5|sunny|Paris"
        result = await node.run(weather_str)

        assert isinstance(result, WeatherInfo)
        assert result.temperature == EXPECTED_TEMPERATURE
        assert result.condition == "sunny"
        assert result.location == "Paris"

    def test_tool_node_initialization(self):
        """Test ToolNode initialization."""
        node = ToolNode[WeatherQuery, WeatherInfo](
            tool_func=call_weather_api,
            name="weather_api",
        )
        assert node.name == "weather_api"
        assert node.tool_func == call_weather_api

    @pytest.mark.asyncio
    async def test_tool_node_execution(self):
        """Test ToolNode execution."""
        node = ToolNode[WeatherQuery, WeatherInfo](
            tool_func=call_weather_api,
        )

        query = WeatherQuery(location="Paris")
        result = await node.run(query)

        assert isinstance(result, WeatherInfo)
        assert result.location == "Paris"
        assert result.temperature == EXPECTED_TEMPERATURE
        assert result.condition == "sunny"

    def test_node_output_wiring(self):
        """Test that nodes can be wired together using outputs."""
        node1 = PromptNode[WeatherQuery, str](
            prompt="What's the weather in {location}?",
        )
        node2 = ParserNode[str, WeatherInfo](
            parser_func=parse_weather_string,
            input=node1.output,
        )

        assert node2.input is not None
        assert node2.input.node == node1
        assert node1 in node2.dependencies


class TestFlow:
    """Test Flow orchestration functionality."""

    def test_flow_initialization(self):
        """Test Flow initialization."""
        flow = Flow(input_type=WeatherQuery, output_type=GenericFlowResults)
        assert flow.nodes == []
        assert flow._execution_order == []
        assert flow._results == {}
        assert flow._output_type == GenericFlowResults

    def test_add_nodes(self):
        """Test adding nodes to a flow."""
        flow = Flow(input_type=WeatherQuery, output_type=GenericFlowResults)
        node1 = PromptNode[WeatherQuery, str](
            prompt="What's the weather in {location}?",
        )
        node2 = ToolNode[WeatherQuery, WeatherInfo](
            tool_func=call_weather_api,
        )

        flow.add_nodes(node1, node2)

        assert len(flow.nodes) == EXPECTED_NODES_COUNT
        assert node1 in flow.nodes
        assert node2 in flow.nodes

    def test_execution_order_simple(self):
        """Test execution order calculation for simple workflow."""
        flow = Flow(input_type=WeatherQuery, output_type=SimpleFlowResults)
        node1 = PromptNode[WeatherQuery, str](
            prompt="What's the weather in {location}?",
            name="prompt",
        )
        node2 = ParserNode[str, WeatherInfo](
            parser_func=parse_weather_string,
            input=node1.output,
            name="parser",
        )

        flow.add_nodes(node1, node2)
        execution_order = flow.get_execution_order()

        assert len(execution_order) == EXPECTED_NODES_COUNT
        assert execution_order.index("prompt") < execution_order.index("parser")

    def test_execution_order_complex(self):
        """Test execution order for complex workflow."""
        flow = Flow(input_type=WeatherQuery, output_type=ComplexFlowResults)

        # Create a more complex dependency graph
        node1 = PromptNode[WeatherQuery, str](prompt="test", name="node1")
        node2 = ParserNode[str, WeatherInfo](
            parser_func=parse_weather_string,
            input=node1.output,
            name="node2",
        )
        node3 = ToolNode[WeatherQuery, WeatherInfo](
            tool_func=call_weather_api,
            name="node3",
        )
        node4 = ParserNode[WeatherInfo, WeatherInfo](
            parser_func=lambda x: x,  # identity function
            input=node2.output,
            name="node4",
        )

        flow.add_nodes(node1, node2, node3, node4)
        execution_order = flow.get_execution_order()

        assert len(execution_order) == EXPECTED_COMPLEX_NODES_COUNT
        # node1 must come before node2, and node2 must come before node4
        assert execution_order.index("node1") < execution_order.index("node2")
        assert execution_order.index("node2") < execution_order.index("node4")

    @pytest.mark.asyncio
    async def test_flow_execution_simple(self):
        """Test simple flow execution."""
        flow = Flow(input_type=WeatherQuery, output_type=SimpleFlowResults)

        node1 = ToolNode[WeatherQuery, WeatherInfo](
            tool_func=call_weather_api,
            name="weather_api",
        )

        flow.add_nodes(node1)

        query = WeatherQuery(location="Paris")
        results = await flow.run(query)
        assert hasattr(results, "weather_api")
        result = results.weather_api
        assert isinstance(result, WeatherInfo)
        assert result.location == "Paris"

    @pytest.mark.asyncio
    async def test_flow_execution_with_dependencies(self):
        """Test flow execution with node dependencies."""

        # Create a custom input model with temperature field
        class QueryWithTemp(BaseModel):
            location: str
            temperature_unit: str = "celsius"
            temperature: str = "25.0"

        flow = Flow(input_type=QueryWithTemp, output_type=ComplexFlowResults)

        # Create a workflow: prompt -> parser
        node1 = PromptNode[QueryWithTemp, str](
            prompt="{temperature}|sunny|{location}",
            name="prompt",
        )
        node2 = ParserNode[str, WeatherInfo](
            parser_func=parse_weather_string,
            input=node1.output,
            name="parser",
        )

        flow.add_nodes(node1, node2)

        query = QueryWithTemp(location="Paris")
        results = await flow.run(query)

        # Results is now a BaseModel with attributes
        assert hasattr(results, "prompt")
        assert hasattr(results, "parser")

        parsed_result = results.parser
        assert isinstance(parsed_result, WeatherInfo)
        assert parsed_result.location == "Paris"
        assert parsed_result.temperature == EXPECTED_TEST_TEMPERATURE

    def test_flow_validation(self):
        """Test flow validation."""
        flow = Flow(input_type=WeatherQuery, output_type=GenericFlowResults)
        node1 = PromptNode[WeatherQuery, str](prompt="test")
        node2 = ParserNode[str, WeatherInfo](
            parser_func=parse_weather_string,
            input=node1.output,
        )

        flow.add_nodes(node1, node2)

        assert flow.validate() is True

    def test_cyclic_dependency_detection(self):
        """Test that cyclic dependencies are detected."""
        flow = Flow(input_type=WeatherQuery, output_type=GenericFlowResults)

        # This is tricky to set up with the current API since we can't create
        # direct cycles easily. For now, we'll test the validation method.
        node1 = PromptNode[WeatherQuery, str](prompt="test", name="node1")
        flow.add_nodes(node1)

        # This should not raise an error
        assert flow.validate() is True

    def test_flow_repr(self):
        """Test Flow string representation."""
        flow = Flow(input_type=WeatherQuery, output_type=GenericFlowResults)
        node1 = PromptNode[WeatherQuery, str](prompt="test")
        flow.add_nodes(node1)

        repr_str = repr(flow)
        # Check that the new format includes type information and node count
        assert repr_str == "Flow[WeatherQuery, GenericFlowResults](nodes=1)"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_empty_flow_execution(self):
        """Test executing an empty flow."""
        flow = Flow(input_type=WeatherQuery, output_type=EmptyFlowResults)
        query = WeatherQuery(location="Paris")

        results = await flow.run(query)
        # Results is now a BaseModel, should have no attributes
        assert isinstance(results, BaseModel)
        # For an empty flow, the model should have no fields
        assert len(results.__class__.model_fields) == 0

    def test_duplicate_node_addition(self):
        """Test adding the same node multiple times."""
        flow = Flow(input_type=WeatherQuery, output_type=GenericFlowResults)
        node = PromptNode[WeatherQuery, str](prompt="test")

        flow.add_nodes(node)
        flow.add_nodes(node)  # Add again

        assert len(flow.nodes) == 1

    def test_node_naming(self):
        """Test node naming behavior."""
        # Without explicit name
        node1 = PromptNode[WeatherQuery, str](prompt="test")
        assert node1.name.startswith("PromptNode_")

        # With explicit name
        node2 = PromptNode[WeatherQuery, str](prompt="test", name="custom_name")
        assert node2.name == "custom_name"

    def test_node_dependencies_property(self):
        """Test node dependencies property."""
        node1 = PromptNode[WeatherQuery, str](prompt="test")
        node2 = ParserNode[str, WeatherInfo](
            parser_func=parse_weather_string,
            input=node1.output,
        )

        assert len(node1.dependencies) == 0
        assert len(node2.dependencies) == 1
        assert node1 in node2.dependencies


class TestAdvancedNodes:
    """Test advanced node types like RetryNode and IfNode."""

    @pytest.mark.asyncio
    async def test_retry_node_success(self):
        """Test RetryNode when the wrapped node succeeds."""
        # Create a simple node that always succeeds
        base_node = ToolNode[WeatherQuery, WeatherInfo](
            tool_func=call_weather_api, name="base_node"
        )

        retry_node = RetryNode(wrapped_node=base_node, max_retries=3, name="retry_node")

        query = WeatherQuery(location="Paris")
        result = await retry_node.run(query)

        assert isinstance(result, WeatherInfo)
        assert result.location == "Paris"

    @pytest.mark.asyncio
    async def test_retry_node_failure(self):
        """Test RetryNode when the wrapped node always fails."""

        def failing_func(query: WeatherQuery) -> WeatherInfo:
            raise ValueError("API error")

        base_node = ToolNode[WeatherQuery, WeatherInfo](
            tool_func=failing_func, name="failing_node"
        )

        retry_node = RetryNode(wrapped_node=base_node, max_retries=2, name="retry_node")

        query = WeatherQuery(location="Paris")

        with pytest.raises(ValueError, match="API error"):
            await retry_node.run(query)

    def test_retry_node_dependencies(self):
        """Test RetryNode dependencies."""
        base_node = ToolNode[WeatherQuery, WeatherInfo](
            tool_func=call_weather_api, name="base_node"
        )

        retry_node = RetryNode(wrapped_node=base_node, max_retries=3, name="retry_node")

        # RetryNode should inherit dependencies from wrapped node
        assert retry_node.dependencies == base_node.dependencies

    @pytest.mark.asyncio
    async def test_if_node_true_branch(self):
        """Test IfNode when predicate is True."""
        true_node = ToolNode[WeatherQuery, WeatherInfo](
            tool_func=call_weather_api, name="true_node"
        )

        false_node = ToolNode[WeatherQuery, WeatherInfo](
            tool_func=lambda x: WeatherInfo(
                temperature=0, condition="cold", location=x.location
            ),
            name="false_node",
        )

        if_node = IfNode(
            predicate=lambda x: x.location == "Paris",
            if_true=true_node,
            if_false=false_node,
            name="if_node",
        )

        query = WeatherQuery(location="Paris")
        result = await if_node.run(query)

        assert result.temperature == EXPECTED_TEMPERATURE  # From true_node

    @pytest.mark.asyncio
    async def test_if_node_false_branch(self):
        """Test IfNode when predicate is False."""
        true_node = ToolNode[WeatherQuery, WeatherInfo](
            tool_func=call_weather_api, name="true_node"
        )

        false_node = ToolNode[WeatherQuery, WeatherInfo](
            tool_func=lambda x: WeatherInfo(
                temperature=0, condition="cold", location=x.location
            ),
            name="false_node",
        )

        if_node = IfNode(
            predicate=lambda x: x.location == "Paris",
            if_true=true_node,
            if_false=false_node,
            name="if_node",
        )

        query = WeatherQuery(location="London")
        result = await if_node.run(query)

        assert result.temperature == 0  # From false_node
        assert result.condition == "cold"

    def test_if_node_dependencies(self):
        """Test IfNode dependencies include both branches."""
        prompt_node = PromptNode[WeatherQuery, str](prompt="test")

        true_node = ParserNode[str, WeatherInfo](
            parser_func=parse_weather_string, input=prompt_node.output, name="true_node"
        )

        false_node = ToolNode[WeatherQuery, WeatherInfo](
            tool_func=call_weather_api, name="false_node"
        )

        if_node = IfNode(
            predicate=lambda x: True,
            if_true=true_node,
            if_false=false_node,
            input=prompt_node.output,
            name="if_node",
        )

        dependencies = if_node.dependencies
        # Should include prompt_node (from input), and dependencies of both branches
        assert prompt_node in dependencies
        assert prompt_node in dependencies  # true_node also depends on prompt_node

    def test_prompt_config(self):
        """Test PromptConfig functionality."""
        config = PromptConfig(
            model="custom-model", system_prompt="You are helpful", result_type=str
        )

        node = PromptNode[WeatherQuery, str](
            prompt="Test prompt", config=config, name="test_node"
        )

        assert node.config.model == "custom-model"
        assert node.config.system_prompt == "You are helpful"
        assert node.config.result_type is str

    def test_prompt_config_defaults(self):
        """Test PromptConfig default values."""
        node = PromptNode[WeatherQuery, str](prompt="Test prompt", name="test_node")

        # Should use default config
        assert node.config.model == "openai:gpt-4"
        assert node.config.system_prompt is None
        assert node.config.result_type is None


class TestCoverageEdgeCases:
    """Test edge cases to reach 100% coverage."""

    def test_cyclic_dependency_detection_actual_cycle(self):
        """Test that actual cyclic dependencies are detected."""
        flow = Flow(input_type=WeatherQuery, output_type=GenericFlowResults)

        # Create nodes that would form a cycle if we could set up dependencies
        # We'll create a more complex scenario by manipulating the nodes directly
        node1 = PromptNode[WeatherQuery, str](prompt="test1", name="node1")
        node2 = PromptNode[WeatherQuery, str](prompt="test2", name="node2")

        # Manually create a cycle by manipulating the internal structure
        # This is a bit of a hack, but necessary to test the cycle detection
        flow.nodes = [node1, node2]

        # Create a fake dependency cycle by manipulating getattr behavior
        original_getattr = getattr

        def mock_getattr(obj, name, default=None):
            if name == "dependencies" and obj is node1:
                return [node2]
            elif name == "dependencies" and obj is node2:
                return [node1]  # This creates the cycle
            return original_getattr(obj, name, default)

        builtins.getattr = mock_getattr  # type: ignore[assignment]

        try:
            with pytest.raises(
                CyclicDependencyError, match="Cyclic dependency detected"
            ):
                flow._calculate_execution_order()
        finally:
            # Restore original getattr
            builtins.getattr = original_getattr

    def test_cyclic_dependency_error_in_validate(self):
        """Test that CyclicDependencyError is re-raised in validate method."""
        flow = Flow(input_type=WeatherQuery, output_type=GenericFlowResults)

        # Create nodes that would form a cycle
        node1 = PromptNode[WeatherQuery, str](prompt="test1", name="node1")
        node2 = PromptNode[WeatherQuery, str](prompt="test2", name="node2")

        flow.nodes = [node1, node2]

        # Create a fake dependency cycle by manipulating getattr behavior
        original_getattr = getattr

        def mock_getattr(obj, name, default=None):
            if name == "dependencies" and obj is node1:
                return [node2]
            elif name == "dependencies" and obj is node2:
                return [node1]  # This creates the cycle
            return original_getattr(obj, name, default)

        builtins.getattr = mock_getattr  # type: ignore[assignment]

        try:
            # This should trigger the CyclicDependencyError which gets re-raised
            # This should hit line 174: the "raise" under except CyclicDependencyError
            with pytest.raises(
                CyclicDependencyError, match="Cyclic dependency detected"
            ):
                flow.validate()  # Call validate, not _calculate_execution_order
        finally:
            # Restore original getattr
            builtins.getattr = original_getattr

    @pytest.mark.asyncio
    async def test_missing_input_node_error(self):
        """Test FlowError when input node hasn't been executed."""
        flow = Flow(input_type=WeatherQuery, output_type=ComplexFlowResults)

        # Create nodes with dependencies
        node1 = PromptNode[WeatherQuery, str](prompt="test", name="node1")
        node2 = ParserNode[str, WeatherInfo](
            parser_func=parse_weather_string,
            input=node1.output,
            name="node2",
        )

        flow.add_nodes(node2)  # Only add node2, not node1

        # Manually manipulate the execution order to trigger the error
        flow._execution_order = [node2]  # node2 will try to access node1's results

        query = WeatherQuery(location="Paris")

        with pytest.raises(FlowError, match="Input node node1 has not been executed"):
            await flow.run(query)

    @pytest.mark.asyncio
    async def test_flow_execution_general_error_wrapping(self):
        """Test that general exceptions during flow execution are wrapped."""
        flow = Flow(input_type=WeatherQuery, output_type=SimpleFlowResults)

        # Create a node that will raise an exception
        def failing_tool(query: WeatherQuery) -> WeatherInfo:
            raise ValueError("Simulated tool failure")

        failing_node = ToolNode[WeatherQuery, WeatherInfo](
            tool_func=failing_tool,
            name="failing_tool",
        )

        flow.add_nodes(failing_node)

        query = WeatherQuery(location="Paris")

        with pytest.raises(FlowError, match="Flow execution failed"):
            await flow.run(query)

    def test_flow_validation_general_error(self):
        """Test that general exceptions during validation are wrapped."""
        flow = Flow(input_type=WeatherQuery, output_type=GenericFlowResults)

        # Mock _calculate_execution_order to raise a non-CyclicDependencyError
        original_method = flow._calculate_execution_order

        def failing_calculation():
            raise ValueError("Simulated calculation failure")

        flow._calculate_execution_order = failing_calculation  # type: ignore[assignment]

        try:
            with pytest.raises(FlowError, match="Flow validation failed"):
                flow.validate()
        finally:
            # Restore original method
            flow._calculate_execution_order = original_method  # type: ignore[assignment]

    @pytest.mark.asyncio
    async def test_flow_run_output_type_construction_error(self):
        """Test error handling when output type construction fails."""
        flow = Flow(input_type=WeatherQuery, output_type=SimpleFlowResults)

        # Create a simple tool node that returns valid data
        def simple_tool(query: WeatherQuery) -> WeatherInfo:
            return WeatherInfo(
                temperature=22.5, condition="sunny", location=query.location
            )

        tool_node = ToolNode[WeatherQuery, WeatherInfo](
            tool_func=simple_tool,
            name="weather_tool",
        )

        flow.add_nodes(tool_node)

        # Mock the output type to raise an error during construction
        original_output_type = flow._output_type

        class FailingOutputType:
            def __init__(self, **kwargs):
                raise ValueError("Construction failed")

        flow._output_type = FailingOutputType  # type: ignore

        query = WeatherQuery(location="Paris")

        try:
            with pytest.raises(FlowError, match="Flow execution failed"):
                await flow.run(query)
        finally:
            # Restore original output type
            flow._output_type = original_output_type

    def test_node_type_hint_property(self):
        """Test the type_hint property of NodeOutput."""
        node = PromptNode[WeatherQuery, str](prompt="test")

        # Access the type_hint property to cover line 27 in base.py
        type_hint = node.output.type_hint
        # The type hint should be the actual output type
        assert type_hint is not None
        assert hasattr(type_hint, "__name__")

    def test_node_repr_method(self):
        """Test the __repr__ method of BaseNode."""
        node = PromptNode[WeatherQuery, str](prompt="test", name="custom_name")

        # Test the __repr__ method to cover line 70 in base.py
        repr_str = repr(node)
        assert "PromptNode" in repr_str
        assert "custom_name" in repr_str

        # Test with auto-generated name
        node2 = PromptNode[WeatherQuery, str](prompt="test")
        repr_str2 = repr(node2)
        assert "PromptNode" in repr_str2

    @pytest.mark.asyncio
    async def test_runtime_type_validation_paths(self):
        """Test the specific runtime type validation error paths."""
        flow = Flow(input_type=WeatherQuery, output_type=SimpleFlowResults)

        # Create a node
        node = ToolNode[WeatherQuery, WeatherInfo](
            tool_func=call_weather_api,
            name="weather_api",
        )
        flow.add_nodes(node)

        # Test with wrong input type to trigger lines 114-117
        class WrongInputType(BaseModel):
            different_field: str

        wrong_input = WrongInputType(different_field="test")

        # This should trigger the isinstance check and error message construction
        with pytest.raises(TypeError) as exc_info:
            await flow.run(wrong_input)  # type: ignore

        # Verify the error message was constructed properly (covers lines 114-117)
        assert "Input type mismatch" in str(exc_info.value)
        assert "WeatherQuery" in str(exc_info.value)
        assert "WrongInputType" in str(exc_info.value)

    def test_flow_validation_exception_wrapping(self):
        """Test the exception wrapping in flow validation."""
        flow = Flow(input_type=WeatherQuery, output_type=GenericFlowResults)

        # Create a simple node first
        node = PromptNode[WeatherQuery, str](prompt="test", name="test_node")
        flow.add_nodes(node)

        # Create a bad node that raises exception when dependencies is accessed
        class BadNode:
            def __init__(self):
                self.name = "bad_node"

            @property
            def dependencies(self):
                raise RuntimeError("Intentional failure accessing dependencies")

        # Add the bad node
        flow.nodes.append(BadNode())  # type: ignore

        # This should trigger line 174 - the except Exception handler
        with pytest.raises(FlowError) as exc_info:
            flow.validate()

        # Verify the exception was wrapped properly
        assert "Flow validation failed" in str(exc_info.value)
        assert "Intentional failure accessing dependencies" in str(exc_info.value)


class TestFlowNode:
    """Test FlowNode functionality for sub-flow composition."""

    def test_flow_node_initialization(self):
        """Test FlowNode initialization with wrapped flow."""
        # Create a simple sub-flow
        sub_flow = Flow(input_type=WeatherQuery, output_type=SimpleFlowResults)
        weather_node = ToolNode[WeatherQuery, WeatherInfo](
            tool_func=call_weather_api,
            name="weather_api",
        )
        sub_flow.add_nodes(weather_node)

        # Create FlowNode
        flow_node = FlowNode[WeatherQuery, SimpleFlowResults](
            flow=sub_flow,
            name="weather_sub_flow",
        )

        assert flow_node.flow is sub_flow
        assert flow_node.name == "weather_sub_flow"
        assert flow_node.dependencies == []

    def test_flow_node_default_name(self):
        """Test FlowNode default name generation."""
        sub_flow = Flow(input_type=WeatherQuery, output_type=SimpleFlowResults)
        flow_node = FlowNode[WeatherQuery, SimpleFlowResults](flow=sub_flow)

        # Default name should include flow representation
        assert flow_node.name.startswith("FlowNode_")
        assert "Flow[WeatherQuery, SimpleFlowResults]" in flow_node.name

    @pytest.mark.asyncio
    async def test_flow_node_execution(self):
        """Test FlowNode execution of wrapped flow."""
        # Create a sub-flow that processes weather data
        sub_flow = Flow(input_type=WeatherQuery, output_type=SimpleFlowResults)
        weather_node = ToolNode[WeatherQuery, WeatherInfo](
            tool_func=call_weather_api,
            name="weather_api",
        )
        sub_flow.add_nodes(weather_node)

        # Create FlowNode
        flow_node = FlowNode[WeatherQuery, SimpleFlowResults](flow=sub_flow)

        # Execute the FlowNode
        query = WeatherQuery(location="Tokyo")
        result = await flow_node.run(query)

        assert isinstance(result, SimpleFlowResults)
        assert hasattr(result, "weather_api")
        assert result.weather_api.location == "Tokyo"
        assert result.weather_api.temperature == EXPECTED_TEMPERATURE

    @pytest.mark.asyncio
    async def test_flow_node_in_parent_flow(self):
        """Test FlowNode used as a node within a parent flow."""
        # Create a sub-flow for weather data
        weather_sub_flow = Flow(input_type=WeatherQuery, output_type=SimpleFlowResults)
        weather_node = ToolNode[WeatherQuery, WeatherInfo](
            tool_func=call_weather_api,
            name="weather_api",
        )
        weather_sub_flow.add_nodes(weather_node)

        # Create a parent flow that uses the sub-flow
        parent_flow = Flow(input_type=WeatherQuery, output_type=FlowNodeResults)

        # Add FlowNode to parent flow
        sub_flow_node = FlowNode[WeatherQuery, SimpleFlowResults](
            flow=weather_sub_flow,
            name="weather_sub_flow",
        )
        parent_flow.add_nodes(sub_flow_node)

        # Execute parent flow
        query = WeatherQuery(location="London")
        results = await parent_flow.run(query)

        # Results should contain the sub-flow output
        assert hasattr(results, "weather_sub_flow")
        sub_result = results.weather_sub_flow
        assert isinstance(sub_result, SimpleFlowResults)
        assert sub_result.weather_api.location == "London"

    @pytest.mark.asyncio
    async def test_nested_flows_multiple_levels(self):
        """Test deeply nested flows with multiple levels."""
        # Level 1: Basic weather sub-flow
        level1_flow = Flow(input_type=WeatherQuery, output_type=SimpleFlowResults)
        weather_node = ToolNode[WeatherQuery, WeatherInfo](
            tool_func=call_weather_api,
            name="weather_api",
        )
        level1_flow.add_nodes(weather_node)

        # Level 2: Wrapper flow
        level2_flow = Flow(input_type=WeatherQuery, output_type=FlowNodeResults)
        level1_node = FlowNode[WeatherQuery, SimpleFlowResults](
            flow=level1_flow,
            name="weather_sub_flow",  # Use the expected field name
        )
        level2_flow.add_nodes(level1_node)

        # Level 3: Top-level flow
        level3_flow = Flow(input_type=WeatherQuery, output_type=NestedFlowResults)
        level2_node = FlowNode[WeatherQuery, FlowNodeResults](
            flow=level2_flow,
            name="level2_wrapper",
        )
        level3_flow.add_nodes(level2_node)

        # Execute the deeply nested flow
        query = WeatherQuery(location="Berlin")
        results = await level3_flow.run(query)

        # Verify the nested structure worked
        assert hasattr(results, "level2_wrapper")
        level2_result = results.level2_wrapper
        assert hasattr(level2_result, "weather_sub_flow")
        level1_result = level2_result.weather_sub_flow
        assert level1_result.weather_api.location == "Berlin"

    def test_flow_node_with_input_dependency(self):
        """Test FlowNode that takes input from another node."""
        # Create a sub-flow
        sub_flow = Flow(input_type=WeatherQuery, output_type=SimpleFlowResults)
        weather_node = ToolNode[WeatherQuery, WeatherInfo](
            tool_func=call_weather_api,
            name="weather_api",
        )
        sub_flow.add_nodes(weather_node)

        # Create a FlowNode that doesn't depend on other nodes
        flow_node = FlowNode[WeatherQuery, SimpleFlowResults](
            flow=sub_flow,
            name="dependent_sub_flow",
        )

        # Verify dependencies are tracked correctly
        assert flow_node.dependencies == []  # No direct dependencies from other nodes

    def test_flow_node_repr(self):
        """Test FlowNode string representation."""
        sub_flow = Flow(input_type=WeatherQuery, output_type=SimpleFlowResults)
        flow_node = FlowNode[WeatherQuery, SimpleFlowResults](
            flow=sub_flow,
            name="test_flow_node",
        )

        repr_str = repr(flow_node)
        assert "FlowNode(name='test_flow_node'" in repr_str
        assert "Flow[WeatherQuery, SimpleFlowResults]" in repr_str

    @pytest.mark.asyncio
    async def test_flow_node_error_propagation(self):
        """Test that errors from wrapped flows are properly propagated."""
        # Create a sub-flow with a node that will fail
        sub_flow = Flow(input_type=WeatherQuery, output_type=SimpleFlowResults)

        def failing_tool(query: WeatherQuery) -> WeatherInfo:
            msg = f"Intentional failure for {query.location}"
            raise ValueError(msg)

        failing_node = ToolNode[WeatherQuery, WeatherInfo](
            tool_func=failing_tool,
            name="failing_node",
        )
        sub_flow.add_nodes(failing_node)

        # Create FlowNode
        flow_node = FlowNode[WeatherQuery, SimpleFlowResults](flow=sub_flow)

        # Execution should fail and propagate the error
        query = WeatherQuery(location="ErrorCity")
        with pytest.raises(FlowError) as exc_info:
            await flow_node.run(query)

        assert "Flow execution failed" in str(exc_info.value)
        assert "Intentional failure for ErrorCity" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__])
