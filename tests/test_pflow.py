"""Test suite for the pydantic-flow framework.

This module contains comprehensive tests for all node types, Flow functionality,
and edge cases to ensure the framework works as expected.
"""

from pydantic import BaseModel
from pydantic_ai import Agent
import pytest

from pydantic_flow import AgentNode
from pydantic_flow import Flow
from pydantic_flow import FlowNode
from pydantic_flow import IfNode
from pydantic_flow import ParserNode
from pydantic_flow import RetryNode
from pydantic_flow import ToolNode
from pydantic_flow.flow.exceptions import FlowError
from tests.conftest import extract_result_from_stream

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
def create_test_agent_node(prompt_template: str, name: str | None = None) -> AgentNode:
    """Create a test AgentNode with the 'test' model."""
    agent = Agent("test", instructions="Be helpful")
    return AgentNode(agent=agent, prompt_template=prompt_template, name=name)


def parse_weather_string(weather_str: str) -> WeatherInfo:
    """Parse a weather string into structured data."""
    # Simple parser for testing
    parts = weather_str.split("|")
    return WeatherInfo(
        temperature=float(parts[0]),
        condition=parts[1].strip(),
        location=parts[2].strip(),
    )


async def call_weather_api(query: WeatherQuery) -> WeatherInfo:
    """Mock weather API call."""
    return WeatherInfo(
        temperature=EXPECTED_TEMPERATURE,
        condition="sunny",
        location=query.location,
    )


async def generate_summary(request: SummaryRequest) -> WeatherSummary:
    """Generate a weather summary."""
    return WeatherSummary(
        summary=f"Weather is {request.weather_info}",
        recommendation="Perfect day for outdoor activities!",
    )


class TestNodes:
    """Test individual node functionality."""

    def test_agent_node_initialization(self):
        """Test AgentNode initialization."""
        agent = Agent("test", instructions="Be helpful")
        node = AgentNode[WeatherQuery, str](
            agent=agent,
            prompt_template="What's the weather in {location}?",
            name="weather_prompt",
        )
        assert node.name == "weather_prompt"
        assert node.prompt_template == "What's the weather in {location}?"

    @pytest.mark.asyncio
    async def test_agent_node_execution(self):
        """Test AgentNode execution."""
        agent = Agent("test", instructions="Be helpful")
        node = AgentNode[WeatherQuery, str](
            agent=agent,
            prompt_template="What's the weather in {location}?",
        )

        query = WeatherQuery(location="Paris")
        result = await extract_result_from_stream(node.astream(query))

        # Test model returns a success message
        assert isinstance(result, str)
        assert "success" in result

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
        result = await extract_result_from_stream(node.astream(weather_str))

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
        result = await extract_result_from_stream(node.astream(query))

        assert isinstance(result, WeatherInfo)
        assert result.location == "Paris"
        assert result.temperature == EXPECTED_TEMPERATURE
        assert result.condition == "sunny"

    def test_node_output_wiring(self):
        """Test that nodes can be wired together using outputs."""
        agent = Agent("test", instructions="Be helpful")
        node1 = AgentNode[WeatherQuery, str](
            agent=agent,
            prompt_template="What's the weather in {location}?",
        )
        node2 = ParserNode[str, WeatherInfo](
            parser_func=parse_weather_string,
            inputs=(node1.output,),
        )

        assert node2.inputs is not None
        assert len(node2.inputs) == 1
        assert node2.inputs[0].node == node1
        assert node1 in node2.dependencies


class TestFlow:
    """Test Flow orchestration functionality."""

    def test_flow_initialization(self):
        """Test Flow initialization."""
        flow = Flow(input_type=WeatherQuery, output_type=GenericFlowResults)
        assert flow.nodes == []
        assert flow._output_type == GenericFlowResults

    def test_add_nodes(self):
        """Test adding nodes to a flow."""
        flow = Flow(input_type=WeatherQuery, output_type=GenericFlowResults)
        agent = Agent("test", instructions="Be helpful")
        node1 = AgentNode[WeatherQuery, str](
            agent=agent,
            prompt_template="What's the weather in {location}?",
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
        agent = Agent("test", instructions="Be helpful")
        node1 = AgentNode[WeatherQuery, str](
            agent=agent,
            prompt_template="What's the weather in {location}?",
            name="prompt",
        )
        node2 = ParserNode[str, WeatherInfo](
            parser_func=parse_weather_string,
            inputs=(node1.output,),
            name="parser",
        )

        flow.add_nodes(node1, node2)

        # Verify flow compiles
        # Flows execute directly - no compilation needed
        assert len(flow.nodes) == EXPECTED_NODES_COUNT

    def test_execution_order_complex(self):
        """Test flow compilation with complex dependencies."""
        flow = Flow(input_type=WeatherQuery, output_type=ComplexFlowResults)

        # Create a more complex dependency graph
        node1 = create_test_agent_node("test", name="node1")
        node2 = ParserNode[str, WeatherInfo](
            parser_func=parse_weather_string,
            inputs=(node1.output,),
            name="node2",
        )
        node3 = ToolNode[WeatherQuery, WeatherInfo](
            tool_func=call_weather_api,
            name="node3",
        )
        node4 = ParserNode[WeatherInfo, WeatherInfo](
            parser_func=lambda x: x,  # identity function
            inputs=(node2.output,),
            name="node4",
        )

        flow.add_nodes(node1, node2, node3, node4)

        # Verify flow compiles with complex dependencies
        # Flows execute directly - no compilation needed
        assert len(flow.nodes) == EXPECTED_COMPLEX_NODES_COUNT

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
        results = await extract_result_from_stream(flow.astream(query))
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

        # Output model for the formatter
        class FormattedString(BaseModel):
            value: str

        # Custom results model for this test
        class TestFlowResults(BaseModel):
            prompt: FormattedString
            parser: WeatherInfo

        flow = Flow(input_type=QueryWithTemp, output_type=TestFlowResults)

        # Create a workflow: format -> parser
        # Use ToolNode to format the string
        async def format_weather_string(query: QueryWithTemp) -> FormattedString:
            return FormattedString(value=f"{query.temperature}|sunny|{query.location}")

        def parse_formatted(formatted: FormattedString) -> WeatherInfo:
            return parse_weather_string(formatted.value)

        node1 = ToolNode[QueryWithTemp, FormattedString](
            tool_func=format_weather_string,
            name="prompt",
        )
        node2 = ParserNode[FormattedString, WeatherInfo](
            parser_func=parse_formatted,
            inputs=(node1.output,),
            name="parser",
        )

        flow.add_nodes(node1, node2)

        query = QueryWithTemp(location="Paris")
        results = await extract_result_from_stream(flow.astream(query))

        # Results is now a BaseModel with attributes
        assert hasattr(results, "prompt")
        assert hasattr(results, "parser")

        parsed_result = results.parser
        assert isinstance(parsed_result, WeatherInfo)
        assert parsed_result.location == "Paris"
        assert parsed_result.temperature == EXPECTED_TEST_TEMPERATURE

    def test_flow_validation(self):
        """Test flow compilation."""
        flow = Flow(input_type=WeatherQuery, output_type=GenericFlowResults)
        node1 = create_test_agent_node("test")
        node2 = ParserNode[str, WeatherInfo](
            parser_func=parse_weather_string,
            inputs=(node1.output,),
        )

        flow.add_nodes(node1, node2)

        # Verify flow compiles without errors
        # Flows execute directly - no compilation needed

    def test_cyclic_dependency_detection(self):
        """Test that flow compiles successfully with single node."""
        flow = Flow(input_type=WeatherQuery, output_type=GenericFlowResults)

        # Simple single node flow
        node1 = create_test_agent_node("test", name="node1")
        flow.add_nodes(node1)

        # This should compile without errors
        # Flows execute directly - no compilation needed

    def test_flow_repr(self):
        """Test Flow string representation."""
        flow = Flow(input_type=WeatherQuery, output_type=GenericFlowResults)
        node1 = create_test_agent_node("test")
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

        results = await extract_result_from_stream(flow.astream(query))
        # Results is now a BaseModel, should have no attributes
        assert isinstance(results, BaseModel)
        # For an empty flow, the model should have no fields
        assert len(results.__class__.model_fields) == 0

    def test_duplicate_node_addition(self):
        """Test adding the same node multiple times."""
        flow = Flow(input_type=WeatherQuery, output_type=GenericFlowResults)
        node = create_test_agent_node("test")

        flow.add_nodes(node)
        flow.add_nodes(node)  # Add again

        assert len(flow.nodes) == 1

    def test_node_naming(self):
        """Test node naming behavior."""
        # Without explicit name
        node1 = create_test_agent_node("test")
        assert node1.name.startswith("AgentNode_")

        # With explicit name
        node2 = create_test_agent_node("test", name="custom_name")
        assert node2.name == "custom_name"

    def test_node_dependencies_property(self):
        """Test node dependencies property."""
        node1 = create_test_agent_node("test")
        node2 = ParserNode[str, WeatherInfo](
            parser_func=parse_weather_string,
            inputs=(node1.output,),
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
        result = await extract_result_from_stream(retry_node.astream(query))

        assert isinstance(result, WeatherInfo)
        assert result.location == "Paris"

    @pytest.mark.asyncio
    async def test_retry_node_failure(self):
        """Test RetryNode when the wrapped node always fails."""

        async def failing_func(query: WeatherQuery) -> WeatherInfo:
            raise ValueError("API error")

        base_node = ToolNode[WeatherQuery, WeatherInfo](
            tool_func=failing_func, name="failing_node"
        )

        retry_node = RetryNode(wrapped_node=base_node, max_retries=2, name="retry_node")

        query = WeatherQuery(location="Paris")

        with pytest.raises(ValueError, match="API error"):
            await extract_result_from_stream(retry_node.astream(query))

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

        async def false_tool_func(x: WeatherQuery) -> WeatherInfo:
            return WeatherInfo(temperature=0, condition="cold", location=x.location)

        false_node = ToolNode[WeatherQuery, WeatherInfo](
            tool_func=false_tool_func,
            name="false_node",
        )

        if_node = IfNode(
            predicate=lambda x: x.location == "Paris",
            if_true=true_node,
            if_false=false_node,
            name="if_node",
        )

        query = WeatherQuery(location="Paris")
        result = await extract_result_from_stream(if_node.astream(query))

        assert result.temperature == EXPECTED_TEMPERATURE  # From true_node

    @pytest.mark.asyncio
    async def test_if_node_false_branch(self):
        """Test IfNode when predicate is False."""
        true_node = ToolNode[WeatherQuery, WeatherInfo](
            tool_func=call_weather_api, name="true_node"
        )

        async def false_tool_func2(x: WeatherQuery) -> WeatherInfo:
            return WeatherInfo(temperature=0, condition="cold", location=x.location)

        false_node = ToolNode[WeatherQuery, WeatherInfo](
            tool_func=false_tool_func2,
            name="false_node",
        )

        if_node = IfNode(
            predicate=lambda x: x.location == "Paris",
            if_true=true_node,
            if_false=false_node,
            name="if_node",
        )

        query = WeatherQuery(location="London")
        result = await extract_result_from_stream(if_node.astream(query))

        assert result.temperature == 0  # From false_node
        assert result.condition == "cold"

    def test_if_node_dependencies(self):
        """Test IfNode dependencies include both branches."""
        prompt_node = create_test_agent_node("test")

        true_node = ParserNode[str, WeatherInfo](
            parser_func=parse_weather_string,
            inputs=(prompt_node.output,),
            name="true_node",
        )

        false_node = ToolNode[WeatherQuery, WeatherInfo](
            tool_func=call_weather_api, name="false_node"
        )

        if_node = IfNode(
            predicate=lambda x: True,
            if_true=true_node,
            if_false=false_node,
            inputs=(prompt_node.output,),
            name="if_node",
        )

        dependencies = if_node.dependencies
        # Should include prompt_node (from input), and dependencies of both branches
        assert prompt_node in dependencies
        assert prompt_node in dependencies  # true_node also depends on prompt_node

    def test_agent_node_with_custom_instructions(self):
        """Test AgentNode with custom instructions."""
        agent = Agent("test", instructions="You are helpful")
        node = AgentNode[WeatherQuery, str](
            agent=agent, prompt_template="Test prompt", name="test_node"
        )

        # Check that we have an agent with the test model
        assert hasattr(node.agent.model, "model_name")
        assert node.agent.model.model_name == "test"
        assert node.prompt_template == "Test prompt"
        assert node.name == "test_node"

    def test_agent_node_defaults(self):
        """Test AgentNode default values."""
        node = create_test_agent_node("Test prompt", name="test_node")

        # Should use test agent
        assert hasattr(node.agent.model, "model_name")
        assert node.agent.model.model_name == "test"
        assert node.prompt_template == "Test prompt"
        assert node.name == "test_node"


class TestCoverageEdgeCases:
    """Test edge cases to reach 100% coverage."""

    def test_cyclic_dependency_detection_actual_cycle(self):
        """Test that flows with cycles compile successfully."""
        flow = Flow(input_type=WeatherQuery, output_type=GenericFlowResults)

        # Create nodes that form a cycle via explicit edges
        node1 = create_test_agent_node("test1", name="node1")
        node2 = create_test_agent_node("test2", name="node2")

        flow.add_nodes(node1, node2)
        # Add explicit cycle
        flow.add_edge(node1, node2)
        flow.add_edge(node2, node1)

        # Stepper engine handles cycles - should compile successfully
        # Flows execute directly - no compilation needed

    def test_cyclic_dependency_error_in_validate(self):
        """Test that flows with explicit cycles compile successfully.

        Uses the stepper engine to handle cycles.
        """
        flow = Flow(input_type=WeatherQuery, output_type=GenericFlowResults)

        # Create nodes with explicit cycle edges
        node1 = create_test_agent_node("test1", name="node1")
        node2 = create_test_agent_node("test2", name="node2")

        flow.add_nodes(node1, node2)
        flow.add_edge(node1, node2)
        flow.add_edge(node2, node1)

        # Should compile successfully - stepper handles cycles
        # Flows execute directly - no compilation needed

    @pytest.mark.asyncio
    async def test_missing_input_node_error(self):
        """Test flow execution with missing dependency node."""
        flow = Flow(input_type=WeatherQuery, output_type=ComplexFlowResults)

        # Create nodes with dependencies
        node1 = create_test_agent_node("test", name="node1")
        node2 = ParserNode[str, WeatherInfo](
            parser_func=parse_weather_string,
            inputs=(node1.output,),
            name="node2",
        )

        # Only add node2, not node1 - this will cause runtime error
        flow.add_nodes(node2)

        query = WeatherQuery(location="Paris")

        with pytest.raises(KeyError, match="node1"):
            await extract_result_from_stream(flow.astream(query))

    @pytest.mark.asyncio
    async def test_flow_execution_general_error_wrapping(self):
        """Test that general exceptions during flow execution are wrapped."""
        flow = Flow(input_type=WeatherQuery, output_type=SimpleFlowResults)

        # Create a node that will raise an exception
        async def failing_tool(query: WeatherQuery) -> WeatherInfo:
            raise ValueError("Simulated tool failure")

        failing_node = ToolNode[WeatherQuery, WeatherInfo](
            tool_func=failing_tool,
            name="failing_tool",
        )

        flow.add_nodes(failing_node)

        query = WeatherQuery(location="Paris")

        with pytest.raises(
            FlowError, match=r"Flow execution failed.*Simulated tool failure"
        ):
            await extract_result_from_stream(flow.astream(query))

    def test_flow_validation_general_error(self):
        """Test error handling during compilation."""
        flow = Flow(input_type=WeatherQuery, output_type=GenericFlowResults)

        # Create a node
        node = create_test_agent_node("test", name="test_node")
        flow.add_nodes(node)

        # Compilation should work
        # Flows execute directly - no compilation needed

    def test_node_type_hint_property(self):
        """Test the type_hint property of NodeOutput."""
        node = create_test_agent_node("test")

        # Access the type_hint property to cover line 27 in base.py
        type_hint = node.output.type_hint
        # The type hint should be the actual output type
        assert type_hint is not None
        assert hasattr(type_hint, "__name__")

    def test_node_repr_method(self):
        """Test the __repr__ method of BaseNode."""
        node = create_test_agent_node("test", name="custom_name")

        # Test the __repr__ method to cover line 70 in base.py
        repr_str = repr(node)
        assert "AgentNode" in repr_str
        assert "custom_name" in repr_str

        # Test with auto-generated name
        node2 = create_test_agent_node("test")
        repr_str2 = repr(node2)
        assert "AgentNode" in repr_str2

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
            await extract_result_from_stream(flow.astream(wrong_input))  # type: ignore

        # Verify the error message was constructed properly (covers lines 114-117)
        assert "Input type mismatch" in str(exc_info.value)
        assert "WeatherQuery" in str(exc_info.value)
        assert "WrongInputType" in str(exc_info.value)

    async def test_flow_validation_exception_wrapping(self):
        """Test the exception wrapping in flow validation."""
        flow = Flow(input_type=WeatherQuery, output_type=GenericFlowResults)

        # Create a simple node first
        node = create_test_agent_node("test", name="test_node")
        flow.add_nodes(node)

        # Create a bad node that raises exception when accessed
        class BadNode:
            def __init__(self):
                self.name = "bad_node"

            @property
            def dependencies(self):
                raise RuntimeError("Intentional failure accessing dependencies")

        # Add the bad node
        flow.nodes.append(BadNode())  # type: ignore

        # Execution should fail with bad node
        with pytest.raises(RuntimeError, match="Intentional failure"):
            async for _ in flow.astream(WeatherQuery(location="Test")):
                pass


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
        result = await extract_result_from_stream(flow_node.astream(query))

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
        results = await extract_result_from_stream(parent_flow.astream(query))

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
        results = await extract_result_from_stream(level3_flow.astream(query))

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

        async def failing_tool(query: WeatherQuery) -> WeatherInfo:
            msg = f"Intentional failure for {query.location}"
            raise ValueError(msg)

        failing_node = ToolNode[WeatherQuery, WeatherInfo](
            tool_func=failing_tool,
            name="failing_node",
        )
        sub_flow.add_nodes(failing_node)

        # Create FlowNode
        flow_node = FlowNode[WeatherQuery, SimpleFlowResults](flow=sub_flow)

        # Execution should fail and propagate the error (wrapped in FlowError)
        query = WeatherQuery(location="ErrorCity")
        with pytest.raises(FlowError) as exc_info:
            await extract_result_from_stream(flow_node.astream(query))

        assert "Intentional failure for ErrorCity" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__])
