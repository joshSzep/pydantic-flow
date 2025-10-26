"""Tests for Flow smart detection of already-constructed output types."""

from pydantic import BaseModel

from pydantic_flow import Flow
from pydantic_flow.nodes import FlowNode
from pydantic_flow.nodes import ToolNode


class InputText(BaseModel):
    """Input text model."""

    text: str


class SimpleOutput(BaseModel):
    """Simple output model."""

    result: str


class TextWrapper(BaseModel):
    """Wrapper for text strings."""

    text: str


class MultiFieldOutput(BaseModel):
    """Multi-field output."""

    result: TextWrapper
    data: TextWrapper


class NestedOutput(BaseModel):
    """Nested output that contains another model."""

    data: SimpleOutput


class TestFlowSmartDetection:
    """Test that Flow can detect when a single node returns the correct output type."""

    async def test_single_node_returns_output_type_directly(self):
        """Test smart detection: single node returns the exact output type."""

        def produce_output(inp: InputText) -> SimpleOutput:
            return SimpleOutput(result=f"Processed: {inp.text}")

        flow = Flow[InputText, SimpleOutput](
            input_type=InputText, output_type=SimpleOutput
        )

        node = ToolNode[InputText, SimpleOutput](
            tool_func=produce_output, name="processor"
        )
        flow.add_nodes(node)
        flow.compile()

        result = await flow.run(InputText(text="test"))

        assert isinstance(result, SimpleOutput)
        assert result.result == "Processed: test"

    async def test_multiple_nodes_uses_field_mapping(self):
        """Test that multiple nodes still use field mapping (no smart detection)."""

        def produce_result(inp: InputText) -> TextWrapper:
            return TextWrapper(text=f"Processed: {inp.text}")

        def produce_data(inp: InputText) -> TextWrapper:
            return TextWrapper(text=f"Data: {inp.text}")

        flow = Flow[InputText, MultiFieldOutput](
            input_type=InputText, output_type=MultiFieldOutput
        )

        # Two nodes that map to the output fields
        node1 = ToolNode[InputText, TextWrapper](
            tool_func=produce_result, name="result"
        )
        node2 = ToolNode[InputText, TextWrapper](tool_func=produce_data, name="data")

        flow.add_nodes(node1, node2)
        flow.compile()

        result = await flow.run(InputText(text="test"))

        assert isinstance(result, MultiFieldOutput)
        assert result.result.text == "Processed: test"
        assert result.data.text == "Data: test"

    async def test_single_node_wrong_type_uses_field_mapping(self):
        """Test that single node with wrong type still uses field mapping."""

        def produce_result(inp: InputText) -> TextWrapper:
            return TextWrapper(text=f"Processed: {inp.text}")

        class WrapperOutput(BaseModel):
            result: TextWrapper

        flow = Flow[InputText, WrapperOutput](
            input_type=InputText, output_type=WrapperOutput
        )

        # Single node but returns TextWrapper, not WrapperOutput
        # Should map result to the "result" field
        node = ToolNode[InputText, TextWrapper](tool_func=produce_result, name="result")
        flow.add_nodes(node)
        flow.compile()

        result = await flow.run(InputText(text="test"))

        assert isinstance(result, WrapperOutput)
        assert result.result.text == "Processed: test"

    async def test_sub_flow_with_matching_output_type(self):
        """Test that sub-flows work when they return the same type as parent."""

        def produce_output(inp: InputText) -> SimpleOutput:
            return SimpleOutput(result=f"Sub: {inp.text}")

        # Sub-flow that produces SimpleOutput
        sub_flow = Flow[InputText, SimpleOutput](
            input_type=InputText, output_type=SimpleOutput
        )
        node = ToolNode[InputText, SimpleOutput](
            tool_func=produce_output, name="processor"
        )
        sub_flow.add_nodes(node)
        sub_flow.compile()

        # Parent flow also produces SimpleOutput
        parent_flow = Flow[InputText, SimpleOutput](
            input_type=InputText, output_type=SimpleOutput
        )

        flow_node = FlowNode[InputText, SimpleOutput](
            flow=sub_flow, name="sub_processor"
        )
        parent_flow.add_nodes(flow_node)
        parent_flow.compile()

        result = await parent_flow.run(InputText(text="test"))

        assert isinstance(result, SimpleOutput)
        assert result.result == "Sub: test"

    async def test_nested_output_smart_detection(self):
        """Test smart detection with nested models."""

        def produce_nested(inp: InputText) -> NestedOutput:
            return NestedOutput(data=SimpleOutput(result=f"Nested: {inp.text}"))

        flow = Flow[InputText, NestedOutput](
            input_type=InputText, output_type=NestedOutput
        )

        node = ToolNode[InputText, NestedOutput](
            tool_func=produce_nested, name="processor"
        )
        flow.add_nodes(node)
        flow.compile()

        result = await flow.run(InputText(text="test"))

        assert isinstance(result, NestedOutput)
        assert isinstance(result.data, SimpleOutput)
        assert result.data.result == "Nested: test"
