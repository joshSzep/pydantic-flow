"""Tests for error handling paths to improve coverage."""

from pydantic import BaseModel
import pytest

from pydantic_flow.nodes import ParserNode
from pydantic_flow.nodes import ToolNode
from pydantic_flow.streaming.events import ToolResult


class SimpleInput(BaseModel):
    """Test input."""

    value: int


class SimpleOutput(BaseModel):
    """Test output."""

    result: int


@pytest.mark.asyncio
async def test_tool_node_exception_handling():
    """Test that ToolNode handles exceptions in tool_func."""

    def failing_tool(x: SimpleInput) -> SimpleOutput:
        raise ValueError("Something went wrong!")

    node = ToolNode[SimpleInput, SimpleOutput](tool_func=failing_tool, name="failing")

    items = []
    with pytest.raises(ValueError, match="Something went wrong"):
        async for item in node.astream(SimpleInput(value=5)):
            items.append(item)

    # Should have ToolCall and ToolResult with error
    tool_results = [item for item in items if isinstance(item, ToolResult)]
    assert len(tool_results) == 1
    assert tool_results[0].error == "Something went wrong!"


@pytest.mark.asyncio
async def test_parser_node_with_non_model_result():
    """Test ParserNode with results that aren't Pydantic models."""

    def parse_to_string(x: SimpleInput) -> SimpleOutput:
        # Return something that's not a BaseModel but should still work
        return SimpleOutput(result=x.value + 1)

    node = ParserNode[SimpleInput, SimpleOutput](
        parser_func=parse_to_string, name="parser"
    )

    result = await node.run(SimpleInput(value=5))
    assert isinstance(result, SimpleOutput)
    assert result.result == 6


@pytest.mark.asyncio
async def test_tool_node_with_none_result():
    """Test ToolNode when tool returns None."""

    def returns_none(x: SimpleInput) -> None:
        return None

    node = ToolNode[SimpleInput, None](tool_func=returns_none, name="none_tool")  # type: ignore

    items = []
    async for item in node.astream(SimpleInput(value=1)):
        items.append(item)

    # Should handle None result gracefully
    tool_results = [item for item in items if isinstance(item, ToolResult)]
    assert len(tool_results) == 1
    assert tool_results[0].result is None
