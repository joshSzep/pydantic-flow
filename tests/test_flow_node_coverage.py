"""Additional tests for FlowNode to improve coverage."""

from pydantic import BaseModel
import pytest

from pydantic_flow.flow.flow import Flow
from pydantic_flow.nodes.flow import FlowNode
from pydantic_flow.nodes.tool import ToolNode
from pydantic_flow.streaming.events import StreamEnd
from pydantic_flow.streaming.events import StreamStart
from pydantic_flow.streaming.events import TokenChunk
from pydantic_flow.streaming.events import ToolResult


class SimpleInput(BaseModel):
    """Simple input for testing."""

    value: str


class SimpleOutput(BaseModel):
    """Simple output for testing."""

    result: str


def simple_tool(input_data: SimpleInput) -> SimpleOutput:
    """Process input data."""
    return SimpleOutput(result=f"processed: {input_data.value}")


@pytest.mark.asyncio
async def test_flow_node_astream_with_flow_astream():
    """Test FlowNode astream when wrapped flow has astream method."""

    class MockFlowWithAstream:
        """Mock flow with astream method."""

        async def astream(self, input_data):
            """Stream progress from flow."""
            yield StreamStart(run_id="inner", node_id="inner")
            yield ToolResult(
                run_id="inner",
                node_id="inner",
                tool_name="test",
                call_id="1",
                result=SimpleOutput(result="test result"),
                error=None,
            )
            yield StreamEnd(
                run_id="inner",
                node_id="inner",
                result_preview={"result": "test result"},
            )

    mock_flow = MockFlowWithAstream()
    flow_node = FlowNode[SimpleInput, SimpleOutput](
        flow=mock_flow,  # type: ignore
        name="test_flow_node",
    )

    # Execute via astream
    input_data = SimpleInput(value="test")
    items = []

    async for item in flow_node.astream(input_data):
        items.append(item)

    # Should have StreamStart and StreamEnd from FlowNode
    assert len(items) >= 2
    assert isinstance(items[0], StreamStart)
    assert items[0].node_id == "test_flow_node"

    # Last item should be StreamEnd
    assert isinstance(items[-1], StreamEnd)
    assert items[-1].node_id == "test_flow_node"

    # Should have forwarded the ToolResult
    tool_results = [i for i in items if isinstance(i, ToolResult)]
    assert len(tool_results) >= 1


@pytest.mark.asyncio
async def test_flow_node_astream_filters_wrapped_stream_events():
    """Test FlowNode filters out StreamStart/StreamEnd from wrapped flow."""

    class MockFlowWithAstream:
        """Mock flow with astream method."""

        async def astream(self, input_data):
            """Stream progress from flow."""
            yield StreamStart(run_id="inner", node_id="inner")
            yield TokenChunk(
                text="test", token_index=0, run_id="inner", node_id="inner"
            )
            yield StreamEnd(run_id="inner", node_id="inner")

    mock_flow = MockFlowWithAstream()
    flow_node = FlowNode[SimpleInput, SimpleOutput](
        flow=mock_flow,  # type: ignore
        name="test_flow_node",
    )

    input_data = SimpleInput(value="test")
    items = []

    async for item in flow_node.astream(input_data):
        items.append(item)

    # Count StreamStart and StreamEnd - should only be 2 (from FlowNode)
    stream_starts = [i for i in items if isinstance(i, StreamStart)]
    stream_ends = [i for i in items if isinstance(i, StreamEnd)]

    assert len(stream_starts) == 1
    assert len(stream_ends) == 1
    assert stream_starts[0].node_id == "test_flow_node"
    assert stream_ends[0].node_id == "test_flow_node"


@pytest.mark.asyncio
async def test_flow_node_astream_captures_tool_result():
    """Test FlowNode captures and emits ToolResult from wrapped flow."""

    class MockFlowWithAstream:
        """Mock flow with astream method."""

        async def astream(self, input_data):
            """Stream progress from flow."""
            yield StreamStart(run_id="inner", node_id="inner")
            yield ToolResult(
                run_id="inner",
                node_id="inner",
                tool_name="test_tool",
                call_id="123",
                result=SimpleOutput(result="captured"),
                error=None,
            )
            yield StreamEnd(run_id="inner", node_id="inner")

    mock_flow = MockFlowWithAstream()
    flow_node = FlowNode[SimpleInput, SimpleOutput](
        flow=mock_flow,  # type: ignore
        name="test_flow_node",
    )

    input_data = SimpleInput(value="test")
    items = []

    async for item in flow_node.astream(input_data):
        items.append(item)

    # Should have at least one ToolResult
    tool_results = [i for i in items if isinstance(i, ToolResult)]
    assert len(tool_results) >= 1

    # Check that final ToolResult is from the FlowNode itself
    final_tool_result = None
    for item in reversed(items):
        if isinstance(item, ToolResult):
            final_tool_result = item
            break

    assert final_tool_result is not None
    assert final_tool_result.tool_name == "flow"
    assert final_tool_result.error is None


@pytest.mark.asyncio
async def test_flow_node_astream_forwards_other_progress():
    """Test FlowNode forwards non-stream progress items from wrapped flow."""

    class MockFlowWithAstream:
        """Mock flow with astream method."""

        async def astream(self, input_data):
            """Stream progress from flow."""
            yield StreamStart(run_id="inner", node_id="inner")
            yield TokenChunk(
                text="chunk1", token_index=0, run_id="inner", node_id="inner"
            )
            yield TokenChunk(
                text="chunk2", token_index=1, run_id="inner", node_id="inner"
            )
            yield StreamEnd(run_id="inner", node_id="inner")

    mock_flow = MockFlowWithAstream()
    flow_node = FlowNode[SimpleInput, SimpleOutput](
        flow=mock_flow,  # type: ignore
        name="test_flow_node",
    )

    input_data = SimpleInput(value="test")
    items = []

    async for item in flow_node.astream(input_data):
        items.append(item)

    # Should forward TokenChunk items from the wrapped flow
    forwarded_tokens = [i for i in items if isinstance(i, TokenChunk)]
    assert len(forwarded_tokens) == 2
    assert forwarded_tokens[0].text == "chunk1"
    assert forwarded_tokens[1].text == "chunk2"


@pytest.mark.asyncio
async def test_flow_node_fallback_to_run():
    """Test FlowNode falls back to run() when flow doesn't have astream."""

    class MockFlowWithoutStream:
        """Mock flow without astream method."""

        async def run(self, input_data):
            """Run method."""
            return SimpleOutput(result="from run")

    mock_flow = MockFlowWithoutStream()
    flow_node = FlowNode[SimpleInput, SimpleOutput](
        flow=mock_flow,  # type: ignore
        name="test_flow_node",
    )

    input_data = SimpleInput(value="test")
    items = []

    async for item in flow_node.astream(input_data):
        items.append(item)

    # Should have StreamStart, ToolResult, and StreamEnd
    assert len(items) == 3
    assert isinstance(items[0], StreamStart)
    assert isinstance(items[1], ToolResult)
    assert isinstance(items[2], StreamEnd)

    # Check ToolResult
    assert items[1].tool_name == "flow"
    assert items[1].result.result == "from run"  # type: ignore

    # Check StreamEnd has result_preview
    assert items[2].result_preview == {"result": "from run"}


@pytest.mark.asyncio
async def test_flow_node_fallback_result_without_model_dump():
    """Test FlowNode handles results without model_dump in fallback path."""

    class MockFlowWithoutStream:
        """Mock flow without astream method."""

        async def run(self, input_data):
            """Run method returning plain string."""
            return "plain string result"

    mock_flow = MockFlowWithoutStream()
    flow_node = FlowNode[SimpleInput, str](  # type: ignore
        flow=mock_flow,  # type: ignore
        name="test_flow_node",
    )

    input_data = SimpleInput(value="test")
    items = []

    async for item in flow_node.astream(input_data):
        items.append(item)

    # Check StreamEnd has result_preview with value wrapper
    stream_end = items[-1]
    assert isinstance(stream_end, StreamEnd)
    assert stream_end.result_preview == {"value": "plain string result"}


@pytest.mark.asyncio
async def test_flow_node_repr():
    """Test FlowNode string representation."""
    sub_flow = Flow(input_type=SimpleInput, output_type=SimpleOutput)
    flow_node = FlowNode[SimpleInput, SimpleOutput](
        flow=sub_flow,
        name="test_flow",
    )

    repr_str = repr(flow_node)
    assert "FlowNode" in repr_str
    assert "test_flow" in repr_str


@pytest.mark.asyncio
async def test_flow_node_dependencies_with_input():
    """Test FlowNode dependencies when it has an input node."""
    sub_flow = Flow(input_type=SimpleInput, output_type=SimpleOutput)
    upstream_node = ToolNode[SimpleInput, SimpleInput](
        tool_func=lambda x: x,
        name="upstream_tool",
    )

    # Create a flow node with input dependency
    flow_node = FlowNode[SimpleInput, SimpleOutput](
        flow=sub_flow,
        input=upstream_node.output,  # type: ignore
        name="dependent_flow_node",
    )

    # Dependencies should include the input node
    deps = flow_node.dependencies
    assert len(deps) == 1
    assert deps[0] is upstream_node
