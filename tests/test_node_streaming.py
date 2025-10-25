"""Tests for node astream() methods - the primary streaming interface.

This module tests that all node types properly implement astream() as the
first-class interface, emitting the correct sequence of ProgressItems.
"""

from pydantic import BaseModel
import pytest

from pydantic_flow.nodes import IfNode
from pydantic_flow.nodes import MergeParserNode
from pydantic_flow.nodes import MergeToolNode
from pydantic_flow.nodes import NodeWithInput
from pydantic_flow.nodes import ParserNode
from pydantic_flow.nodes import RetryNode
from pydantic_flow.nodes import ToolNode
from pydantic_flow.streaming.events import NonFatalError
from pydantic_flow.streaming.events import ProgressType
from pydantic_flow.streaming.events import StreamEnd
from pydantic_flow.streaming.events import StreamStart
from pydantic_flow.streaming.events import ToolCall
from pydantic_flow.streaming.events import ToolResult


class SimpleInput(BaseModel):
    """Test input model."""

    value: int


class SimpleOutput(BaseModel):
    """Test output model."""

    result: int


class TransformedOutput(BaseModel):
    """Transformed output model."""

    transformed: str


class MergedOutput(BaseModel):
    """Merged output model."""

    combined: str


@pytest.mark.asyncio
async def test_tool_node_astream_sequence():
    """Test that ToolNode emits correct streaming sequence."""

    def multiply_by_two(input_data: SimpleInput) -> SimpleOutput:
        return SimpleOutput(result=input_data.value * 2)

    node = ToolNode[SimpleInput, SimpleOutput](
        tool_func=multiply_by_two, name="test_tool"
    )

    items = []
    async for item in node.astream(SimpleInput(value=5)):
        items.append(item)

    # Verify sequence: StreamStart -> ToolCall -> ToolResult -> StreamEnd
    assert len(items) == 4
    assert isinstance(items[0], StreamStart)
    assert items[0].node_id == "test_tool"

    assert isinstance(items[1], ToolCall)
    assert items[1].tool_name == "multiply_by_two"
    assert items[1].call_id != ""

    assert isinstance(items[2], ToolResult)
    assert items[2].tool_name == "multiply_by_two"
    assert items[2].result == SimpleOutput(result=10)
    assert items[2].error is None

    assert isinstance(items[3], StreamEnd)
    assert items[3].result_preview is not None


@pytest.mark.asyncio
async def test_tool_node_error_handling():
    """Test that ToolNode emits error in ToolResult on failure."""

    def failing_tool(input_data: SimpleInput) -> SimpleOutput:
        raise ValueError("Tool failed!")

    node = ToolNode[SimpleInput, SimpleOutput](
        tool_func=failing_tool, name="failing_tool"
    )

    items = []
    with pytest.raises(ValueError):
        async for item in node.astream(SimpleInput(value=5)):
            items.append(item)

    # Should have StreamStart, ToolCall, and ToolResult with error
    assert len(items) >= 3
    assert isinstance(items[0], StreamStart)
    assert isinstance(items[1], ToolCall)
    assert isinstance(items[2], ToolResult)
    assert items[2].error == "Tool failed!"


@pytest.mark.asyncio
async def test_parser_node_astream_sequence():
    """Test that ParserNode emits correct streaming sequence."""

    def parse_to_string(input_data: SimpleInput) -> TransformedOutput:
        return TransformedOutput(transformed=f"value_{input_data.value}")

    node = ParserNode[SimpleInput, TransformedOutput](
        parser_func=parse_to_string, name="test_parser"
    )

    items = []
    async for item in node.astream(SimpleInput(value=42)):
        items.append(item)

    # Verify sequence: StreamStart -> ToolResult -> StreamEnd
    assert len(items) == 3
    assert isinstance(items[0], StreamStart)
    assert items[0].node_id == "test_parser"

    assert isinstance(items[1], ToolResult)
    assert items[1].result == TransformedOutput(transformed="value_42")

    assert isinstance(items[2], StreamEnd)
    assert items[2].result_preview is not None
    assert items[2].result_preview.get("transformed") == "value_42"


@pytest.mark.asyncio
async def test_if_node_astream_forwards_branch_progress():
    """Test that IfNode forwards progress from the chosen branch."""

    class BranchNode(NodeWithInput[SimpleInput, SimpleOutput]):
        """Test branch node."""

        async def astream(self, input_data: SimpleInput):
            yield StreamStart(run_id="", node_id=self.name)
            result = SimpleOutput(result=input_data.value * 10)
            yield StreamEnd(
                run_id="", node_id=self.name, result_preview=result.model_dump()
            )

    true_branch = BranchNode(name="true_branch")
    false_branch = BranchNode(name="false_branch")

    def is_positive(input_data: SimpleInput) -> bool:
        return input_data.value > 0

    if_node = IfNode[SimpleOutput](
        predicate=is_positive,
        if_true=true_branch,
        if_false=false_branch,
        name="if_node",
    )

    # Test true branch
    items = []
    async for item in if_node.astream(SimpleInput(value=5)):
        items.append(item)

    # Should have: IfNode StreamStart -> BranchNode StreamStart
    # -> BranchNode StreamEnd -> IfNode StreamEnd
    assert len(items) >= 3
    assert isinstance(items[0], StreamStart)
    assert items[0].node_id == "if_node"

    # Find the branch's StreamStart (not forwarded as StreamEnd by IfNode)
    branch_items = [
        item for item in items if isinstance(item, (StreamStart, StreamEnd))
    ]
    assert any(
        item.node_id == "true_branch"
        for item in branch_items
        if isinstance(item, StreamStart)
    )

    # Last item should be IfNode's StreamEnd
    assert isinstance(items[-1], StreamEnd)
    assert items[-1].node_id == "if_node"


@pytest.mark.asyncio
async def test_retry_node_emits_errors_on_retry():
    """Test that RetryNode emits NonFatalError on retries."""

    class UnreliableNode(NodeWithInput[SimpleInput, SimpleOutput]):
        """Node that fails twice then succeeds."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.attempt_count = 0

        async def astream(self, input_data: SimpleInput):
            yield StreamStart(run_id="", node_id=self.name)
            self.attempt_count += 1
            if self.attempt_count < 3:
                raise RuntimeError(f"Attempt {self.attempt_count} failed")
            result = SimpleOutput(result=input_data.value)
            yield StreamEnd(
                run_id="", node_id=self.name, result_preview=result.model_dump()
            )

    unreliable = UnreliableNode(name="unreliable")
    retry_node = RetryNode[SimpleOutput](
        wrapped_node=unreliable, max_retries=3, name="retry_node"
    )

    items = []
    async for item in retry_node.astream(SimpleInput(value=10)):
        items.append(item)

    # Should have RetryNode StreamStart, NonFatalError items, and RetryNode StreamEnd
    error_items = [item for item in items if isinstance(item, NonFatalError)]
    assert len(error_items) >= 2  # At least 2 retry warnings

    # Verify error messages mention attempts
    assert any("Attempt 1 failed" in item.message for item in error_items)
    assert any("Attempt 2 failed" in item.message for item in error_items)

    # Should eventually succeed
    assert isinstance(items[-1], StreamEnd)
    assert items[-1].node_id == "retry_node"


@pytest.mark.asyncio
async def test_merge_tool_node_astream_sequence():
    """Test that MergeToolNode emits correct streaming sequence."""

    def merge_two_values(a: SimpleInput, b: SimpleInput) -> MergedOutput:
        return MergedOutput(combined=f"{a.value}+{b.value}")

    # Create source nodes
    node_a = ToolNode[SimpleInput, SimpleInput](tool_func=lambda x: x, name="node_a")
    node_b = ToolNode[SimpleInput, SimpleInput](tool_func=lambda x: x, name="node_b")

    merge_node = MergeToolNode[SimpleInput, SimpleInput, MergedOutput](
        inputs=(node_a.output, node_b.output),
        tool_func=merge_two_values,
        name="merge_node",
    )

    items = []
    input_tuple = (SimpleInput(value=1), SimpleInput(value=2))
    async for item in merge_node.astream(input_tuple):
        items.append(item)

    # Verify sequence: StreamStart -> ToolCall -> ToolResult -> StreamEnd
    assert len(items) == 4
    assert isinstance(items[0], StreamStart)
    assert isinstance(items[1], ToolCall)
    assert isinstance(items[2], ToolResult)
    assert isinstance(items[3], StreamEnd)

    # Verify the merge happened
    assert items[3].result_preview is not None
    assert items[3].result_preview.get("combined") == "1+2"


@pytest.mark.asyncio
async def test_merge_parser_node_astream_sequence():
    """Test that MergeParserNode emits correct streaming sequence."""

    def parse_combined(a: SimpleInput, b: SimpleInput) -> MergedOutput:
        return MergedOutput(combined=f"a={a.value},b={b.value}")

    node_a = ToolNode[SimpleInput, SimpleInput](tool_func=lambda x: x, name="node_a")
    node_b = ToolNode[SimpleInput, SimpleInput](tool_func=lambda x: x, name="node_b")

    merge_parser = MergeParserNode[SimpleInput, SimpleInput, MergedOutput](
        inputs=(node_a.output, node_b.output),
        parser_func=parse_combined,
        name="merge_parser",
    )

    items = []
    input_tuple = (SimpleInput(value=10), SimpleInput(value=20))
    async for item in merge_parser.astream(input_tuple):
        items.append(item)

    # Verify sequence: StreamStart -> StreamEnd (parser nodes are simple)
    assert len(items) == 2
    assert isinstance(items[0], StreamStart)
    assert isinstance(items[1], StreamEnd)
    assert items[1].result_preview is not None
    assert items[1].result_preview.get("combined") == "a=10,b=20"


@pytest.mark.asyncio
async def test_stream_coherence():
    """Test that all streams have coherent start and end markers."""

    def simple_func(input_data: SimpleInput) -> SimpleOutput:
        return SimpleOutput(result=input_data.value + 1)

    node = ToolNode[SimpleInput, SimpleOutput](
        tool_func=simple_func, name="coherence_test"
    )

    items = []
    async for item in node.astream(SimpleInput(value=1)):
        items.append(item)

    # First item must be StreamStart
    assert isinstance(items[0], StreamStart)
    assert items[0].type == ProgressType.START

    # Last item must be StreamEnd
    assert isinstance(items[-1], StreamEnd)
    assert items[-1].type == ProgressType.END

    # All items should have consistent run_id and node_id
    run_id = items[0].run_id
    node_id = items[0].node_id
    for item in items:
        assert item.run_id == run_id
        assert item.node_id == node_id


@pytest.mark.asyncio
async def test_progress_item_metadata():
    """Test that all progress items have proper metadata."""

    def simple_func(input_data: SimpleInput) -> SimpleOutput:
        return SimpleOutput(result=input_data.value)

    node = ToolNode[SimpleInput, SimpleOutput](
        tool_func=simple_func, name="metadata_test"
    )

    items = []
    async for item in node.astream(SimpleInput(value=42)):
        items.append(item)

    # All items should have timestamps
    for item in items:
        assert item.timestamp is not None
        assert item.node_id == "metadata_test"
        assert item.type in [t.value for t in ProgressType]


@pytest.mark.asyncio
async def test_run_wraps_astream():
    """Test that run() is a convenience wrapper that consumes astream()."""

    def simple_func(input_data: SimpleInput) -> SimpleOutput:
        return SimpleOutput(result=input_data.value * 3)

    node = ToolNode[SimpleInput, SimpleOutput](
        tool_func=simple_func, name="wrapper_test"
    )

    # Call run() which should internally consume astream()
    result = await node.run(SimpleInput(value=7))

    # Verify we get the actual model object, not a dict
    assert isinstance(result, SimpleOutput)
    assert result.result == 21


@pytest.mark.asyncio
async def test_multiple_tool_calls_in_sequence():
    """Test multiple tool nodes streaming in sequence."""

    def double(input_data: SimpleInput) -> SimpleOutput:
        return SimpleOutput(result=input_data.value * 2)

    def triple(input_data: SimpleOutput) -> SimpleOutput:
        return SimpleOutput(result=input_data.result * 3)

    node1 = ToolNode[SimpleInput, SimpleOutput](tool_func=double, name="node1")
    node2 = ToolNode[SimpleOutput, SimpleOutput](tool_func=triple, name="node2")

    # Stream through first node
    result1 = await node1.run(SimpleInput(value=5))
    assert result1.result == 10

    # Stream through second node
    items = []
    async for item in node2.astream(result1):
        items.append(item)

    # Verify second node's stream
    assert isinstance(items[0], StreamStart)
    assert items[0].node_id == "node2"
    assert isinstance(items[-1], StreamEnd)

    # Verify final result
    result2 = await node2.run(result1)
    assert result2.result == 30
