"""Tests for additional coverage of near-complete modules."""

from pydantic import BaseModel
import pytest

from pydantic_flow.nodes import IfNode
from pydantic_flow.nodes import ParserNode
from pydantic_flow.nodes import RetryNode
from pydantic_flow.nodes import ToolNode
from pydantic_flow.streaming.events import NonFatalError
from pydantic_flow.streaming.events import StreamEnd
from pydantic_flow.streaming.events import StreamStart


class SimpleInput(BaseModel):
    """Test input."""

    value: int


class SimpleOutput(BaseModel):
    """Test output."""

    result: int


@pytest.mark.asyncio
async def test_parser_node_with_primitive_result():
    """Test ParserNode that returns primitive (not BaseModel)."""

    def parse_to_int(x: SimpleInput) -> int:  # type: ignore
        return x.value * 2

    # ParserNode expects BaseModel output, but let's test the branch
    node = ParserNode[SimpleInput, int](  # type: ignore
        parser_func=parse_to_int, name="parser"
    )

    items = []
    async for item in node.astream(SimpleInput(value=5)):
        items.append(item)

    # Should have wrapped primitive in {"value": str(result)}
    end_items = [item for item in items if isinstance(item, StreamEnd)]
    assert len(end_items) == 1
    assert end_items[0].result_preview == {"value": "10"}


@pytest.mark.asyncio
async def test_if_node_false_branch():
    """Test IfNode taking the false branch."""

    def always_false(x: SimpleInput) -> bool:
        return False

    def double(x: SimpleInput) -> SimpleOutput:
        return SimpleOutput(result=x.value * 2)

    def triple(x: SimpleInput) -> SimpleOutput:
        return SimpleOutput(result=x.value * 3)

    true_node = ToolNode[SimpleInput, SimpleOutput](tool_func=double, name="true")
    false_node = ToolNode[SimpleInput, SimpleOutput](tool_func=triple, name="false")

    if_node = IfNode[SimpleOutput](
        predicate=always_false,
        if_true=true_node,
        if_false=false_node,
        name="if",
    )

    result = await if_node.run(SimpleInput(value=5))
    # Should have taken false branch (triple)
    assert result.result == 15


@pytest.mark.asyncio
async def test_retry_node_success_on_retry():
    """Test RetryNode that succeeds after retries."""

    class UnreliableNode(ToolNode[SimpleInput, SimpleOutput]):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.attempts = 0

        async def astream(self, input_data: SimpleInput):
            self.attempts += 1
            if self.attempts < 2:
                # Fail first attempt
                raise RuntimeError("Temporary failure")
            # Succeed on second attempt
            async for item in super().astream(input_data):
                yield item

    def process(x: SimpleInput) -> SimpleOutput:
        return SimpleOutput(result=x.value + 1)

    unreliable = UnreliableNode(tool_func=process, name="unreliable")
    retry_node = RetryNode[SimpleOutput](
        wrapped_node=unreliable, max_retries=3, name="retry"
    )

    items = []
    async for item in retry_node.astream(SimpleInput(value=10)):
        items.append(item)

    # Should have NonFatalError from first attempt
    errors = [item for item in items if isinstance(item, NonFatalError)]
    assert len(errors) >= 1

    # Should eventually succeed
    end_items = [item for item in items if isinstance(item, StreamEnd)]
    assert len(end_items) == 1


@pytest.mark.asyncio
async def test_base_node_run_without_result():
    """Test base node run() when no result is produced."""

    class EmptyNode(ToolNode[SimpleInput, None]):  # type: ignore
        async def astream(self, input_data: SimpleInput):
            # Yield start but no result
            yield StreamStart(run_id="", node_id=self.name)
            # Don't yield StreamEnd or ToolResult

    def no_result(x: SimpleInput) -> None:
        return None

    node = EmptyNode(tool_func=no_result, name="empty")

    with pytest.raises(RuntimeError, match="did not produce a result"):
        await node.run(SimpleInput(value=1))
