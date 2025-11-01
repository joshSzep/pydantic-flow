"""Tests for MergePromptNode implementation."""

from collections.abc import AsyncIterator

from pydantic import BaseModel
import pytest

from pydantic_flow.hitl.decisions import InterruptDecision
from pydantic_flow.hitl.interrupts import InterruptionRequested
from pydantic_flow.nodes import MergePromptNode
from pydantic_flow.nodes.base import BaseNode
from pydantic_flow.nodes.prompt import PromptConfig
from pydantic_flow.streaming.base import ProgressItem
from pydantic_flow.streaming.base import ProgressType
from pydantic_flow.streaming.core_events import StreamEnd
from pydantic_flow.streaming.core_events import StreamStart


class DataA(BaseModel):
    """First input type."""

    value_a: str


class DataB(BaseModel):
    """Second input type."""

    value_b: int


class NodeA(BaseNode[BaseModel, DataA]):
    """Node that produces DataA."""

    async def astream(self, input_data: BaseModel) -> AsyncIterator[ProgressItem]:
        """Stream DataA output."""
        yield StreamStart(run_id=self.run_id or "", node_id=self.name)
        result = DataA(value_a="test_a")
        yield StreamEnd(
            run_id=self.run_id or "",
            node_id=self.name,
            result_preview=result.model_dump(),
        )


class NodeB(BaseNode[BaseModel, DataB]):
    """Node that produces DataB."""

    async def astream(self, input_data: BaseModel) -> AsyncIterator[ProgressItem]:
        """Stream DataB output."""
        yield StreamStart(run_id=self.run_id or "", node_id=self.name)
        result = DataB(value_b=42)
        yield StreamEnd(
            run_id=self.run_id or "",
            node_id=self.name,
            result_preview=result.model_dump(),
        )


@pytest.mark.asyncio
async def test_merge_prompt_node_initialization():
    """Test that MergePromptNode can be initialized properly."""
    node_a = NodeA(name="node_a")
    node_b = NodeB(name="node_b")

    merge_node = MergePromptNode[DataA, DataB, str](
        prompt="Combine {0} and {1}",
        inputs=(node_a.output, node_b.output),
        name="merge_prompt",
    )

    assert merge_node.name == "merge_prompt"
    assert merge_node.prompt == "Combine {0} and {1}"
    assert len(merge_node.inputs) == 2
    assert merge_node._agent is not None


@pytest.mark.asyncio
async def test_merge_prompt_node_format_prompt_positional():
    """Test prompt formatting with positional arguments."""
    node_a = NodeA(name="node_a")
    node_b = NodeB(name="node_b")

    merge_node = MergePromptNode[DataA, DataB, str](
        prompt="First: {0}, Second: {1}",
        inputs=(node_a.output, node_b.output),
        name="merge_prompt",
    )

    data_a = DataA(value_a="hello")
    data_b = DataB(value_b=123)
    input_data = (data_a, data_b)

    formatted = merge_node._format_prompt(input_data)
    assert "hello" in formatted
    assert "123" in formatted


@pytest.mark.asyncio
async def test_merge_prompt_node_format_prompt_with_model_dump():
    """Test prompt formatting using model_dump fields."""
    node_a = NodeA(name="node_a")
    node_b = NodeB(name="node_b")

    merge_node = MergePromptNode[DataA, DataB, str](
        prompt="A: {value_a}, B: {value_b}",
        inputs=(node_a.output, node_b.output),
        name="merge_prompt",
    )

    data_a = DataA(value_a="world")
    data_b = DataB(value_b=456)
    input_data = (data_a, data_b)

    formatted = merge_node._format_prompt(input_data)
    assert "world" in formatted
    assert "456" in formatted


@pytest.mark.asyncio
async def test_merge_prompt_node_astream_emits_start():
    """Test that MergePromptNode emits StreamStart."""
    node_a = NodeA(name="node_a")
    node_b = NodeB(name="node_b")

    merge_node = MergePromptNode[DataA, DataB, str](
        prompt="Test prompt: {0} and {1}",
        inputs=(node_a.output, node_b.output),
        config=PromptConfig(model="test"),
        name="merge_prompt",
    )

    data_a = DataA(value_a="test")
    data_b = DataB(value_b=99)
    input_data = (data_a, data_b)

    items = []
    async for item in merge_node.astream(input_data):
        items.append(item)
        # Break after a few items to avoid running the full agent
        if len(items) >= 2:
            break

    # Should have at least StreamStart
    assert len(items) >= 1
    assert items[0].type == ProgressType.START
    assert items[0].node_id == "merge_prompt"
    assert items[0].input_preview is not None
    assert "prompt" in items[0].input_preview
    assert "num_inputs" in items[0].input_preview
    assert items[0].input_preview["num_inputs"] == 2


@pytest.mark.asyncio
async def test_merge_prompt_node_with_config():
    """Test MergePromptNode with custom PromptConfig."""
    node_a = NodeA(name="node_a")
    node_b = NodeB(name="node_b")

    config = PromptConfig(
        model="test",
        system_prompt="You are a helpful assistant.",
    )

    merge_node = MergePromptNode[DataA, DataB, str](
        prompt="Merge: {0} and {1}",
        inputs=(node_a.output, node_b.output),
        config=config,
        name="merge_prompt",
    )

    assert merge_node.config == config
    assert merge_node.model == "test"


@pytest.mark.asyncio
async def test_merge_prompt_node_dependencies():
    """Test that MergePromptNode tracks dependencies correctly."""
    node_a = NodeA(name="node_a")
    node_b = NodeB(name="node_b")

    merge_node = MergePromptNode[DataA, DataB, str](
        prompt="Combine {0} and {1}",
        inputs=(node_a.output, node_b.output),
        name="merge_prompt",
    )

    deps = merge_node.dependencies
    assert len(deps) == 2
    assert node_a in deps
    assert node_b in deps


@pytest.mark.asyncio
async def test_merge_prompt_node_format_prompt_fallback():
    """Test that prompt formatting falls back to concatenation."""
    node_a = NodeA(name="node_a")
    node_b = NodeB(name="node_b")

    # Use a prompt that won't match any formatting strategy
    merge_node = MergePromptNode[DataA, DataB, str](
        prompt="Fixed prompt without placeholders: {}",
        inputs=(node_a.output, node_b.output),
        name="merge_prompt",
    )

    data_a = DataA(value_a="alpha")
    data_b = DataB(value_b=789)
    input_data = (data_a, data_b)

    formatted = merge_node._format_prompt(input_data)
    # Should contain the concatenated string representation
    assert "Fixed prompt" in formatted or "alpha" in formatted


@pytest.mark.asyncio
async def test_merge_prompt_node_explicit_model():
    """Test MergePromptNode with explicit model parameter."""
    node_a = NodeA(name="node_a")
    node_b = NodeB(name="node_b")

    merge_node = MergePromptNode[DataA, DataB, str](
        prompt="Test: {0} {1}",
        inputs=(node_a.output, node_b.output),
        model="test",
        name="merge_prompt",
    )

    assert merge_node.model == "test"


@pytest.mark.asyncio
async def test_merge_prompt_node_with_result_type():
    """Test MergePromptNode with result_type in config."""
    node_a = NodeA(name="node_a")
    node_b = NodeB(name="node_b")

    class OutputModel(BaseModel):
        summary: str

    config = PromptConfig(
        model="test",
        result_type=OutputModel,
    )

    merge_node = MergePromptNode[DataA, DataB, OutputModel](
        prompt="Summarize: {0} and {1}",
        inputs=(node_a.output, node_b.output),
        config=config,
        name="merge_prompt",
    )

    assert merge_node._agent is not None
    assert merge_node.config.result_type == OutputModel


@pytest.mark.asyncio
async def test_merge_prompt_node_format_indexed_kwargs():
    """Test prompt formatting with indexed keyword arguments."""
    node_a = NodeA(name="node_a")
    node_b = NodeB(name="node_b")

    # Use template that requires indexed keywords
    merge_node = MergePromptNode[DataA, DataB, str](
        prompt="Item 0: {0}\nItem 1: {1}",
        inputs=(node_a.output, node_b.output),
        name="merge_prompt",
    )

    # Use plain strings instead of models to trigger indexed kwargs path
    input_data = ("first", "second")

    formatted = merge_node._format_prompt(input_data)
    assert "first" in formatted
    assert "second" in formatted


@pytest.mark.asyncio
async def test_merge_prompt_node_format_mixed_types():
    """Test prompt formatting with mixed types (model and non-model)."""
    node_a = NodeA(name="node_a")
    node_b = NodeB(name="node_b")

    merge_node = MergePromptNode[DataA, DataB, str](
        prompt="Data: {value_a} and plain: {1}",
        inputs=(node_a.output, node_b.output),
        name="merge_prompt",
    )

    data_a = DataA(value_a="from_model")
    plain_string = "plain_text"
    input_data = (data_a, plain_string)

    formatted = merge_node._format_prompt(input_data)
    # Mixed model_dump + indexed merge should work
    assert "from_model" in formatted
    assert "plain_text" in formatted


@pytest.mark.asyncio
async def test_merge_prompt_node_create_checkpoint():
    """Test checkpoint creation."""
    node_a = NodeA(name="node_a")
    node_b = NodeB(name="node_b")

    merge_node = MergePromptNode[DataA, DataB, str](
        prompt="Test",
        inputs=(node_a.output, node_b.output),
        name="merge_checkpoint_test",
    )

    start_item = StreamStart(
        node_id="merge_checkpoint_test",
    )

    checkpoint = merge_node._create_checkpoint(start_item)

    assert checkpoint.interrupted_node_id == "merge_checkpoint_test"
    # When StreamStart has no run_id, a new one is generated
    assert checkpoint.run_id != ""
    assert checkpoint.wave_number == 0
    assert checkpoint.reason == "hitl_interrupt"


@pytest.mark.asyncio
async def test_merge_prompt_node_interrupt_handler():
    """Test MergePromptNode with interrupt handler."""
    node_a = NodeA(name="node_a")
    node_b = NodeB(name="node_b")

    merge_node = MergePromptNode[DataA, DataB, str](
        prompt="Test: {0} {1}",
        inputs=(node_a.output, node_b.output),
        model="test",
        name="merge_prompt",
    )

    # Register an interrupt handler that always interrupts at start
    async def interrupt_at_start(item: ProgressItem) -> InterruptDecision:
        if item.type == ProgressType.START:
            return InterruptDecision.interrupt("Testing interrupt")
        return InterruptDecision.proceed()

    merge_node.register_interrupt_handler(interrupt_at_start)

    data_a = DataA(value_a="test")
    data_b = DataB(value_b=42)

    with pytest.raises(InterruptionRequested) as exc_info:
        async for _ in merge_node.astream((data_a, data_b)):
            pass

    exception = exc_info.value
    assert isinstance(exception, InterruptionRequested)
    assert exception.decision.should_interrupt
    assert "Testing interrupt" in exception.decision.reason


@pytest.mark.asyncio
async def test_merge_prompt_node_exception_handling():
    """Test MergePromptNode exception handling during streaming."""
    node_a = NodeA(name="node_a")
    node_b = NodeB(name="node_b")

    # Create a node with an invalid model that will cause agent errors
    merge_node = MergePromptNode[DataA, DataB, str](
        prompt="Test: {0} {1}",
        inputs=(node_a.output, node_b.output),
        model="test",
        name="merge_prompt",
    )

    # Patch the agent to raise an exception
    original_agent = merge_node._agent

    class FailingAgent:
        def run_stream(self, *args, **kwargs):
            raise RuntimeError("Simulated agent failure")

    merge_node._agent = FailingAgent()  # type: ignore

    data_a = DataA(value_a="test")
    data_b = DataB(value_b=42)

    items = []
    with pytest.raises(RuntimeError):
        async for item in merge_node.astream((data_a, data_b)):
            items.append(item)

    # Should have emitted at least StreamStart and NonFatalError
    assert len(items) >= 2
    assert items[0].type == ProgressType.START
    # Last item before exception should be error
    assert any(item.type == "error" for item in items)

    # Restore original agent
    merge_node._agent = original_agent
