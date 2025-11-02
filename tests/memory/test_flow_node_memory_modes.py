"""Tests for FlowNode memory mode functionality."""

from pydantic import BaseModel
from pydantic_ai.messages import ModelRequest
from pydantic_ai.messages import SystemPromptPart
import pytest

from pydantic_flow.memory import ConversationMemory
from pydantic_flow.memory import MemoryMode
from pydantic_flow.memory import ReadOnlyConversationMemory
from pydantic_flow.memory import ReadOnlyMemoryError
from pydantic_flow.memory import _active_flow_memory
from pydantic_flow.nodes import FlowNode
from pydantic_flow.streaming.core_events import FlowResult


class SimpleInput(BaseModel):
    """Simple input model for testing."""

    value: str


class SimpleOutput(BaseModel):
    """Simple output model for testing."""

    result: str


@pytest.mark.asyncio
async def test_flow_node_shared_memory_context():
    """Test SHARED mode sets sub-flow memory to parent memory."""
    parent_memory = ConversationMemory()
    parent_memory.append(ModelRequest(parts=[SystemPromptPart(content="parent")]))

    class MockFlow:
        async def astream(self, input_data):
            ctx_memory = _active_flow_memory.get()
            assert ctx_memory is parent_memory
            yield FlowResult(result=SimpleOutput(result="test"))

    flow_node = FlowNode[SimpleInput, SimpleOutput](
        flow=MockFlow(),  # type: ignore[arg-type]
        name="test",
        memory_mode=MemoryMode.SHARED,
    )

    token = _active_flow_memory.set(parent_memory)
    try:
        items = [item async for item in flow_node.astream(SimpleInput(value="test"))]
        assert len(items) > 0
    finally:
        _active_flow_memory.reset(token)


@pytest.mark.asyncio
async def test_flow_node_isolated_memory_context():
    """Test ISOLATED mode creates new memory for sub-flow."""
    parent_memory = ConversationMemory()
    parent_memory.append(ModelRequest(parts=[SystemPromptPart(content="parent")]))

    class MockFlow:
        async def astream(self, input_data):
            ctx_memory = _active_flow_memory.get()
            assert ctx_memory is not parent_memory
            assert len(ctx_memory) == 0  # type: ignore[arg-type]
            yield FlowResult(result=SimpleOutput(result="test"))

    flow_node = FlowNode[SimpleInput, SimpleOutput](
        flow=MockFlow(),  # type: ignore[arg-type]
        name="test",
        memory_mode=MemoryMode.ISOLATED,
        seed_isolated_memory=False,
    )

    token = _active_flow_memory.set(parent_memory)
    try:
        items = [item async for item in flow_node.astream(SimpleInput(value="test"))]
        assert len(items) > 0
    finally:
        _active_flow_memory.reset(token)


@pytest.mark.asyncio
async def test_flow_node_isolated_memory_with_seed():
    """Test ISOLATED mode with seeding copies parent memory."""
    parent_memory = ConversationMemory()
    parent_memory.append(ModelRequest(parts=[SystemPromptPart(content="parent")]))

    class MockFlow:
        async def astream(self, input_data):
            ctx_memory = _active_flow_memory.get()
            assert ctx_memory is not parent_memory
            assert len(ctx_memory) == 1  # type: ignore[arg-type]
            yield FlowResult(result=SimpleOutput(result="test"))

    flow_node = FlowNode[SimpleInput, SimpleOutput](
        flow=MockFlow(),  # type: ignore[arg-type]
        name="test",
        memory_mode=MemoryMode.ISOLATED,
        seed_isolated_memory=True,
    )

    token = _active_flow_memory.set(parent_memory)
    try:
        items = [item async for item in flow_node.astream(SimpleInput(value="test"))]
        assert len(items) > 0
    finally:
        _active_flow_memory.reset(token)


@pytest.mark.asyncio
async def test_flow_node_readonly_memory_context():
    """Test READONLY mode wraps parent memory in read-only wrapper."""
    parent_memory = ConversationMemory()
    parent_memory.append(ModelRequest(parts=[SystemPromptPart(content="parent")]))

    class MockFlow:
        async def astream(self, input_data):
            ctx_memory = _active_flow_memory.get()
            assert isinstance(ctx_memory, ReadOnlyConversationMemory)
            assert len(ctx_memory) == 1
            yield FlowResult(result=SimpleOutput(result="test"))

    flow_node = FlowNode[SimpleInput, SimpleOutput](
        flow=MockFlow(),  # type: ignore[arg-type]
        name="test",
        memory_mode=MemoryMode.READONLY,
    )

    token = _active_flow_memory.set(parent_memory)
    try:
        items = [item async for item in flow_node.astream(SimpleInput(value="test"))]
        assert len(items) > 0
    finally:
        _active_flow_memory.reset(token)


@pytest.mark.asyncio
async def test_flow_node_no_parent_memory():
    """Test FlowNode handles case when parent has no memory."""

    class MockFlow:
        async def astream(self, input_data):
            ctx_memory = _active_flow_memory.get()
            assert ctx_memory is None
            yield FlowResult(result=SimpleOutput(result="test"))

    flow_node = FlowNode[SimpleInput, SimpleOutput](
        flow=MockFlow(),  # type: ignore[arg-type]
        name="test",
        memory_mode=MemoryMode.SHARED,
    )

    items = [item async for item in flow_node.astream(SimpleInput(value="test"))]
    assert len(items) > 0


@pytest.mark.asyncio
async def test_readonly_memory_append_error():
    """Test ReadOnlyConversationMemory raises error on append."""
    mem = ConversationMemory()
    readonly = ReadOnlyConversationMemory(mem)

    with pytest.raises(ReadOnlyMemoryError):
        readonly.append(ModelRequest(parts=[SystemPromptPart(content="test")]))


@pytest.mark.asyncio
async def test_readonly_memory_extend_error():
    """Test ReadOnlyConversationMemory raises error on extend."""
    mem = ConversationMemory()
    readonly = ReadOnlyConversationMemory(mem)

    with pytest.raises(ReadOnlyMemoryError):
        readonly.extend([ModelRequest(parts=[SystemPromptPart(content="test")])])


@pytest.mark.asyncio
async def test_readonly_memory_clear_error():
    """Test ReadOnlyConversationMemory raises error on clear."""
    mem = ConversationMemory()
    readonly = ReadOnlyConversationMemory(mem)

    with pytest.raises(ReadOnlyMemoryError):
        readonly.clear()


def test_readonly_memory_read_operations():
    """Test ReadOnlyConversationMemory allows read operations."""
    mem = ConversationMemory()
    mem.append(ModelRequest(parts=[SystemPromptPart(content="test")]))

    readonly = ReadOnlyConversationMemory(mem)

    assert len(readonly) == 1
    assert len(readonly.get()) == 1

    copy = readonly.copy()
    assert len(copy) == 1
