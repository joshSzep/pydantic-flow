"""Tests for automatic message capture in conversation memory."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

from pydantic import BaseModel
import pytest

from pydantic_flow.flow.flow import Flow
from pydantic_flow.memory.config import MemoryConfig
from pydantic_flow.memory.memory import ConversationMemory
from pydantic_flow.memory.memory import _active_flow_memory
from pydantic_flow.nodes.agent import AgentNode
from pydantic_flow.streaming.events import StreamEnd
from pydantic_flow.streaming.events import StreamStart
from pydantic_flow.streaming.events import TokenChunk
from pydantic_flow.streaming.observers import observe_agent_stream


class SimpleInput(BaseModel):
    """Test input."""

    text: str


class SimpleOutput(BaseModel):
    """Test output."""

    response: str


@pytest.mark.asyncio
async def test_observe_agent_stream_captures_messages():
    """Test that observe_agent_stream automatically captures messages."""
    # Create a mock agent
    mock_agent = MagicMock()

    # Create mock stream result
    mock_stream = AsyncMock()

    # Mock text chunks as async generator
    async def mock_stream_text():
        yield "Hello"
        yield " world"

    mock_stream.stream_text = mock_stream_text
    mock_stream.get_output = AsyncMock(return_value="Hello world")

    # Mock messages (just need objects that extend() can handle)
    mock_messages = [MagicMock(), MagicMock()]
    mock_stream.new_messages = MagicMock(return_value=mock_messages)

    mock_agent.run_stream = MagicMock(return_value=mock_stream)
    mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
    mock_stream.__aexit__ = AsyncMock(return_value=None)

    # Create and set active memory
    memory = ConversationMemory()
    token = _active_flow_memory.set(memory)

    try:
        # Call observe_agent_stream with empty message_history (signals enabled)
        items = []
        async for item in observe_agent_stream(
            mock_agent,
            "test prompt",
            message_history=[],
            run_id="test-run",
            node_id="test-node",
        ):
            items.append(item)

        # Verify stream events were emitted
        assert len(items) == 4  # Start + 2 tokens + End
        assert isinstance(items[0], StreamStart)
        assert isinstance(items[1], TokenChunk)
        assert isinstance(items[2], TokenChunk)
        assert isinstance(items[3], StreamEnd)

        # Verify messages were captured in memory
        captured_messages = memory.get()
        assert len(captured_messages) == 2
        # Verify these are the same mock objects we provided
        assert captured_messages[0] is mock_messages[0]
        assert captured_messages[1] is mock_messages[1]

    finally:
        _active_flow_memory.reset(token)


@pytest.mark.asyncio
async def test_observe_agent_stream_no_memory_context():
    """Test observe_agent_stream works without active memory."""
    # Create a mock agent
    mock_agent = MagicMock()

    # Create mock stream result
    mock_stream = AsyncMock()

    async def mock_stream_text():
        yield "test"

    mock_stream.stream_text = mock_stream_text
    mock_stream.get_output = AsyncMock(return_value="test")
    mock_stream.new_messages = MagicMock(return_value=[MagicMock()])

    mock_agent.run_stream = MagicMock(return_value=mock_stream)
    mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
    mock_stream.__aexit__ = AsyncMock(return_value=None)

    # No active memory set
    items = []
    async for item in observe_agent_stream(
        mock_agent, "test", run_id="test", node_id="test"
    ):
        items.append(item)

    # Should complete without error
    assert len(items) == 3  # Start + token + End


@pytest.mark.asyncio
async def test_observe_agent_stream_capture_error_handled():
    """Test that errors during message capture don't break the stream."""
    # Create a mock agent
    mock_agent = MagicMock()

    # Create mock stream result that raises on new_messages()
    mock_stream = AsyncMock()

    async def mock_stream_text():
        yield "test"

    mock_stream.stream_text = mock_stream_text
    mock_stream.get_output = AsyncMock(return_value="test")
    mock_stream.new_messages = MagicMock(side_effect=Exception("Capture failed"))

    mock_agent.run_stream = MagicMock(return_value=mock_stream)
    mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
    mock_stream.__aexit__ = AsyncMock(return_value=None)

    # Create and set active memory
    memory = ConversationMemory()
    token = _active_flow_memory.set(memory)

    try:
        # Call observe_agent_stream - should not raise
        items = []
        async for item in observe_agent_stream(
            mock_agent, "test", run_id="test", node_id="test"
        ):
            items.append(item)

        # Should complete successfully despite capture error
        assert len(items) == 3

        # Memory should be empty (capture failed but didn't break flow)
        assert len(memory.get()) == 0

    finally:
        _active_flow_memory.reset(token)


@pytest.mark.asyncio
async def test_agent_node_auto_capture_integration():
    """Test that AgentNode automatically captures messages via observe_agent_stream."""
    # Create a simple flow with memory enabled
    flow = Flow[SimpleInput, SimpleOutput](
        input_type=SimpleInput,
        output_type=SimpleOutput,
        memory_config=MemoryConfig(enable_conversation_memory=True),
    )

    # Create mock agent
    mock_agent = MagicMock()

    # Create mock stream result
    mock_stream = AsyncMock()

    async def mock_stream_text():
        yield "Response"

    mock_stream.stream_text = mock_stream_text
    mock_stream.get_output = AsyncMock(return_value="Response text")
    mock_messages = [MagicMock(), MagicMock()]
    mock_stream.new_messages = MagicMock(return_value=mock_messages)

    mock_agent.run_stream = MagicMock(return_value=mock_stream)
    mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
    mock_stream.__aexit__ = AsyncMock(return_value=None)

    # Create agent node - AgentNode returns the raw agent output
    node = AgentNode[SimpleInput, str](
        agent=mock_agent,
        prompt_template="Process: {text}",
        name="response",
        use_conversation_memory=True,
    )

    flow.add_nodes(node)
    flow.compile()

    # Set memory context manually (normally done by Flow.run())
    token = _active_flow_memory.set(flow._conversation_memory)
    try:
        # Test memory capture by running the node directly
        items = []
        async for item in node.astream(SimpleInput(text="Test input")):
            items.append(item)

        # Verify messages were captured in flow memory
        assert flow._conversation_memory is not None
        captured_messages = flow._conversation_memory.get()
        assert len(captured_messages) == 2
    finally:
        _active_flow_memory.reset(token)

    # Verify memory was populated
    assert flow._conversation_memory is not None
    captured_messages = flow._conversation_memory.get()
    assert len(captured_messages) == 2


@pytest.mark.asyncio
async def test_agent_node_no_capture_when_memory_disabled():
    """Test that AgentNode doesn't capture when use_conversation_memory=False."""
    # Create mock agent
    mock_agent = MagicMock()
    mock_stream = AsyncMock()

    async def mock_stream_text():
        yield "Response"

    mock_stream.stream_text = mock_stream_text
    mock_stream.get_output = AsyncMock(return_value="Response")
    mock_stream.new_messages = MagicMock(return_value=[MagicMock()])

    mock_agent.run_stream = MagicMock(return_value=mock_stream)
    mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
    mock_stream.__aexit__ = AsyncMock(return_value=None)

    # Create agent node with memory disabled
    node = AgentNode[SimpleInput, str](
        agent=mock_agent,
        name="response",
        use_conversation_memory=False,  # Disabled
    )

    # Create a memory and set it in context
    memory = ConversationMemory()
    token = _active_flow_memory.set(memory)

    try:
        # Run the node
        items = []
        async for item in node.astream(SimpleInput(text="Test")):
            items.append(item)

        # Verify memory is still empty (node didn't use it)
        assert len(memory.get()) == 0

        # Verify agent was called with no message_history
        mock_agent.run_stream.assert_called_once()
        call_args = mock_agent.run_stream.call_args
        assert call_args.kwargs.get("message_history") is None
    finally:
        _active_flow_memory.reset(token)
