"""Tests for streaming observers."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
import uuid

from pydantic_ai import Agent
import pytest

from pydantic_flow.streaming.events import NonFatalError
from pydantic_flow.streaming.events import StreamEnd
from pydantic_flow.streaming.events import StreamStart
from pydantic_flow.streaming.events import TokenChunk
from pydantic_flow.streaming.observers import observe_agent_stream
from pydantic_flow.streaming.observers import stream_agent_text


@pytest.mark.asyncio
async def test_observe_agent_stream_success():
    """Test observe_agent_stream emits correct progress items."""
    # Create a mock agent
    mock_agent = MagicMock(spec=Agent)

    # Create mock stream context manager
    mock_stream = AsyncMock()

    # Mock text chunks
    async def mock_stream_text():
        yield "Hello"
        yield " world"
        yield "!"

    mock_stream.stream_text = mock_stream_text

    # Mock final result with model_dump
    class MockResult:
        def model_dump(self):
            return {"text": "Hello world!"}

    mock_stream.get_output = AsyncMock(return_value=MockResult())

    # Mock run_stream as async context manager
    mock_agent.run_stream = MagicMock()
    mock_agent.run_stream.return_value.__aenter__ = AsyncMock(return_value=mock_stream)
    mock_agent.run_stream.return_value.__aexit__ = AsyncMock(return_value=None)

    # Collect progress items
    items = []
    async for item in observe_agent_stream(
        agent=mock_agent,
        prompt="Test prompt",
        message_history=None,
        run_id="test-run-123",
        node_id="test-node",
    ):
        items.append(item)

    # Verify items
    assert len(items) == 5  # Start + 3 tokens + End

    # Check StreamStart
    assert isinstance(items[0], StreamStart)
    assert items[0].run_id == "test-run-123"
    assert items[0].node_id == "test-node"
    assert "Test prompt" in str(items[0].input_preview)

    # Check TokenChunks
    assert isinstance(items[1], TokenChunk)
    assert items[1].text == "Hello"
    assert items[1].token_index == 0
    assert items[1].run_id == "test-run-123"

    assert isinstance(items[2], TokenChunk)
    assert items[2].text == " world"
    assert items[2].token_index == 1

    assert isinstance(items[3], TokenChunk)
    assert items[3].text == "!"
    assert items[3].token_index == 2

    # Check StreamEnd
    assert isinstance(items[4], StreamEnd)
    assert items[4].run_id == "test-run-123"
    assert items[4].node_id == "test-node"
    assert items[4].result_preview == {"text": "Hello world!"}


@pytest.mark.asyncio
async def test_observe_agent_stream_generates_run_id():
    """Test observe_agent_stream generates run_id if not provided."""
    mock_agent = MagicMock(spec=Agent)

    mock_stream = AsyncMock()

    async def mock_stream_text():
        yield "test"

    mock_stream.stream_text = mock_stream_text
    mock_stream.get_output = AsyncMock(return_value="result")

    mock_agent.run_stream = MagicMock()
    mock_agent.run_stream.return_value.__aenter__ = AsyncMock(return_value=mock_stream)
    mock_agent.run_stream.return_value.__aexit__ = AsyncMock(return_value=None)

    items = []
    async for item in observe_agent_stream(
        agent=mock_agent,
        prompt="Test",
        message_history=None,
    ):
        items.append(item)

    # All items should have same run_id
    run_ids = {item.run_id for item in items}
    assert len(run_ids) == 1

    # Run ID should be a valid UUID
    run_id = run_ids.pop()
    try:
        uuid.UUID(run_id)
    except ValueError:
        pytest.fail(f"Generated run_id '{run_id}' is not a valid UUID")


@pytest.mark.asyncio
async def test_observe_agent_stream_with_message_history():
    """Test observe_agent_stream passes message history correctly."""
    mock_agent = MagicMock(spec=Agent)

    mock_stream = AsyncMock()

    async def mock_stream_text():
        yield "response"

    mock_stream.stream_text = mock_stream_text
    mock_stream.get_output = AsyncMock(return_value="result")

    mock_agent.run_stream = MagicMock()
    mock_agent.run_stream.return_value.__aenter__ = AsyncMock(return_value=mock_stream)
    mock_agent.run_stream.return_value.__aexit__ = AsyncMock(return_value=None)

    message_history = [{"role": "user", "content": "previous message"}]

    items = []
    async for item in observe_agent_stream(
        agent=mock_agent,
        prompt="New prompt",
        message_history=message_history,
    ):
        items.append(item)

    # Verify run_stream was called with message_history
    mock_agent.run_stream.assert_called_once_with(
        "New prompt", message_history=message_history
    )


@pytest.mark.asyncio
async def test_observe_agent_stream_result_without_model_dump():
    """Test observe_agent_stream handles results without model_dump."""
    mock_agent = MagicMock(spec=Agent)

    mock_stream = AsyncMock()

    async def mock_stream_text():
        yield "test"

    mock_stream.stream_text = mock_stream_text
    # Return a plain string result
    mock_stream.get_output = AsyncMock(return_value="Plain string result")

    mock_agent.run_stream = MagicMock()
    mock_agent.run_stream.return_value.__aenter__ = AsyncMock(return_value=mock_stream)
    mock_agent.run_stream.return_value.__aexit__ = AsyncMock(return_value=None)

    items = []
    async for item in observe_agent_stream(
        agent=mock_agent,
        prompt="Test",
        message_history=None,
    ):
        items.append(item)

    # Check final item is StreamEnd with value preview
    assert isinstance(items[-1], StreamEnd)
    assert items[-1].result_preview == {"value": "Plain string result"}


@pytest.mark.asyncio
async def test_observe_agent_stream_error_handling():
    """Test observe_agent_stream emits error and re-raises."""
    mock_agent = MagicMock(spec=Agent)

    # Make run_stream raise an exception
    mock_agent.run_stream.side_effect = ValueError("Test error")

    items = []
    with pytest.raises(ValueError, match="Test error"):
        async for item in observe_agent_stream(
            agent=mock_agent,
            prompt="Test",
            message_history=None,
            run_id="error-run",
            node_id="error-node",
        ):
            items.append(item)

    # Should have emitted StreamStart and NonFatalError
    assert len(items) == 2

    assert isinstance(items[0], StreamStart)
    assert items[0].run_id == "error-run"

    assert isinstance(items[1], NonFatalError)
    assert items[1].run_id == "error-run"
    assert items[1].node_id == "error-node"
    assert "Test error" in items[1].message
    assert items[1].recoverable is False


@pytest.mark.asyncio
async def test_stream_agent_text_simple():
    """Test stream_agent_text yields text chunks directly."""
    mock_agent = MagicMock(spec=Agent)

    mock_stream = AsyncMock()

    async def mock_stream_text():
        yield "chunk1"
        yield "chunk2"
        yield "chunk3"

    mock_stream.stream_text = mock_stream_text

    mock_agent.run_stream = MagicMock()
    mock_agent.run_stream.return_value.__aenter__ = AsyncMock(return_value=mock_stream)
    mock_agent.run_stream.return_value.__aexit__ = AsyncMock(return_value=None)

    # Collect chunks
    chunks = []
    async for chunk in stream_agent_text(
        agent=mock_agent,
        prompt="Test prompt",
        message_history=None,
    ):
        chunks.append(chunk)

    assert chunks == ["chunk1", "chunk2", "chunk3"]


@pytest.mark.asyncio
async def test_stream_agent_text_with_message_history():
    """Test stream_agent_text passes message history correctly."""
    mock_agent = MagicMock(spec=Agent)

    mock_stream = AsyncMock()

    async def mock_stream_text():
        yield "response"

    mock_stream.stream_text = mock_stream_text

    mock_agent.run_stream = MagicMock()
    mock_agent.run_stream.return_value.__aenter__ = AsyncMock(return_value=mock_stream)
    mock_agent.run_stream.return_value.__aexit__ = AsyncMock(return_value=None)

    message_history = [{"role": "user", "content": "previous"}]

    chunks = []
    async for chunk in stream_agent_text(
        agent=mock_agent,
        prompt="New prompt",
        message_history=message_history,
    ):
        chunks.append(chunk)

    # Verify run_stream was called with message_history
    mock_agent.run_stream.assert_called_once_with(
        "New prompt", message_history=message_history
    )
