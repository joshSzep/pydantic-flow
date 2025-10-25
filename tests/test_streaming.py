"""Tests for streaming infrastructure."""

from pydantic import BaseModel
import pytest

from pydantic_flow.streaming.events import ProgressType
from pydantic_flow.streaming.events import StreamEnd
from pydantic_flow.streaming.events import StreamStart
from pydantic_flow.streaming.events import TokenChunk
from pydantic_flow.streaming.helpers import collect_all_tokens
from pydantic_flow.streaming.helpers import collect_final_result
from pydantic_flow.streaming.helpers import iter_tokens


class DummyInput(BaseModel):
    """Test input model."""

    value: str


class DummyOutput(BaseModel):
    """Test output model."""

    result: str


async def dummy_stream():
    """Create a simple test stream."""
    yield StreamStart(run_id="test", node_id="test_node")
    yield TokenChunk(text="Hello", run_id="test", node_id="test_node")
    yield TokenChunk(text=" ", run_id="test", node_id="test_node")
    yield TokenChunk(text="World", run_id="test", node_id="test_node")
    yield StreamEnd(
        run_id="test",
        node_id="test_node",
        result_preview={"result": "Hello World"},
    )


@pytest.mark.asyncio
async def test_stream_start_end():
    """Test that streams have coherent start and end markers."""
    items = []
    async for item in dummy_stream():
        items.append(item)

    # First item should be StreamStart
    assert items[0].type == ProgressType.START
    assert items[0].node_id == "test_node"

    # Last item should be StreamEnd
    assert items[-1].type == ProgressType.END
    assert items[-1].node_id == "test_node"


@pytest.mark.asyncio
async def test_iter_tokens():
    """Test token extraction helper."""
    tokens = []
    async for token in iter_tokens(dummy_stream()):
        tokens.append(token)

    assert tokens == ["Hello", " ", "World"]


@pytest.mark.asyncio
async def test_collect_all_tokens():
    """Test collecting all tokens into a string."""
    full_text = await collect_all_tokens(dummy_stream())
    assert full_text == "Hello World"


@pytest.mark.asyncio
async def test_collect_final_result():
    """Test extracting final result from stream."""
    result = await collect_final_result(dummy_stream())
    assert result == {"result": "Hello World"}


@pytest.mark.asyncio
async def test_progress_item_timestamps():
    """Test that progress items have timestamps."""
    async for item in dummy_stream():
        assert item.timestamp is not None
        assert item.run_id == "test"
        assert item.node_id == "test_node"


@pytest.mark.asyncio
async def test_token_ordering():
    """Test that tokens maintain proper ordering."""
    token_texts = []
    token_indices = []

    stream = dummy_stream()
    async for item in stream:
        if isinstance(item, TokenChunk):
            token_texts.append(item.text)
            if item.token_index is not None:
                token_indices.append(item.token_index)

    # Tokens should be in order
    assert token_texts == ["Hello", " ", "World"]
