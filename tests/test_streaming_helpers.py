"""Tests for streaming helper functions."""

import pytest

from pydantic_flow.streaming.events import PartialFields
from pydantic_flow.streaming.events import StreamEnd
from pydantic_flow.streaming.events import StreamStart
from pydantic_flow.streaming.events import TokenChunk
from pydantic_flow.streaming.events import ToolCall
from pydantic_flow.streaming.helpers import collect_all_tokens
from pydantic_flow.streaming.helpers import collect_final_result
from pydantic_flow.streaming.helpers import iter_fields
from pydantic_flow.streaming.helpers import iter_tokens


@pytest.mark.asyncio
async def test_iter_tokens():
    """Test iter_tokens extracts only token events."""

    async def mock_stream():
        yield StreamStart(run_id="1", node_id="test")
        yield TokenChunk(run_id="1", node_id="test", text="Hello")
        yield TokenChunk(run_id="1", node_id="test", text=" ")
        yield TokenChunk(run_id="1", node_id="test", text="World")
        yield ToolCall(run_id="1", node_id="test", tool_name="test", call_id="1")
        yield StreamEnd(run_id="1", node_id="test")

    tokens = []
    async for token in iter_tokens(mock_stream()):
        tokens.append(token)

    assert len(tokens) == 3
    assert tokens[0] == "Hello"
    assert tokens[1] == " "
    assert tokens[2] == "World"


@pytest.mark.asyncio
async def test_iter_fields():
    """Test iter_fields extracts only field events."""

    async def mock_stream():
        yield StreamStart(run_id="1", node_id="test")
        yield PartialFields(run_id="1", node_id="test", fields={"name": "Alice"})
        yield TokenChunk(run_id="1", node_id="test", text="test")
        yield PartialFields(run_id="1", node_id="test", fields={"age": 30})
        yield StreamEnd(run_id="1", node_id="test")

    fields = []
    async for field_event in iter_fields(mock_stream()):
        fields.append(field_event)

    assert len(fields) == 2
    assert fields[0] == {"name": "Alice"}
    assert fields[1] == {"age": 30}


@pytest.mark.asyncio
async def test_collect_all_tokens():
    """Test collect_all_tokens concatenates all token text."""

    async def mock_stream():
        yield StreamStart(run_id="1", node_id="test")
        yield TokenChunk(run_id="1", node_id="test", text="The ")
        yield TokenChunk(run_id="1", node_id="test", text="quick ")
        yield TokenChunk(run_id="1", node_id="test", text="brown ")
        yield TokenChunk(run_id="1", node_id="test", text="fox")
        yield StreamEnd(run_id="1", node_id="test")

    result = await collect_all_tokens(mock_stream())
    assert result == "The quick brown fox"


@pytest.mark.asyncio
async def test_collect_all_tokens_empty():
    """Test collect_all_tokens with no tokens."""

    async def mock_stream():
        yield StreamStart(run_id="1", node_id="test")
        yield StreamEnd(run_id="1", node_id="test")

    result = await collect_all_tokens(mock_stream())
    assert result == ""


@pytest.mark.asyncio
async def test_collect_final_result():
    """Test collect_final_result extracts StreamEnd result_preview."""

    async def mock_stream():
        yield StreamStart(run_id="1", node_id="test")
        yield TokenChunk(run_id="1", node_id="test", text="processing")
        yield StreamEnd(
            run_id="1", node_id="test", result_preview={"status": "complete"}
        )

    result = await collect_final_result(mock_stream())
    assert result == {"status": "complete"}


@pytest.mark.asyncio
async def test_collect_final_result_none():
    """Test collect_final_result with no result_preview."""

    async def mock_stream():
        yield StreamStart(run_id="1", node_id="test")
        yield StreamEnd(run_id="1", node_id="test", result_preview=None)

    result = await collect_final_result(mock_stream())
    assert result is None
