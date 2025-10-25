"""Comprehensive tests for AgentNode and LLMNode with mocked pydantic-ai."""

from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from pydantic import BaseModel
from pydantic_ai import Agent
import pytest

from pydantic_flow.nodes.agent import AgentNode
from pydantic_flow.nodes.agent import LLMNode
from pydantic_flow.streaming.events import NonFatalError
from pydantic_flow.streaming.events import StreamEnd
from pydantic_flow.streaming.events import StreamStart
from pydantic_flow.streaming.events import TokenChunk


class InputModel(BaseModel):
    """Test input model."""

    query: str
    context: str | None = None


class OutputModel(BaseModel):
    """Test structured output model."""

    answer: str
    confidence: float


@pytest.mark.asyncio
async def test_agent_node_basic_streaming():
    """Test AgentNode streams tokens via observe_agent_stream."""
    # Create a mock agent
    mock_agent = MagicMock(spec=Agent)

    # Create node with prompt template
    node = AgentNode[InputModel, str](
        agent=mock_agent,
        prompt_template="Query: {query}",
        name="test_agent",
        run_id="test-run-123",
    )

    # Mock the observe_agent_stream function
    mock_items = [
        StreamStart(
            run_id="test-run-123",
            node_id="test_agent",
            input_preview={"prompt": "Query: test"},
        ),
        TokenChunk(
            text="Hello", token_index=0, run_id="test-run-123", node_id="test_agent"
        ),
        TokenChunk(
            text=" ", token_index=1, run_id="test-run-123", node_id="test_agent"
        ),
        TokenChunk(
            text="world", token_index=2, run_id="test-run-123", node_id="test_agent"
        ),
        StreamEnd(
            run_id="test-run-123",
            node_id="test_agent",
            result_preview={"value": "Hello world"},
        ),
    ]

    with patch("pydantic_flow.nodes.agent.observe_agent_stream") as mock_observe:
        # Make observe_agent_stream return our mock items
        async def mock_stream(*args, **kwargs):
            for item in mock_items:
                yield item

        mock_observe.return_value = mock_stream()

        # Run the node
        input_data = InputModel(query="test")
        items = []
        async for item in node.astream(input_data):
            items.append(item)

        # Verify we got all the items
        assert len(items) == 5
        assert isinstance(items[0], StreamStart)
        assert isinstance(items[1], TokenChunk)
        assert items[1].text == "Hello"
        assert isinstance(items[-1], StreamEnd)

        # Verify observe_agent_stream was called with correct args
        mock_observe.assert_called_once()
        call_args = mock_observe.call_args
        assert call_args.args[0] == mock_agent
        assert call_args.args[1] == "Query: test"
        assert call_args.kwargs["run_id"] == "test-run-123"
        assert call_args.kwargs["node_id"] == "test_agent"


@pytest.mark.asyncio
async def test_agent_node_format_prompt_with_template():
    """Test _format_prompt uses template with input fields."""
    mock_agent = MagicMock(spec=Agent)
    node = AgentNode[InputModel, str](
        agent=mock_agent,
        prompt_template="Query: {query}, Context: {context}",
    )

    input_data = InputModel(query="What is AI?", context="machine learning")
    prompt = node._format_prompt(input_data)

    assert prompt == "Query: What is AI?, Context: machine learning"


@pytest.mark.asyncio
async def test_agent_node_format_prompt_without_template():
    """Test _format_prompt without template uses model_dump_json."""
    mock_agent = MagicMock(spec=Agent)
    node = AgentNode[InputModel, str](
        agent=mock_agent,
        prompt_template=None,  # No template
    )

    input_data = InputModel(query="test query", context="some context")
    prompt = node._format_prompt(input_data)

    # Should return JSON representation
    assert "test query" in prompt
    assert "some context" in prompt
    # Verify it's valid JSON-ish
    assert "{" in prompt


@pytest.mark.asyncio
async def test_agent_node_format_prompt_no_template_no_model_dump_json():
    """Test _format_prompt fallback to str() when no model_dump_json."""

    class SimpleInput:
        def __str__(self):
            return "simple string representation"

    mock_agent = MagicMock(spec=Agent)
    node = AgentNode[Any, str](agent=mock_agent, prompt_template=None)

    input_data = SimpleInput()
    prompt = node._format_prompt(input_data)

    assert prompt == "simple string representation"


@pytest.mark.asyncio
async def test_agent_node_generates_run_id_if_none():
    """Test that AgentNode generates a run_id if not provided."""
    mock_agent = MagicMock(spec=Agent)
    node = AgentNode[InputModel, str](
        agent=mock_agent,
        prompt_template="Query: {query}",
        name="test_agent",
        # No run_id provided
    )

    with patch("pydantic_flow.nodes.agent.observe_agent_stream") as mock_observe:
        # Mock items to return
        async def mock_stream(*args, **kwargs):
            yield StreamStart(
                run_id="gen-id",
                node_id="test_agent",
                input_preview={},
            )
            yield StreamEnd(
                run_id="gen-id",
                node_id="test_agent",
                result_preview={},
            )

        mock_observe.return_value = mock_stream()

        input_data = InputModel(query="test")
        items = []
        async for item in node.astream(input_data):
            items.append(item)

        # Verify observe was called with a generated run_id
        mock_observe.assert_called_once()
        call_kwargs = mock_observe.call_args.kwargs
        # run_id should be generated (not None)
        assert "run_id" in call_kwargs
        run_id = call_kwargs["run_id"]
        assert run_id is not None
        assert len(run_id) > 0


@pytest.mark.asyncio
async def test_llm_node_basic_streaming():
    """Test LLMNode streams tokens and returns structured output."""
    # Create mock agent with structured output
    mock_agent = MagicMock(spec=Agent)

    # Mock the run_stream context manager
    mock_stream = AsyncMock()
    mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
    mock_stream.__aexit__ = AsyncMock(return_value=None)

    # Mock stream_text to yield token chunks
    async def mock_stream_text():
        yield "The"
        yield " answer"
        yield " is"
        yield " 42"

    mock_stream.stream_text = mock_stream_text

    # Mock get_output to return structured result
    mock_output = OutputModel(answer="The answer is 42", confidence=0.95)
    mock_stream.get_output = AsyncMock(return_value=mock_output)

    mock_agent.run_stream = MagicMock(return_value=mock_stream)

    # Create LLMNode
    node = LLMNode[InputModel, OutputModel](
        agent=mock_agent,
        prompt_template="Question: {query}",
        name="llm_node",
        run_id="llm-run-456",
    )

    # Execute
    input_data = InputModel(query="What is the meaning of life?")
    items = []
    async for item in node.astream(input_data):
        items.append(item)

    # Verify items
    assert len(items) == 6  # StreamStart + 4 tokens + StreamEnd
    assert isinstance(items[0], StreamStart)
    assert items[0].run_id == "llm-run-456"
    assert items[0].node_id == "llm_node"

    # Check token chunks
    tokens = [item for item in items if isinstance(item, TokenChunk)]
    assert len(tokens) == 4
    assert tokens[0].text == "The"
    assert tokens[1].text == " answer"
    assert tokens[0].token_index == 0
    assert tokens[1].token_index == 1

    # Check StreamEnd
    assert isinstance(items[-1], StreamEnd)
    expected_preview = {"answer": "The answer is 42", "confidence": 0.95}
    assert items[-1].result_preview == expected_preview


@pytest.mark.asyncio
async def test_llm_node_formats_prompt():
    """Test LLMNode formats prompt template with input fields."""
    mock_agent = MagicMock(spec=Agent)

    # Set up minimal mock
    mock_stream = AsyncMock()
    mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
    mock_stream.__aexit__ = AsyncMock(return_value=None)

    async def mock_stream_text():
        yield "response"

    mock_stream.stream_text = mock_stream_text
    output_model = OutputModel(answer="test", confidence=1.0)
    mock_stream.get_output = AsyncMock(return_value=output_model)
    mock_agent.run_stream = MagicMock(return_value=mock_stream)

    node = LLMNode[InputModel, OutputModel](
        agent=mock_agent,
        prompt_template="Query: {query}, Context: {context}",
    )

    input_data = InputModel(query="AI question", context="deep learning")
    items = []
    async for item in node.astream(input_data):
        items.append(item)

    # Verify agent was called with formatted prompt
    mock_agent.run_stream.assert_called_once_with(
        "Query: AI question, Context: deep learning"
    )


@pytest.mark.asyncio
async def test_llm_node_generates_run_id():
    """Test LLMNode generates run_id if not provided."""
    mock_agent = MagicMock(spec=Agent)

    mock_stream = AsyncMock()
    mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
    mock_stream.__aexit__ = AsyncMock(return_value=None)

    async def mock_stream_text():
        yield "test"

    mock_stream.stream_text = mock_stream_text
    output_model = OutputModel(answer="test", confidence=1.0)
    mock_stream.get_output = AsyncMock(return_value=output_model)
    mock_agent.run_stream = MagicMock(return_value=mock_stream)

    node = LLMNode[InputModel, OutputModel](
        agent=mock_agent,
        prompt_template="Query: {query}",
        # No run_id provided
    )

    input_data = InputModel(query="test")
    items = []
    async for item in node.astream(input_data):
        items.append(item)

    # Verify StreamStart has a generated run_id
    start_item = items[0]
    assert isinstance(start_item, StreamStart)
    assert start_item.run_id is not None
    assert len(start_item.run_id) > 0


@pytest.mark.asyncio
async def test_llm_node_handles_exception():
    """Test LLMNode emits NonFatalError and re-raises on exception."""
    mock_agent = MagicMock(spec=Agent)

    # Make run_stream raise an exception
    mock_stream = AsyncMock()
    mock_stream.__aenter__ = AsyncMock(side_effect=ValueError("LLM API error"))
    mock_agent.run_stream = MagicMock(return_value=mock_stream)

    node = LLMNode[InputModel, OutputModel](
        agent=mock_agent,
        prompt_template="Query: {query}",
        name="error_node",
        run_id="error-run",
    )

    input_data = InputModel(query="test")
    items = []

    # Should emit error and re-raise
    with pytest.raises(ValueError, match="LLM API error"):
        async for item in node.astream(input_data):
            items.append(item)

    # Should have emitted StreamStart and NonFatalError
    assert len(items) >= 2
    assert isinstance(items[0], StreamStart)
    assert isinstance(items[-1], NonFatalError)
    assert "LLM execution failed" in items[-1].message
    assert "LLM API error" in items[-1].message
    assert items[-1].recoverable is False


@pytest.mark.asyncio
async def test_llm_node_result_without_model_dump():
    """Test LLMNode handles results without model_dump method."""
    mock_agent = MagicMock(spec=Agent)

    mock_stream = AsyncMock()
    mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
    mock_stream.__aexit__ = AsyncMock(return_value=None)

    async def mock_stream_text():
        yield "plain"

    mock_stream.stream_text = mock_stream_text

    # Return a plain string instead of a model - wrap in a BaseModel
    class StringResult(BaseModel):
        value: str

    mock_stream.get_output = AsyncMock(return_value=StringResult(value="plain text"))
    mock_agent.run_stream = MagicMock(return_value=mock_stream)

    node = LLMNode[InputModel, StringResult](
        agent=mock_agent,
        prompt_template="Query: {query}",
    )

    input_data = InputModel(query="test")
    items = []
    async for item in node.astream(input_data):
        items.append(item)

    # Check StreamEnd has result_preview with model_dump
    stream_end = items[-1]
    assert isinstance(stream_end, StreamEnd)
    assert stream_end.result_preview == {"value": "plain text"}


@pytest.mark.asyncio
async def test_llm_node_empty_token_stream():
    """Test LLMNode handles empty token stream."""
    mock_agent = MagicMock(spec=Agent)

    mock_stream = AsyncMock()
    mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
    mock_stream.__aexit__ = AsyncMock(return_value=None)

    # Empty stream
    async def mock_stream_text():
        return
        yield  # pragma: no cover

    mock_stream.stream_text = mock_stream_text
    output_model = OutputModel(answer="", confidence=0.0)
    mock_stream.get_output = AsyncMock(return_value=output_model)
    mock_agent.run_stream = MagicMock(return_value=mock_stream)

    node = LLMNode[InputModel, OutputModel](
        agent=mock_agent,
        prompt_template="Query: {query}",
    )

    input_data = InputModel(query="test")
    items = []
    async for item in node.astream(input_data):
        items.append(item)

    # Should have StreamStart and StreamEnd, but no TokenChunks
    assert len(items) == 2
    assert isinstance(items[0], StreamStart)
    assert isinstance(items[1], StreamEnd)
    token_chunks = [item for item in items if isinstance(item, TokenChunk)]
    assert len(token_chunks) == 0


@pytest.mark.asyncio
async def test_agent_node_with_input_parameter():
    """Test AgentNode accepts input parameter in constructor."""
    mock_agent = MagicMock(spec=Agent)

    # Create a mock input source
    mock_input = MagicMock()

    node = AgentNode[InputModel, str](
        agent=mock_agent,
        prompt_template="Query: {query}",
        input=mock_input,  # Pass input parameter
        name="test_node",
    )

    # Verify input was set (from NodeWithInput base class)
    assert node.input == mock_input


@pytest.mark.asyncio
async def test_llm_node_with_input_parameter():
    """Test LLMNode accepts input parameter in constructor."""
    mock_agent = MagicMock(spec=Agent)
    mock_input = MagicMock()

    node = LLMNode[InputModel, OutputModel](
        agent=mock_agent,
        prompt_template="Query: {query}",
        input=mock_input,
        name="test_node",
    )

    assert node.input == mock_input


@pytest.mark.asyncio
async def test_agent_node_empty_prompt_template():
    """Test AgentNode with empty string prompt template."""
    mock_agent = MagicMock(spec=Agent)

    node = AgentNode[InputModel, str](
        agent=mock_agent,
        prompt_template="",  # Empty template
    )

    input_data = InputModel(query="test", context="context")
    prompt = node._format_prompt(input_data)

    # Should use model_dump_json since template is empty
    assert "test" in prompt
    assert "context" in prompt


@pytest.mark.asyncio
async def test_llm_node_stream_end_with_model_dump():
    """Test LLMNode StreamEnd includes result_preview from model_dump."""
    mock_agent = MagicMock(spec=Agent)

    mock_stream = AsyncMock()
    mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
    mock_stream.__aexit__ = AsyncMock(return_value=None)

    async def mock_stream_text():
        yield "token"

    mock_stream.stream_text = mock_stream_text

    # Return model with model_dump
    result = OutputModel(answer="detailed answer", confidence=0.88)
    mock_stream.get_output = AsyncMock(return_value=result)
    mock_agent.run_stream = MagicMock(return_value=mock_stream)

    node = LLMNode[InputModel, OutputModel](
        agent=mock_agent,
        prompt_template="Query: {query}",
    )

    input_data = InputModel(query="test")
    items = []
    async for item in node.astream(input_data):
        items.append(item)

    stream_end = items[-1]
    assert isinstance(stream_end, StreamEnd)
    assert stream_end.result_preview is not None
    assert stream_end.result_preview["answer"] == "detailed answer"
    assert stream_end.result_preview["confidence"] == 0.88


@pytest.mark.asyncio
async def test_llm_node_result_truly_no_model_dump():
    """Test LLMNode with result that has no model_dump attribute."""
    mock_agent = MagicMock(spec=Agent)

    mock_stream = AsyncMock()
    mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
    mock_stream.__aexit__ = AsyncMock(return_value=None)

    async def mock_stream_text():
        yield "text"

    mock_stream.stream_text = mock_stream_text

    # Return a plain dict (no model_dump attribute)
    mock_stream.get_output = AsyncMock(return_value={"key": "value"})
    mock_agent.run_stream = MagicMock(return_value=mock_stream)

    # Use Any for output type to allow dict
    node = LLMNode[InputModel, Any](
        agent=mock_agent,
        prompt_template="Query: {query}",
    )

    input_data = InputModel(query="test")
    items = []
    async for item in node.astream(input_data):
        items.append(item)

    # Should complete without error
    stream_end = items[-1]
    assert isinstance(stream_end, StreamEnd)
    # result_preview should be None since dict doesn't have model_dump
    assert stream_end.result_preview is None
