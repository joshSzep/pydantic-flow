"""Tests for built-in memory compression strategies."""

from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from pydantic_flow.memory.compression import CompressionMetrics
from pydantic_flow.memory.compression import MemoryCompressor
from pydantic_flow.memory.compressors import HybridCompressor
from pydantic_flow.memory.compressors import SlidingWindowCompressor
from pydantic_flow.memory.compressors import SummarizationCompressor


# Mock message helpers
def mock_message(role: str, content: str) -> dict[str, Any]:
    """Create a mock message dictionary."""
    return {"role": role, "content": content, "kind": "request"}


def mock_messages(count: int, prefix: str = "msg") -> list[dict[str, Any]]:
    """Create a list of mock messages."""
    return [mock_message("user", f"{prefix}_{i}") for i in range(count)]


# SlidingWindowCompressor Tests


def test_sliding_window_init_defaults():
    """Test SlidingWindowCompressor initialization with defaults."""
    compressor = SlidingWindowCompressor()

    assert compressor.window_size == 10
    assert compressor.max_tokens == 4000
    assert compressor.preserve_system_messages is True
    assert compressor.preserve_recent_messages == 5


def test_sliding_window_init_custom():
    """Test SlidingWindowCompressor initialization with custom values."""
    compressor = SlidingWindowCompressor(
        window_size=20,
        max_tokens=8000,
        preserve_system_messages=False,
        preserve_recent_messages=10,
    )

    assert compressor.window_size == 20
    assert compressor.max_tokens == 8000
    assert compressor.preserve_system_messages is False
    assert compressor.preserve_recent_messages == 10


def test_sliding_window_name():
    """Test SlidingWindowCompressor name property."""
    compressor = SlidingWindowCompressor(window_size=15)
    assert compressor.name == "sliding_window_15"


@pytest.mark.asyncio
async def test_sliding_window_compress_basic():
    """Test basic sliding window compression."""
    compressor = SlidingWindowCompressor(window_size=5)
    messages = mock_messages(20)

    compressed, metrics = await compressor.compress(messages)

    assert len(compressed) == 5
    assert metrics.messages_before == 20
    assert metrics.messages_after == 5
    assert metrics.compression_strategy == "sliding_window_5"
    assert metrics.tokens_saved >= 0  # May be 0 with mock messages
    assert metrics.compression_ratio == 0.25


@pytest.mark.asyncio
async def test_sliding_window_with_system_messages():
    """Test sliding window preserves system messages."""
    compressor = SlidingWindowCompressor(window_size=5, preserve_system_messages=True)
    messages = [
        mock_message("system", "You are helpful"),
        *mock_messages(20, "user"),
    ]

    compressed, metrics = await compressor.compress(messages)

    # Should have 1 system + 5 recent
    assert len(compressed) == 6
    assert compressed[0]["role"] == "system"
    assert metrics.metadata["system_messages_preserved"] == 1


@pytest.mark.asyncio
async def test_sliding_window_fewer_than_window():
    """Test sliding window with fewer messages than window size."""
    compressor = SlidingWindowCompressor(window_size=10)
    messages = mock_messages(5)

    compressed, metrics = await compressor.compress(messages)

    # All messages should be kept
    assert len(compressed) == 5
    assert metrics.compression_ratio == 1.0


# SummarizationCompressor Tests


def test_summarization_init():
    """Test SummarizationCompressor initialization."""
    agent = MagicMock()
    compressor = SummarizationCompressor(agent=agent, max_tokens=6000)

    assert compressor.agent == agent
    assert compressor.max_tokens == 6000
    assert compressor.name == "summarization"


def test_summarization_format_messages():
    """Test message formatting for summarization."""
    agent = MagicMock()
    compressor = SummarizationCompressor(agent=agent)
    messages = [
        mock_message("user", "Hello"),
        mock_message("assistant", "Hi there"),
        mock_message("user", "How are you?"),
    ]

    formatted = compressor._format_messages_for_summary(messages)

    assert "user: Hello" in formatted
    assert "assistant: Hi there" in formatted
    assert "user: How are you?" in formatted


def test_summarization_create_summary_message():
    """Test summary message creation."""
    agent = MagicMock()
    compressor = SummarizationCompressor(agent=agent)
    summary = "User greeted and asked about wellbeing"

    result = compressor._create_summary_message(summary)

    assert result["role"] == "system"
    assert summary in result["content"]
    assert "[Conversation Summary]" in result["content"]


@pytest.mark.asyncio
async def test_summarization_compress_basic():
    """Test basic summarization compression."""
    agent = AsyncMock()
    agent.run = AsyncMock(return_value=MagicMock(output="Summary of conversation"))

    compressor = SummarizationCompressor(agent=agent, preserve_recent_messages=3)
    messages = mock_messages(15)

    compressed, metrics = await compressor.compress(messages)

    # Should have summary + 3 recent
    assert len(compressed) <= len(messages)
    assert metrics.compression_strategy == "summarization"
    assert metrics.tokens_saved >= 0
    agent.run.assert_called_once()


@pytest.mark.asyncio
async def test_summarization_no_compressible():
    """Test summarization with no compressible messages."""
    agent = AsyncMock()
    compressor = SummarizationCompressor(agent=agent, preserve_recent_messages=10)
    messages = mock_messages(5)  # Fewer than preserve_recent

    compressed, metrics = await compressor.compress(messages)

    # No compression needed
    assert len(compressed) == len(messages)
    assert metrics.compression_ratio == 1.0
    assert metrics.metadata["reason"] == "no_compressible_messages"
    agent.run.assert_not_called()


@pytest.mark.asyncio
async def test_summarization_agent_failure():
    """Test summarization handles agent failures."""
    agent = AsyncMock()
    agent.run = AsyncMock(side_effect=RuntimeError("API error"))

    compressor = SummarizationCompressor(agent=agent)
    messages = mock_messages(15)

    with pytest.raises(RuntimeError, match="Summarization agent failed"):
        await compressor.compress(messages)


# HybridCompressor Tests


def test_hybrid_init_defaults():
    """Test HybridCompressor initialization with defaults."""
    compressor = HybridCompressor()

    assert compressor.summarizer_agent is None
    assert compressor.summarization_threshold == 15
    assert compressor.window_size == 10
    assert compressor.name == "hybrid"


def test_hybrid_init_with_agent():
    """Test HybridCompressor initialization with agent."""
    agent = MagicMock()
    compressor = HybridCompressor(
        summarizer_agent=agent,
        summarization_threshold=20,
        window_size=15,
    )

    assert compressor.summarizer_agent == agent
    assert compressor.summarization_threshold == 20
    assert compressor.window_size == 15


def test_hybrid_select_strategy_no_agent():
    """Test hybrid strategy selection without agent."""
    compressor = HybridCompressor(summarizer_agent=None)
    messages = mock_messages(50)

    strategy = compressor._select_strategy(messages)

    assert strategy == "sliding_window"


def test_hybrid_select_strategy_few_messages():
    """Test hybrid strategy selection with few messages."""
    agent = MagicMock()
    compressor = HybridCompressor(summarizer_agent=agent, summarization_threshold=20)
    messages = mock_messages(10)  # Less than threshold

    strategy = compressor._select_strategy(messages)

    assert strategy == "sliding_window"


def test_hybrid_select_strategy_many_messages():
    """Test hybrid strategy selection with many messages."""
    agent = MagicMock()
    compressor = HybridCompressor(
        summarizer_agent=agent,
        summarization_threshold=15,
        preserve_recent_messages=5,
    )
    messages = mock_messages(30)  # More than threshold

    strategy = compressor._select_strategy(messages)

    assert strategy == "summarization"


@pytest.mark.asyncio
async def test_hybrid_compress_sliding_window():
    """Test hybrid compression using sliding window strategy."""
    compressor = HybridCompressor(
        summarizer_agent=None,  # Forces sliding window
        window_size=10,
    )
    messages = mock_messages(25)

    _compressed, metrics = await compressor.compress(messages)

    assert len(_compressed) == 10
    assert metrics.compression_strategy == "hybrid"
    assert metrics.metadata["selected_strategy"] == "sliding_window"


@pytest.mark.asyncio
async def test_hybrid_compress_summarization():
    """Test hybrid compression using summarization strategy."""
    agent = AsyncMock()
    agent.run = AsyncMock(return_value=MagicMock(output="Summary of long conversation"))

    compressor = HybridCompressor(
        summarizer_agent=agent,
        summarization_threshold=10,
        preserve_recent_messages=3,
    )
    messages = mock_messages(25)

    _compressed, metrics = await compressor.compress(messages)

    assert metrics.compression_strategy == "hybrid"
    assert metrics.metadata["selected_strategy"] == "summarization"
    agent.run.assert_called_once()


@pytest.mark.asyncio
async def test_hybrid_threshold_boundary():
    """Test hybrid compression at threshold boundary."""
    agent = AsyncMock()
    agent.run = AsyncMock(return_value=MagicMock(output="Summary"))

    compressor = HybridCompressor(
        summarizer_agent=agent,
        summarization_threshold=15,
        preserve_recent_messages=5,
    )

    # Exactly at threshold (15 compressible messages)
    messages = [*mock_messages(5), *mock_messages(15, "middle")]
    # preserve_recent=5 means last 5 are recent, so 15 are compressible

    _compressed, metrics = await compressor.compress(messages)

    # Should use summarization when >= threshold
    assert metrics.metadata["selected_strategy"] == "summarization"


# Integration Tests


@pytest.mark.asyncio
async def test_all_compressors_protocol_conformance():
    """Test that all compressors conform to MemoryCompressor protocol."""
    agent = AsyncMock()
    agent.run = AsyncMock(return_value=MagicMock(output="Summary"))

    compressors = [
        SlidingWindowCompressor(),
        SummarizationCompressor(agent=agent),
        HybridCompressor(summarizer_agent=agent),
    ]

    for compressor in compressors:
        assert isinstance(compressor, MemoryCompressor)
        assert hasattr(compressor, "should_compress")
        assert hasattr(compressor, "compress")
        assert hasattr(compressor, "name")


@pytest.mark.asyncio
async def test_compression_preserves_system_messages():
    """Test that all compressors preserve system messages when configured."""
    agent = AsyncMock()
    agent.run = AsyncMock(return_value=MagicMock(output="Summary"))

    messages = [
        mock_message("system", "You are helpful"),
        *mock_messages(20, "user"),
    ]

    compressors = [
        SlidingWindowCompressor(window_size=5, preserve_system_messages=True),
        SummarizationCompressor(
            agent=agent,
            preserve_system_messages=True,
            preserve_recent_messages=5,
        ),
        HybridCompressor(
            summarizer_agent=None,
            window_size=5,
            preserve_system_messages=True,
        ),
    ]

    for compressor in compressors:
        compressed, _metrics = await compressor.compress(messages)
        # All should preserve the system message
        assert any(msg["role"] == "system" for msg in compressed)


@pytest.mark.asyncio
async def test_compression_metrics_consistency():
    """Test that all compressors return valid compression metrics."""
    agent = AsyncMock()
    agent.run = AsyncMock(return_value=MagicMock(output="Summary"))

    messages = mock_messages(20)

    compressors = [
        SlidingWindowCompressor(window_size=5),
        SummarizationCompressor(agent=agent, preserve_recent_messages=5),
        HybridCompressor(summarizer_agent=None, window_size=5),
    ]

    for compressor in compressors:
        compressed, metrics = await compressor.compress(messages)

        # Validate metric structure
        assert isinstance(metrics, CompressionMetrics)
        assert metrics.messages_before == len(messages)
        assert metrics.messages_after == len(compressed)
        assert 0.0 <= metrics.compression_ratio <= 1.0
        assert metrics.compression_time_ms >= 0
        assert metrics.estimated_tokens_before >= metrics.estimated_tokens_after
