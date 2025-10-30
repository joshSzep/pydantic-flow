"""Tests for compression event emission in observers.

This module tests that observe_agent_stream properly captures and yields
compression events emitted during memory operations.
"""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from pydantic_flow.memory import _memory_event_emitter
from pydantic_flow.memory.compression import CompressionMetrics
from pydantic_flow.memory.events import MemoryCompressionComplete
from pydantic_flow.memory.events import MemoryCompressionPending
from pydantic_flow.streaming.base import ProgressType
from pydantic_flow.streaming.core_events import StreamEnd
from pydantic_flow.streaming.core_events import StreamStart
from pydantic_flow.streaming.observers import observe_agent_stream


async def async_gen(items):
    """Create an async generator from a list."""
    for item in items:
        yield item


@pytest.mark.asyncio
@patch("pydantic_flow.streaming.observers._active_flow_memory")
async def test_observer_context_setup_and_cleanup(mock_memory: MagicMock):
    """Test that observer properly sets up and cleans up emitter context."""
    # Setup mock agent
    mock_agent = MagicMock()
    mock_stream = MagicMock()

    # stream_text() returns an async generator directly (not awaited)
    mock_stream.stream_text = MagicMock(return_value=async_gen([]))
    mock_stream.get_output = AsyncMock(return_value="test")
    mock_stream.new_messages = MagicMock(return_value=[])
    mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
    mock_stream.__aexit__ = AsyncMock(return_value=None)
    mock_agent.run_stream = MagicMock(return_value=mock_stream)

    # Mock memory as not active
    mock_memory.get = MagicMock(return_value=None)

    # Verify emitter is None initially
    assert _memory_event_emitter.get(None) is None

    # Run observer
    async for _ in observe_agent_stream(mock_agent, "Test"):
        pass

    # Context should be reset after completion
    assert _memory_event_emitter.get(None) is None


@pytest.mark.asyncio
@patch("pydantic_flow.streaming.observers._active_flow_memory")
async def test_observer_context_cleanup_on_error(mock_memory: MagicMock):
    """Test that emitter context is reset even when an error occurs."""

    async def error_gen():
        if False:
            yield
        raise Exception("Stream error")

    mock_agent = MagicMock()
    mock_stream = MagicMock()
    mock_stream.stream_text = MagicMock(return_value=error_gen())
    mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
    mock_stream.__aexit__ = AsyncMock(return_value=None)
    mock_agent.run_stream = MagicMock(return_value=mock_stream)

    mock_memory.get = MagicMock(return_value=None)

    # Verify emitter is None initially
    assert _memory_event_emitter.get(None) is None

    try:
        async for _ in observe_agent_stream(mock_agent, "Test"):
            pass
    except Exception:
        pass  # Expected

    # Context should be reset even after error
    assert _memory_event_emitter.get(None) is None


@pytest.mark.asyncio
@patch("pydantic_flow.streaming.observers._active_flow_memory")
async def test_observer_captures_compression_pending_event(mock_memory: MagicMock):
    """Test that observer captures MemoryCompressionPending events."""
    mock_agent = MagicMock()
    mock_stream = MagicMock()
    mock_stream.stream_text = MagicMock(return_value=async_gen(["test"]))
    mock_stream.get_output = AsyncMock(return_value="test")
    mock_stream.new_messages = MagicMock(return_value=[{"role": "user"}])
    mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
    mock_stream.__aexit__ = AsyncMock(return_value=None)
    mock_agent.run_stream = MagicMock(return_value=mock_stream)

    # Setup mock memory that will trigger event emission
    mock_memory_inst = MagicMock()

    def mock_extend(messages):
        # Simulate emitting a compression event during extend()
        emitter = _memory_event_emitter.get()
        if emitter:
            event = MemoryCompressionPending(
                run_id="",
                node_id="",
                message_count=5,
                estimated_tokens=1000,
                compressor_name="test_compressor",
                compression_reason="Testing compression event emission",
            )
            emitter(event)

    mock_memory_inst.extend = mock_extend
    mock_memory.get = MagicMock(return_value=mock_memory_inst)

    # Run observer with message_history to trigger memory capture
    events = []
    async for event in observe_agent_stream(
        mock_agent, "Test", message_history=[], run_id="test-run", node_id="test-node"
    ):
        events.append(event)

    # Verify compression event was captured and yielded
    pending_events = [
        e for e in events if e.type == ProgressType.MEMORY_COMPRESSION_PENDING
    ]
    assert len(pending_events) == 1

    # Verify run_id and node_id were set correctly
    assert pending_events[0].run_id == "test-run"
    assert pending_events[0].node_id == "test-node"


@pytest.mark.asyncio
@patch("pydantic_flow.streaming.observers._active_flow_memory")
async def test_observer_captures_compression_complete_event(mock_memory: MagicMock):
    """Test that observer captures MemoryCompressionComplete events."""
    mock_agent = MagicMock()
    mock_stream = MagicMock()
    mock_stream.stream_text = MagicMock(return_value=async_gen(["test"]))
    mock_stream.get_output = AsyncMock(return_value="test")
    mock_stream.new_messages = MagicMock(return_value=[{"role": "user"}])
    mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
    mock_stream.__aexit__ = AsyncMock(return_value=None)
    mock_agent.run_stream = MagicMock(return_value=mock_stream)

    mock_memory_inst = MagicMock()

    def mock_extend(messages):
        emitter = _memory_event_emitter.get()
        if emitter:
            metrics = CompressionMetrics(
                messages_before=10,
                messages_after=5,
                estimated_tokens_before=1000,
                estimated_tokens_after=500,
                tokens_saved=500,
                compression_strategy="test",
                compression_ratio=0.5,
                compression_time_ms=1.0,
            )
            event = MemoryCompressionComplete(
                run_id="",
                node_id="",
                metrics=metrics,
            )
            emitter(event)

    mock_memory_inst.extend = mock_extend
    mock_memory.get = MagicMock(return_value=mock_memory_inst)

    events = []
    async for event in observe_agent_stream(
        mock_agent, "Test", message_history=[], run_id="test-run"
    ):
        events.append(event)

    complete_events = [
        e for e in events if e.type == ProgressType.MEMORY_COMPRESSION_COMPLETE
    ]
    assert len(complete_events) == 1
    assert complete_events[0].run_id == "test-run"


@pytest.mark.asyncio
@patch("pydantic_flow.streaming.observers._active_flow_memory")
async def test_observer_event_ordering(mock_memory: MagicMock):
    """Test that compression events are yielded in correct order."""
    mock_agent = MagicMock()
    mock_stream = MagicMock()
    mock_stream.stream_text = MagicMock(return_value=async_gen(["a", "b"]))
    mock_stream.get_output = AsyncMock(return_value="ab")
    mock_stream.new_messages = MagicMock(return_value=[{"role": "user"}])
    mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
    mock_stream.__aexit__ = AsyncMock(return_value=None)
    mock_agent.run_stream = MagicMock(return_value=mock_stream)

    mock_memory_inst = MagicMock()

    def mock_extend(messages):
        emitter = _memory_event_emitter.get()
        if emitter:
            emitter(
                MemoryCompressionPending(
                    run_id="",
                    node_id="",
                    message_count=5,
                    estimated_tokens=1000,
                    compressor_name="test",
                    compression_reason="Testing",
                )
            )

    mock_memory_inst.extend = mock_extend
    mock_memory.get = MagicMock(return_value=mock_memory_inst)

    events = []
    async for event in observe_agent_stream(mock_agent, "Test", message_history=[]):
        events.append(event)

    # Verify order: Start -> Tokens -> Compression -> End
    assert isinstance(events[0], StreamStart)
    assert any(e.type == ProgressType.TOKEN for e in events[1:-2])
    compression_idx = next(
        i
        for i, e in enumerate(events)
        if e.type == ProgressType.MEMORY_COMPRESSION_PENDING
    )
    end_idx = next(i for i, e in enumerate(events) if isinstance(e, StreamEnd))

    # Compression should be after tokens but before end
    assert compression_idx > 1
    assert compression_idx < end_idx
