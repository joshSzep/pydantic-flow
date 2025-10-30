"""Integration tests for ConversationMemory.maybe_compress() method.

This module tests the full compression pipeline including:
- Threshold checking
- Actual compression execution
- Event emission
- HITL interrupts (pending and complete)
- Rollback scenarios
- Compressor replacement
"""

from pydantic_ai.messages import ModelMessage
from pydantic_ai.messages import ModelRequest
from pydantic_ai.messages import ModelResponse
from pydantic_ai.messages import SystemPromptPart
from pydantic_ai.messages import TextPart
from pydantic_ai.messages import UserPromptPart
import pytest

from pydantic_flow.hitl.decisions import InterruptDecision
from pydantic_flow.hitl.interrupts import InterruptionRequested
from pydantic_flow.memory.compression import CompressionMetrics
from pydantic_flow.memory.compressors import SlidingWindowCompressor
from pydantic_flow.memory.events import MemoryCompressionComplete
from pydantic_flow.memory.events import MemoryCompressionPending
from pydantic_flow.memory.memory import ConversationMemory
from pydantic_flow.memory.memory import _memory_event_emitter
from pydantic_flow.streaming.base import ProgressItem


@pytest.fixture
def sample_messages() -> list[ModelMessage]:
    """Create sample messages for testing."""
    return [
        ModelRequest(parts=[SystemPromptPart(content="You are a helpful assistant")]),
        ModelRequest(parts=[UserPromptPart(content="Hello")]),
        ModelResponse(parts=[TextPart(content="Hi there!")]),
        ModelRequest(parts=[UserPromptPart(content="How are you?")]),
        ModelResponse(parts=[TextPart(content="I'm doing well!")]),
        ModelRequest(parts=[UserPromptPart(content="That's great!")]),
        ModelResponse(parts=[TextPart(content="Thank you!")]),
        ModelRequest(parts=[UserPromptPart(content="Goodbye")]),
        ModelResponse(parts=[TextPart(content="Bye!")]),
    ]


@pytest.mark.asyncio
async def test_maybe_compress_no_compressor(sample_messages: list[ModelMessage]):
    """Test that maybe_compress returns None when no compressor is configured."""
    memory = ConversationMemory()
    memory._messages = sample_messages.copy()

    result = await memory.maybe_compress()

    assert result is None
    assert len(memory._messages) == len(sample_messages)


@pytest.mark.asyncio
async def test_maybe_compress_below_threshold(sample_messages: list[ModelMessage]):
    """Test that maybe_compress returns None when below compression threshold."""
    compressor = SlidingWindowCompressor(window_size=100, max_tokens=10000)
    memory = ConversationMemory(compressor=compressor)
    memory._messages = sample_messages[:3]  # Only 3 messages

    result = await memory.maybe_compress()

    assert result is None
    assert len(memory._messages) == 3


@pytest.mark.asyncio
async def test_maybe_compress_success_no_events(sample_messages: list[ModelMessage]):
    """Test successful compression without event emitter context."""
    compressor = SlidingWindowCompressor(window_size=5, max_tokens=5)
    memory = ConversationMemory(compressor=compressor)
    memory._messages = sample_messages.copy()

    result = await memory.maybe_compress()

    assert result is not None
    assert isinstance(result, CompressionMetrics)
    assert result.messages_before == len(sample_messages)
    assert len(memory._messages) < len(sample_messages)
    assert result.messages_after == len(memory._messages)
    assert memory._last_compression_rejected is False
    assert len(memory._compression_history) == 1


@pytest.mark.asyncio
async def test_maybe_compress_with_event_emission(sample_messages: list[ModelMessage]):
    """Test compression with event emission."""
    compressor = SlidingWindowCompressor(window_size=5, max_tokens=5)
    memory = ConversationMemory(compressor=compressor)
    memory._messages = sample_messages.copy()

    emitted_events = []

    def event_emitter(event):
        emitted_events.append(event)

    token = _memory_event_emitter.set(event_emitter)
    try:
        result = await memory.maybe_compress()
    finally:
        _memory_event_emitter.reset(token)

    assert result is not None
    assert len(emitted_events) == 2
    assert isinstance(emitted_events[0], MemoryCompressionPending)
    assert isinstance(emitted_events[1], MemoryCompressionComplete)

    # Verify pending event
    pending = emitted_events[0]
    assert pending.message_count == len(sample_messages)
    assert pending.compressor_name == compressor.name

    # Verify complete event
    complete = emitted_events[1]
    assert complete.metrics.messages_before == len(sample_messages)
    assert len(complete.compressed_messages_preview) <= 5


@pytest.mark.asyncio
async def test_maybe_compress_pending_interrupt_reject(
    sample_messages: list[ModelMessage],
):
    """Test rejection at pending stage."""
    compressor = SlidingWindowCompressor(window_size=5, max_tokens=5)
    memory = ConversationMemory(compressor=compressor)
    memory._messages = sample_messages.copy()
    original_count = len(memory._messages)

    async def interrupt_callback(item: ProgressItem) -> InterruptDecision:
        return InterruptDecision(should_interrupt=True)

    def event_emitter(event):
        if isinstance(event, MemoryCompressionPending):
            event.interrupt_callback = interrupt_callback

    token = _memory_event_emitter.set(event_emitter)
    try:
        with pytest.raises(InterruptionRequested) as exc_info:
            await memory.maybe_compress()

        # Verify state
        assert memory._last_compression_rejected is True
        assert len(memory._messages) == original_count
        assert len(memory._compression_history) == 0

        # Verify exception
        exc = exc_info.value
        assert isinstance(exc, InterruptionRequested)
        assert exc.checkpoint.interrupted_node_id == "memory_compression"
    finally:
        _memory_event_emitter.reset(token)


@pytest.mark.asyncio
async def test_maybe_compress_pending_interrupt_with_replacement(
    sample_messages: list[ModelMessage],
):
    """Test compressor replacement at pending stage."""
    original_compressor = SlidingWindowCompressor(window_size=5, max_tokens=5)
    replacement_compressor = SlidingWindowCompressor(window_size=3, max_tokens=5)
    memory = ConversationMemory(compressor=original_compressor)
    memory._messages = sample_messages.copy()

    async def interrupt_callback(item: ProgressItem) -> InterruptDecision:
        return InterruptDecision(
            should_interrupt=True,
            replacement_value=replacement_compressor,
        )

    emitted_events = []

    def event_emitter(event):
        emitted_events.append(event)
        if isinstance(event, MemoryCompressionPending) and len(emitted_events) == 1:
            event.interrupt_callback = interrupt_callback

    token = _memory_event_emitter.set(event_emitter)
    try:
        result = await memory.maybe_compress()
    finally:
        _memory_event_emitter.reset(token)

    # Should have succeeded with replacement compressor
    assert result is not None
    assert memory._compressor is replacement_compressor
    assert memory._last_compression_rejected is False
    # Should have 3 events: pending1, pending2, complete
    assert len(emitted_events) == 3


@pytest.mark.asyncio
async def test_maybe_compress_complete_interrupt_reject(
    sample_messages: list[ModelMessage],
):
    """Test rejection at complete stage with rollback."""
    compressor = SlidingWindowCompressor(window_size=5, max_tokens=5)
    memory = ConversationMemory(compressor=compressor)
    memory._messages = sample_messages.copy()
    original_count = len(memory._messages)

    async def interrupt_callback(item: ProgressItem) -> InterruptDecision:
        return InterruptDecision(should_interrupt=True)

    def event_emitter(event):
        if isinstance(event, MemoryCompressionComplete):
            event.interrupt_callback = interrupt_callback

    token = _memory_event_emitter.set(event_emitter)
    try:
        with pytest.raises(InterruptionRequested) as exc_info:
            await memory.maybe_compress()

        # Verify rollback occurred
        assert memory._last_compression_rejected is True
        assert len(memory._messages) == original_count
        assert len(memory._compression_history) == 0

        # Verify exception
        exc = exc_info.value
        assert isinstance(exc, InterruptionRequested)
        assert exc.checkpoint.interrupted_node_id == "memory_compression"
        assert len(exc.checkpoint.conversation_memory) == original_count
    finally:
        _memory_event_emitter.reset(token)


@pytest.mark.asyncio
async def test_maybe_compress_complete_interrupt_with_replacement(
    sample_messages: list[ModelMessage],
):
    """Test custom message replacement at complete stage."""
    compressor = SlidingWindowCompressor(window_size=5, max_tokens=5)
    memory = ConversationMemory(compressor=compressor)
    memory._messages = sample_messages.copy()

    custom_messages = sample_messages[:2]  # Keep only first 2

    async def interrupt_callback(item: ProgressItem) -> InterruptDecision:
        return InterruptDecision(
            should_interrupt=True,
            replacement_value=custom_messages,
        )

    def event_emitter(event):
        if isinstance(event, MemoryCompressionComplete):
            event.interrupt_callback = interrupt_callback

    token = _memory_event_emitter.set(event_emitter)
    try:
        result = await memory.maybe_compress()
    finally:
        _memory_event_emitter.reset(token)

    # Should have succeeded with custom messages
    assert result is not None
    assert len(memory._messages) == 2
    assert memory._messages == custom_messages
    assert len(memory._compression_history) == 1


@pytest.mark.asyncio
async def test_maybe_compress_multiple_compressions(
    sample_messages: list[ModelMessage],
):
    """Test multiple compressions build up history."""
    compressor = SlidingWindowCompressor(window_size=5, max_tokens=5)
    memory = ConversationMemory(compressor=compressor)
    memory._messages = sample_messages.copy()

    result1 = await memory.maybe_compress()
    assert result1 is not None
    assert len(memory._compression_history) == 1

    # Add more messages
    memory._messages.extend(sample_messages[3:])

    result2 = await memory.maybe_compress()
    assert result2 is not None
    assert len(memory._compression_history) == 2


@pytest.mark.asyncio
async def test_maybe_compress_preview_generation(sample_messages: list[ModelMessage]):
    """Test that compression complete event includes message preview."""
    compressor = SlidingWindowCompressor(window_size=5, max_tokens=5)
    memory = ConversationMemory(compressor=compressor)
    memory._messages = sample_messages.copy()

    emitted_events = []

    def event_emitter(event):
        emitted_events.append(event)

    token = _memory_event_emitter.set(event_emitter)
    try:
        await memory.maybe_compress()
    finally:
        _memory_event_emitter.reset(token)

    complete_event = emitted_events[1]
    assert isinstance(complete_event, MemoryCompressionComplete)
    assert len(complete_event.compressed_messages_preview) > 0

    # Check preview structure
    for preview_msg in complete_event.compressed_messages_preview:
        assert "role" in preview_msg
        assert "content" in preview_msg
        assert len(preview_msg["content"]) <= 103  # 100 + "..."


@pytest.mark.asyncio
async def test_maybe_compress_preserves_system_messages(
    sample_messages: list[ModelMessage],
):
    """Test that compression preserves system messages."""
    compressor = SlidingWindowCompressor(window_size=3, max_tokens=5)
    memory = ConversationMemory(compressor=compressor)
    memory._messages = sample_messages.copy()

    # Verify first message is system
    assert isinstance(memory._messages[0].parts[0], SystemPromptPart)

    result = await memory.maybe_compress()

    assert result is not None
    # System message should still be present (may not be at index 0 due to window)
    system_messages = [
        msg
        for msg in memory._messages
        if any(isinstance(part, SystemPromptPart) for part in msg.parts)
    ]
    assert len(system_messages) >= 1


@pytest.mark.asyncio
async def test_maybe_compress_metrics_accuracy(sample_messages: list[ModelMessage]):
    """Test that compression metrics are accurate."""
    compressor = SlidingWindowCompressor(window_size=5, max_tokens=5)
    memory = ConversationMemory(compressor=compressor)
    memory._messages = sample_messages.copy()
    original_count = len(sample_messages)

    result = await memory.maybe_compress()

    assert result is not None
    assert result.messages_before == original_count
    assert result.messages_after == len(memory._messages)
    assert result.estimated_tokens_before > 0
    assert result.estimated_tokens_after > 0
    assert result.tokens_saved == (
        result.estimated_tokens_before - result.estimated_tokens_after
    )
    assert 0 <= result.compression_ratio <= 1
