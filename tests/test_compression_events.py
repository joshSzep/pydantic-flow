"""Tests for memory compression streaming events."""

import pytest

from pydantic_flow.hitl.decisions import InterruptDecision
from pydantic_flow.memory.compression import CompressionMetrics
from pydantic_flow.streaming.events import MemoryCompressionComplete
from pydantic_flow.streaming.events import MemoryCompressionPending
from pydantic_flow.streaming.events import ProgressItem
from pydantic_flow.streaming.events import ProgressType


def test_memory_compression_pending_creation():
    """Test basic MemoryCompressionPending event creation."""
    event = MemoryCompressionPending(
        estimated_tokens=5000,
        message_count=25,
        compressor_name="SlidingWindowCompressor",
        compression_reason="Token limit approaching",
        run_id="test-run",
        node_id="test-node",
    )

    assert event.type == ProgressType.MEMORY_COMPRESSION_PENDING
    assert event.estimated_tokens == 5000
    assert event.message_count == 25
    assert event.compressor_name == "SlidingWindowCompressor"
    assert event.compression_reason == "Token limit approaching"
    assert event.run_id == "test-run"
    assert event.node_id == "test-node"
    assert event.metadata == {}


def test_memory_compression_pending_with_metadata():
    """Test MemoryCompressionPending with additional metadata."""
    event = MemoryCompressionPending(
        estimated_tokens=3000,
        message_count=15,
        compressor_name="HybridCompressor",
        compression_reason="Threshold exceeded",
        metadata={
            "threshold": 4000,
            "current_tokens": 4500,
            "strategy": "sliding_window",
        },
    )

    assert event.metadata["threshold"] == 4000
    assert event.metadata["current_tokens"] == 4500
    assert event.metadata["strategy"] == "sliding_window"


def test_memory_compression_complete_creation():
    """Test basic MemoryCompressionComplete event creation."""
    metrics = CompressionMetrics(
        messages_before=25,
        messages_after=12,
        estimated_tokens_before=5000,
        estimated_tokens_after=2000,
        tokens_saved=3000,
        compression_ratio=0.4,
        compression_strategy="sliding_window",
        compression_time_ms=1.5,
    )

    preview = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well..."},
    ]

    event = MemoryCompressionComplete(
        metrics=metrics,
        compressed_messages_preview=preview,
        run_id="test-run",
        node_id="test-node",
    )

    assert event.type == ProgressType.MEMORY_COMPRESSION_COMPLETE
    assert event.metrics == metrics
    assert len(event.compressed_messages_preview) == 3
    assert event.compressed_messages_preview[0]["role"] == "system"


def test_memory_compression_complete_empty_preview():
    """Test MemoryCompressionComplete with empty preview."""
    metrics = CompressionMetrics(
        messages_before=10,
        messages_after=5,
        estimated_tokens_before=2000,
        estimated_tokens_after=1000,
        tokens_saved=1000,
        compression_ratio=0.5,
        compression_strategy="sliding_window",
        compression_time_ms=1.0,
    )

    event = MemoryCompressionComplete(metrics=metrics)

    assert event.compressed_messages_preview == []


@pytest.mark.asyncio
async def test_pending_event_no_interrupt_callback():
    """Test that pending event proceeds when no callback is set."""
    event = MemoryCompressionPending(
        estimated_tokens=3000,
        message_count=15,
        compressor_name="SlidingWindowCompressor",
        compression_reason="Test",
    )

    decision = await event.check_interrupt()
    assert decision.should_interrupt is False
    assert decision.reason is None


@pytest.mark.asyncio
async def test_pending_event_with_proceed_callback():
    """Test pending event with callback that allows compression."""

    async def approve_compression(item: ProgressItem) -> InterruptDecision:
        if isinstance(item, MemoryCompressionPending):
            if item.message_count >= 10:
                return InterruptDecision.proceed("Sufficient messages")
        return InterruptDecision.interrupt("Too few messages")

    event = MemoryCompressionPending(
        estimated_tokens=3000,
        message_count=15,
        compressor_name="SlidingWindowCompressor",
        compression_reason="Test",
    )
    event.set_interrupt_callback(approve_compression)

    decision = await event.check_interrupt()
    assert decision.should_interrupt is False
    assert decision.reason == "Sufficient messages"


@pytest.mark.asyncio
async def test_pending_event_with_reject_callback():
    """Test pending event with callback that blocks compression."""

    async def reject_compression(item: ProgressItem) -> InterruptDecision:
        if isinstance(item, MemoryCompressionPending):
            if item.message_count < 20:
                return InterruptDecision.interrupt(
                    "Too few messages",
                    metadata={"min_messages": 20},
                )
        return InterruptDecision.proceed()

    event = MemoryCompressionPending(
        estimated_tokens=3000,
        message_count=15,
        compressor_name="SlidingWindowCompressor",
        compression_reason="Test",
    )
    event.set_interrupt_callback(reject_compression)

    decision = await event.check_interrupt()
    assert decision.should_interrupt is True
    assert decision.reason == "Too few messages"
    assert decision.metadata["min_messages"] == 20


@pytest.mark.asyncio
async def test_pending_event_with_replacement_callback():
    """Test pending event with callback that provides replacement compressor."""

    async def replace_compressor(item: ProgressItem) -> InterruptDecision:
        if isinstance(item, MemoryCompressionPending):
            if item.compressor_name == "SlidingWindowCompressor":
                return InterruptDecision.interrupt(
                    "Using different strategy",
                    replacement_value="SummarizationCompressor",
                )
        return InterruptDecision.proceed()

    event = MemoryCompressionPending(
        estimated_tokens=3000,
        message_count=15,
        compressor_name="SlidingWindowCompressor",
        compression_reason="Test",
    )
    event.set_interrupt_callback(replace_compressor)

    decision = await event.check_interrupt()
    assert decision.should_interrupt is True
    assert decision.reason == "Using different strategy"
    assert decision.replacement_value == "SummarizationCompressor"


@pytest.mark.asyncio
async def test_complete_event_no_interrupt_callback():
    """Test that complete event proceeds when no callback is set."""
    metrics = CompressionMetrics(
        messages_before=25,
        messages_after=12,
        estimated_tokens_before=5000,
        estimated_tokens_after=2000,
        tokens_saved=3000,
        compression_ratio=0.4,
        compression_strategy="sliding_window",
        compression_time_ms=1.5,
    )

    event = MemoryCompressionComplete(metrics=metrics)

    decision = await event.check_interrupt()
    assert decision.should_interrupt is False


@pytest.mark.asyncio
async def test_complete_event_with_accept_callback():
    """Test complete event with callback that accepts compression."""

    async def accept_compression(item: ProgressItem) -> InterruptDecision:
        if isinstance(item, MemoryCompressionComplete):
            if item.metrics.compression_ratio < 0.5:
                return InterruptDecision.proceed("Good compression ratio")
        return InterruptDecision.interrupt("Poor compression")

    metrics = CompressionMetrics(
        messages_before=25,
        messages_after=12,
        estimated_tokens_before=5000,
        estimated_tokens_after=2000,
        tokens_saved=3000,
        compression_ratio=0.4,
        compression_strategy="sliding_window",
        compression_time_ms=1.5,
    )

    event = MemoryCompressionComplete(metrics=metrics)
    event.set_interrupt_callback(accept_compression)

    decision = await event.check_interrupt()
    assert decision.should_interrupt is False
    assert decision.reason == "Good compression ratio"


@pytest.mark.asyncio
async def test_complete_event_with_reject_callback():
    """Test complete event with callback that rejects compression."""

    async def reject_compression(item: ProgressItem) -> InterruptDecision:
        if isinstance(item, MemoryCompressionComplete):
            if item.metrics.compression_ratio >= 0.8:
                return InterruptDecision.interrupt(
                    "Insufficient compression",
                    metadata={"required_ratio": 0.5},
                )
        return InterruptDecision.proceed()

    metrics = CompressionMetrics(
        messages_before=25,
        messages_after=22,
        estimated_tokens_before=5000,
        estimated_tokens_after=4500,
        tokens_saved=500,
        compression_ratio=0.9,
        compression_strategy="sliding_window",
        compression_time_ms=1.2,
    )

    event = MemoryCompressionComplete(metrics=metrics)
    event.set_interrupt_callback(reject_compression)

    decision = await event.check_interrupt()
    assert decision.should_interrupt is True
    assert decision.reason == "Insufficient compression"
    assert decision.metadata["required_ratio"] == 0.5


@pytest.mark.asyncio
async def test_complete_event_with_replacement_callback():
    """Test complete event with callback that provides replacement messages."""

    async def replace_messages(item: ProgressItem) -> InterruptDecision:
        if isinstance(item, MemoryCompressionComplete):
            replacement = [
                {"role": "system", "content": "Modified system message"},
                {"role": "user", "content": "Custom compression result"},
            ]
            return InterruptDecision.interrupt(
                "Using custom compression",
                replacement_value=replacement,
            )
        return InterruptDecision.proceed()

    metrics = CompressionMetrics(
        messages_before=25,
        messages_after=12,
        estimated_tokens_before=5000,
        estimated_tokens_after=2000,
        tokens_saved=3000,
        compression_ratio=0.4,
        compression_strategy="sliding_window",
        compression_time_ms=1.5,
    )

    event = MemoryCompressionComplete(metrics=metrics)
    event.set_interrupt_callback(replace_messages)

    decision = await event.check_interrupt()
    assert decision.should_interrupt is True
    assert decision.reason == "Using custom compression"
    assert len(decision.replacement_value) == 2
    assert decision.replacement_value[0]["role"] == "system"


def test_pending_event_serialization():
    """Test that pending event can be serialized to dict."""
    event = MemoryCompressionPending(
        estimated_tokens=3000,
        message_count=15,
        compressor_name="SlidingWindowCompressor",
        compression_reason="Test",
        metadata={"key": "value"},
    )

    data = event.model_dump()

    assert data["type"] == "memory_compression_pending"
    assert data["estimated_tokens"] == 3000
    assert data["message_count"] == 15
    assert data["compressor_name"] == "SlidingWindowCompressor"
    assert data["compression_reason"] == "Test"
    assert data["metadata"]["key"] == "value"


def test_complete_event_serialization():
    """Test that complete event can be serialized to dict."""
    metrics = CompressionMetrics(
        messages_before=25,
        messages_after=12,
        estimated_tokens_before=5000,
        estimated_tokens_after=2000,
        tokens_saved=3000,
        compression_ratio=0.4,
        compression_strategy="sliding_window",
        compression_time_ms=1.5,
    )

    preview = [{"role": "system", "content": "Test"}]

    event = MemoryCompressionComplete(
        metrics=metrics,
        compressed_messages_preview=preview,
    )

    data = event.model_dump()

    assert data["type"] == "memory_compression_complete"
    assert data["metrics"]["messages_before"] == 25
    assert data["metrics"]["messages_after"] == 12
    assert len(data["compressed_messages_preview"]) == 1


def test_pending_event_type_discriminator():
    """Test that pending event has correct type discriminator."""
    event = MemoryCompressionPending(
        estimated_tokens=3000,
        message_count=15,
        compressor_name="Test",
        compression_reason="Test",
    )

    assert isinstance(event, ProgressItem)
    assert event.type == ProgressType.MEMORY_COMPRESSION_PENDING
    assert event.type.value == "memory_compression_pending"


def test_complete_event_type_discriminator():
    """Test that complete event has correct type discriminator."""
    metrics = CompressionMetrics(
        messages_before=10,
        messages_after=5,
        estimated_tokens_before=2000,
        estimated_tokens_after=1000,
        tokens_saved=1000,
        compression_ratio=0.5,
        compression_strategy="sliding_window",
        compression_time_ms=1.0,
    )

    event = MemoryCompressionComplete(metrics=metrics)

    assert isinstance(event, ProgressItem)
    assert event.type == ProgressType.MEMORY_COMPRESSION_COMPLETE
    assert event.type.value == "memory_compression_complete"


def test_pending_event_timestamp_auto_generated():
    """Test that pending event gets automatic timestamp."""
    event = MemoryCompressionPending(
        estimated_tokens=3000,
        message_count=15,
        compressor_name="Test",
        compression_reason="Test",
    )

    assert event.timestamp is not None


def test_complete_event_timestamp_auto_generated():
    """Test that complete event gets automatic timestamp."""
    metrics = CompressionMetrics(
        messages_before=10,
        messages_after=5,
        estimated_tokens_before=2000,
        estimated_tokens_after=1000,
        tokens_saved=1000,
        compression_ratio=0.5,
        compression_strategy="sliding_window",
        compression_time_ms=1.0,
    )

    event = MemoryCompressionComplete(metrics=metrics)

    assert event.timestamp is not None


@pytest.mark.asyncio
async def test_interrupt_callback_receives_correct_event_type():
    """Test that interrupt callback receives the correct event instance."""
    received_events = []

    async def capture_event(item: ProgressItem) -> InterruptDecision:
        received_events.append(item)
        return InterruptDecision.proceed()

    pending_event = MemoryCompressionPending(
        estimated_tokens=3000,
        message_count=15,
        compressor_name="Test",
        compression_reason="Test",
    )
    pending_event.set_interrupt_callback(capture_event)
    await pending_event.check_interrupt()

    assert len(received_events) == 1
    assert isinstance(received_events[0], MemoryCompressionPending)
    assert received_events[0].estimated_tokens == 3000

    received_events.clear()

    metrics = CompressionMetrics(
        messages_before=10,
        messages_after=5,
        estimated_tokens_before=2000,
        estimated_tokens_after=1000,
        tokens_saved=1000,
        compression_ratio=0.5,
        compression_strategy="sliding_window",
        compression_time_ms=1.0,
    )

    complete_event = MemoryCompressionComplete(metrics=metrics)
    complete_event.set_interrupt_callback(capture_event)
    await complete_event.check_interrupt()

    assert len(received_events) == 1
    assert isinstance(received_events[0], MemoryCompressionComplete)
    assert received_events[0].metrics.compression_ratio == 0.5
