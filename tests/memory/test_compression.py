"""Tests for memory compression protocol and base classes."""

from collections.abc import Sequence
from typing import Any
from unittest.mock import MagicMock

from pydantic import ValidationError
import pytest

from pydantic_flow.memory.compression import BaseMemoryCompressor
from pydantic_flow.memory.compression import CompressionMetrics
from pydantic_flow.memory.compression import MemoryCompressor

# Test CompressionMetrics Model


def test_compression_metrics_basic():
    """Test basic CompressionMetrics creation."""
    metrics = CompressionMetrics(
        messages_before=50,
        messages_after=10,
        estimated_tokens_before=5000,
        estimated_tokens_after=1000,
        tokens_saved=4000,
        compression_ratio=0.2,
        compression_strategy="test_strategy",
        compression_time_ms=1.5,
    )

    assert metrics.messages_before == 50
    assert metrics.messages_after == 10
    assert metrics.estimated_tokens_before == 5000
    assert metrics.estimated_tokens_after == 1000
    assert metrics.tokens_saved == 4000
    assert metrics.compression_ratio == 0.2
    assert metrics.compression_strategy == "test_strategy"
    assert metrics.compression_time_ms == 1.5
    assert metrics.metadata == {}


def test_compression_metrics_with_metadata():
    """Test CompressionMetrics with custom metadata."""
    metrics = CompressionMetrics(
        messages_before=20,
        messages_after=5,
        estimated_tokens_before=2000,
        estimated_tokens_after=500,
        tokens_saved=1500,
        compression_ratio=0.25,
        compression_strategy="custom",
        compression_time_ms=2.0,
        metadata={"custom_key": "custom_value", "count": 42},
    )

    assert metrics.metadata["custom_key"] == "custom_value"
    assert metrics.metadata["count"] == 42


def test_compression_metrics_percentage_reduction():
    """Test percentage_reduction computed property."""
    metrics = CompressionMetrics(
        messages_before=100,
        messages_after=25,
        estimated_tokens_before=10000,
        estimated_tokens_after=2500,
        tokens_saved=7500,
        compression_ratio=0.25,
        compression_strategy="test",
        compression_time_ms=1.0,
    )

    assert metrics.percentage_reduction == 75.0


def test_compression_metrics_percentage_reduction_zero_tokens():
    """Test percentage_reduction when tokens_before is zero."""
    metrics = CompressionMetrics(
        messages_before=0,
        messages_after=0,
        estimated_tokens_before=0,
        estimated_tokens_after=0,
        tokens_saved=0,
        compression_ratio=0.0,
        compression_strategy="test",
        compression_time_ms=0.0,
    )

    assert metrics.percentage_reduction == 0.0


def test_compression_metrics_messages_removed():
    """Test messages_removed computed property."""
    metrics = CompressionMetrics(
        messages_before=50,
        messages_after=10,
        estimated_tokens_before=5000,
        estimated_tokens_after=1000,
        tokens_saved=4000,
        compression_ratio=0.2,
        compression_strategy="test",
        compression_time_ms=1.0,
    )

    assert metrics.messages_removed == 40


def test_compression_metrics_messages_removed_edge_case():
    """Test messages_removed when after > before (shouldn't happen, but handle)."""
    metrics = CompressionMetrics(
        messages_before=10,
        messages_after=10,
        estimated_tokens_before=1000,
        estimated_tokens_after=1000,
        tokens_saved=0,
        compression_ratio=1.0,
        compression_strategy="test",
        compression_time_ms=0.0,
    )

    assert metrics.messages_removed == 0


def test_compression_metrics_validation_negative_values():
    """Test validation rejects negative values."""
    with pytest.raises(ValidationError):
        CompressionMetrics(
            messages_before=-1,
            messages_after=10,
            estimated_tokens_before=5000,
            estimated_tokens_after=1000,
            tokens_saved=4000,
            compression_ratio=0.2,
            compression_strategy="test",
            compression_time_ms=1.0,
        )


def test_compression_metrics_validation_ratio_out_of_range():
    """Test validation rejects compression_ratio outside [0.0, 1.0]."""
    with pytest.raises(ValidationError) as exc_info:
        CompressionMetrics(
            messages_before=50,
            messages_after=10,
            estimated_tokens_before=5000,
            estimated_tokens_after=1000,
            tokens_saved=4000,
            compression_ratio=1.5,  # Invalid: > 1.0
            compression_strategy="test",
            compression_time_ms=1.0,
        )

    assert "compression_ratio" in str(exc_info.value).lower()


def test_compression_metrics_validation_ratio_negative():
    """Test validation rejects negative compression_ratio."""
    with pytest.raises(ValidationError) as exc_info:
        CompressionMetrics(
            messages_before=50,
            messages_after=10,
            estimated_tokens_before=5000,
            estimated_tokens_after=1000,
            tokens_saved=4000,
            compression_ratio=-0.1,  # Invalid: < 0.0
            compression_strategy="test",
            compression_time_ms=1.0,
        )

    assert "compression_ratio" in str(exc_info.value).lower()


def test_compression_metrics_edge_values():
    """Test CompressionMetrics with edge values."""
    # Perfect compression (ratio = 0.0)
    metrics = CompressionMetrics(
        messages_before=100,
        messages_after=0,
        estimated_tokens_before=10000,
        estimated_tokens_after=0,
        tokens_saved=10000,
        compression_ratio=0.0,
        compression_strategy="perfect",
        compression_time_ms=0.0,
    )
    assert metrics.percentage_reduction == 100.0

    # No compression (ratio = 1.0)
    metrics = CompressionMetrics(
        messages_before=100,
        messages_after=100,
        estimated_tokens_before=10000,
        estimated_tokens_after=10000,
        tokens_saved=0,
        compression_ratio=1.0,
        compression_strategy="none",
        compression_time_ms=0.0,
    )
    assert metrics.percentage_reduction == 0.0


# Test MemoryCompressor Protocol


def test_memory_compressor_protocol_conformance():
    """Test that a class conforming to MemoryCompressor protocol is recognized."""

    class ConformingCompressor:
        @property
        def name(self) -> str:
            return "conforming"

        async def should_compress(
            self, messages: Sequence[Any], estimated_tokens: int
        ) -> bool:
            return True

        async def compress(
            self, messages: Sequence[Any]
        ) -> tuple[list[Any], CompressionMetrics]:
            return [], CompressionMetrics(
                messages_before=0,
                messages_after=0,
                estimated_tokens_before=0,
                estimated_tokens_after=0,
                tokens_saved=0,
                compression_ratio=0.0,
                compression_strategy="test",
                compression_time_ms=0.0,
            )

    compressor = ConformingCompressor()
    assert isinstance(compressor, MemoryCompressor)


def test_memory_compressor_protocol_non_conforming():
    """Test that a class not conforming to protocol is not recognized."""

    class NonConformingCompressor:
        pass

    compressor = NonConformingCompressor()
    assert not isinstance(compressor, MemoryCompressor)


# Test BaseMemoryCompressor


class MockCompressor(BaseMemoryCompressor):
    """Mock compressor for testing BaseMemoryCompressor."""

    @property
    def name(self) -> str:
        """Return the name of this mock compressor."""
        return "mock"

    async def compress(
        self, messages: Sequence[Any]
    ) -> tuple[list[Any], CompressionMetrics]:
        """Mock compression that keeps only recent messages."""
        _, _, recent = self._partition_messages(messages)
        metrics = CompressionMetrics(
            messages_before=len(messages),
            messages_after=len(recent),
            estimated_tokens_before=self._estimate_tokens(messages),
            estimated_tokens_after=self._estimate_tokens(recent),
            tokens_saved=0,
            compression_ratio=len(recent) / len(messages) if messages else 1.0,
            compression_strategy=self.name,
            compression_time_ms=0.0,
        )
        return list(recent), metrics


def test_base_compressor_initialization_defaults():
    """Test BaseMemoryCompressor initialization with defaults."""
    compressor = MockCompressor()

    assert compressor.max_tokens == 8000
    assert compressor.preserve_system_messages is True
    assert compressor.preserve_recent_messages == 5


def test_base_compressor_initialization_custom():
    """Test BaseMemoryCompressor initialization with custom values."""
    compressor = MockCompressor(
        max_tokens=4000,
        preserve_system_messages=False,
        preserve_recent_messages=10,
    )

    assert compressor.max_tokens == 4000
    assert compressor.preserve_system_messages is False
    assert compressor.preserve_recent_messages == 10


@pytest.mark.asyncio
async def test_base_compressor_should_compress_under_limit():
    """Test should_compress returns False when under limit."""
    compressor = MockCompressor(max_tokens=1000)

    messages = [MagicMock()]
    should = await compressor.should_compress(messages, estimated_tokens=500)

    assert should is False


@pytest.mark.asyncio
async def test_base_compressor_should_compress_over_limit():
    """Test should_compress returns True when over limit."""
    compressor = MockCompressor(max_tokens=1000)

    messages = [MagicMock()]
    should = await compressor.should_compress(messages, estimated_tokens=1500)

    assert should is True


@pytest.mark.asyncio
async def test_base_compressor_should_compress_at_limit():
    """Test should_compress returns False when exactly at limit."""
    compressor = MockCompressor(max_tokens=1000)

    messages = [MagicMock()]
    should = await compressor.should_compress(messages, estimated_tokens=1000)

    assert should is False


def test_base_compressor_estimate_tokens_simple_string():
    """Test token estimation with simple string content."""
    compressor = MockCompressor()

    # Create mock message with string content
    msg = MagicMock()
    msg.parts = None  # No parts attribute, fallback to .content
    msg.content = "This is a test message"  # 22 chars = ~5 tokens
    messages = [msg]

    tokens = compressor._estimate_tokens(messages)

    # 22 chars / 4 = 5 tokens
    assert tokens == 5


def test_base_compressor_estimate_tokens_multiple_messages():
    """Test token estimation with multiple messages."""
    compressor = MockCompressor()

    msg1 = MagicMock()
    msg1.parts = None  # No parts attribute, fallback to .content
    msg1.content = "First message"  # 13 chars

    msg2 = MagicMock()
    msg2.parts = None  # No parts attribute, fallback to .content
    msg2.content = "Second message here"  # 19 chars

    messages = [msg1, msg2]
    tokens = compressor._estimate_tokens(messages)

    # (13 + 19) / 4 = 8 tokens
    assert tokens == 8


def test_base_compressor_estimate_tokens_list_content():
    """Test token estimation with list content (multi-part messages)."""
    compressor = MockCompressor()

    msg = MagicMock()
    msg.parts = None  # No parts attribute, fallback to .content
    part1 = MagicMock()
    part1.text = "Part one"  # 8 chars
    part2 = MagicMock()
    part2.text = "Part two"  # 8 chars
    msg.content = [part1, part2]

    messages = [msg]
    tokens = compressor._estimate_tokens(messages)

    # 16 chars / 4 = 4 tokens
    assert tokens == 4


def test_base_compressor_estimate_tokens_no_content():
    """Test token estimation with messages without content attribute."""
    compressor = MockCompressor()

    msg = MagicMock(spec=[])  # No attributes
    messages = [msg]

    tokens = compressor._estimate_tokens(messages)

    assert tokens == 0


def test_base_compressor_partition_messages_all_recent():
    """Test message partitioning when all messages are recent."""
    compressor = MockCompressor(preserve_recent_messages=10)

    messages = [MagicMock() for _ in range(5)]
    system, compressible, recent = compressor._partition_messages(messages)

    assert len(system) == 0
    assert len(compressible) == 0
    assert len(recent) == 5


def test_base_compressor_partition_messages_with_system():
    """Test message partitioning with system messages."""
    compressor = MockCompressor(preserve_recent_messages=2)

    msg1 = MagicMock()
    msg1.role = "system"
    msg2 = MagicMock()
    msg2.role = "user"
    msg3 = MagicMock()
    msg3.role = "assistant"
    msg4 = MagicMock()
    msg4.role = "user"

    messages = [msg1, msg2, msg3, msg4]
    system, compressible, recent = compressor._partition_messages(messages)

    # msg1 is system, msg2 is compressible, msg3-4 are recent (last 2)
    assert len(system) == 1
    assert system[0] == msg1
    assert len(compressible) == 1
    assert compressible[0] == msg2
    assert len(recent) == 2
    assert recent == [msg3, msg4]


def test_base_compressor_partition_messages_no_system_preservation():
    """Test message partitioning when system preservation is disabled."""
    compressor = MockCompressor(
        preserve_system_messages=False, preserve_recent_messages=2
    )

    msg1 = MagicMock()
    msg1.role = "system"
    msg2 = MagicMock()
    msg2.role = "user"
    msg3 = MagicMock()
    msg3.role = "assistant"

    messages = [msg1, msg2, msg3]
    system, compressible, recent = compressor._partition_messages(messages)

    # msg1 is compressible (system preservation off), msg2-3 are recent
    assert len(system) == 0
    assert len(compressible) == 1
    assert compressible[0] == msg1
    assert len(recent) == 2


def test_base_compressor_is_system_message_role():
    """Test system message detection via role attribute."""
    compressor = MockCompressor()

    msg = MagicMock()
    msg.role = "system"

    assert compressor._is_system_message(msg) is True


def test_base_compressor_is_system_message_kind():
    """Test system message detection via kind attribute."""
    compressor = MockCompressor()

    msg = MagicMock(spec=["kind"])
    msg.kind = "system"

    assert compressor._is_system_message(msg) is True


def test_base_compressor_is_system_message_not_system():
    """Test system message detection returns False for non-system."""
    compressor = MockCompressor()

    msg = MagicMock()
    msg.role = "user"

    assert compressor._is_system_message(msg) is False


def test_base_compressor_is_system_message_no_attributes():
    """Test system message detection with message without role/kind."""
    compressor = MockCompressor()

    msg = MagicMock(spec=[])

    assert compressor._is_system_message(msg) is False


@pytest.mark.asyncio
async def test_base_compressor_protocol_conformance():
    """Test that BaseMemoryCompressor conforms to MemoryCompressor protocol."""
    compressor = MockCompressor()

    assert isinstance(compressor, MemoryCompressor)


@pytest.mark.asyncio
async def test_base_compressor_compress_integration():
    """Test complete compression flow with BaseMemoryCompressor."""
    compressor = MockCompressor(preserve_recent_messages=3)

    # Create 10 messages
    messages = [MagicMock() for _ in range(10)]
    for i, msg in enumerate(messages):
        msg.content = f"Message {i}"

    compressed, metrics = await compressor.compress(messages)

    # Should keep only 3 recent messages
    assert len(compressed) == 3
    assert metrics.messages_before == 10
    assert metrics.messages_after == 3
    assert metrics.compression_strategy == "mock"
