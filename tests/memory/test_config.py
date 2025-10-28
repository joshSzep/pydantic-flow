"""Tests for MemoryConfig class."""

from pydantic import ValidationError
import pytest

from pydantic_flow.memory import MemoryConfig
from pydantic_flow.memory.compressors import SlidingWindowCompressor


def test_memory_config_defaults():
    """Test MemoryConfig has correct default values."""
    config = MemoryConfig()

    assert config.enable_conversation_memory is True
    assert config.compressor is None
    assert config.emit_compression_events is True


def test_memory_config_custom_values():
    """Test MemoryConfig accepts custom values."""
    config = MemoryConfig(
        enable_conversation_memory=False,
    )

    assert config.enable_conversation_memory is False


def test_memory_config_validation():
    """Test MemoryConfig validates field types."""
    # Should fail with wrong types
    with pytest.raises(ValidationError):
        MemoryConfig(enable_conversation_memory="not_a_bool")  # type: ignore[arg-type]


def test_memory_config_with_compressor():
    """Test MemoryConfig accepts compressor instance."""
    compressor = SlidingWindowCompressor(window_size=10, max_tokens=1000)
    config = MemoryConfig(compressor=compressor)

    assert config.compressor is compressor
    assert isinstance(config.compressor, SlidingWindowCompressor)
    assert config.compressor.window_size == 10  # type: ignore[union-attr]
    assert config.compressor.max_tokens == 1000  # type: ignore[union-attr]


def test_memory_config_with_emit_compression_events():
    """Test MemoryConfig emit_compression_events field."""
    config = MemoryConfig(emit_compression_events=False)
    assert config.emit_compression_events is False

    config = MemoryConfig(emit_compression_events=True)
    assert config.emit_compression_events is True


def test_memory_config_arbitrary_types_allowed():
    """Test MemoryConfig allows arbitrary types for compressor field."""
    # This should not raise a ValidationError despite compressor being a complex type
    compressor = SlidingWindowCompressor(window_size=5, max_tokens=500)
    config = MemoryConfig(
        enable_conversation_memory=True,
        compressor=compressor,
        emit_compression_events=True,
    )

    assert config.compressor is not None
    assert isinstance(config.compressor, SlidingWindowCompressor)
