"""Tests for MemoryConfig class."""

from pydantic import ValidationError
import pytest

from pydantic_flow.memory import MemoryConfig


def test_memory_config_defaults():
    """Test MemoryConfig has correct default values."""
    config = MemoryConfig()

    assert config.enable_conversation_memory is True
    assert config.max_messages is None
    assert config.auto_trim is False


def test_memory_config_custom_values():
    """Test MemoryConfig accepts custom values."""
    config = MemoryConfig(
        enable_conversation_memory=False,
        max_messages=100,
        auto_trim=True,
    )

    assert config.enable_conversation_memory is False
    assert config.max_messages == 100
    assert config.auto_trim is True


def test_memory_config_partial_override():
    """Test MemoryConfig can override specific fields."""
    config = MemoryConfig(max_messages=50)

    assert config.enable_conversation_memory is True
    assert config.max_messages == 50
    assert config.auto_trim is False


def test_memory_config_validation():
    """Test MemoryConfig validates field types."""
    # Should fail with wrong types
    with pytest.raises(ValidationError):
        MemoryConfig(enable_conversation_memory="not_a_bool")  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        MemoryConfig(max_messages="not_an_int")  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        MemoryConfig(auto_trim="not_a_bool")  # type: ignore[arg-type]
