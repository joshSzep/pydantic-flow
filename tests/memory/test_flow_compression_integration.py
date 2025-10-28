"""Tests for Flow integration with memory compression.

This module tests that Flow properly initializes ConversationMemory
with the compressor from MemoryConfig.
"""

from pydantic import BaseModel

from pydantic_flow.flow import Flow
from pydantic_flow.memory import MemoryConfig
from pydantic_flow.memory.compressors import SlidingWindowCompressor


class FlowInput(BaseModel):
    """Test input model."""

    value: str


class FlowOutput(BaseModel):
    """Test output model."""

    result: str


def test_flow_creates_memory_with_compressor():
    """Test that Flow passes compressor to ConversationMemory."""
    compressor = SlidingWindowCompressor(window_size=10, max_tokens=1000)
    config = MemoryConfig(
        enable_conversation_memory=True,
        compressor=compressor,
    )

    flow = Flow(input_type=FlowInput, output_type=FlowOutput, memory_config=config)

    # Verify memory was created
    assert flow._conversation_memory is not None
    # Verify compressor was passed through
    assert flow._conversation_memory._compressor is compressor


def test_flow_creates_memory_without_compressor():
    """Test that Flow works with memory enabled but no compressor."""
    config = MemoryConfig(
        enable_conversation_memory=True,
        compressor=None,
    )

    flow = Flow(input_type=FlowInput, output_type=FlowOutput, memory_config=config)

    # Verify memory was created
    assert flow._conversation_memory is not None
    # Verify no compressor
    assert flow._conversation_memory._compressor is None


def test_flow_no_memory_when_disabled():
    """Test that Flow doesn't create memory when disabled."""
    config = MemoryConfig(
        enable_conversation_memory=False,
    )

    flow = Flow(input_type=FlowInput, output_type=FlowOutput, memory_config=config)

    # Verify memory was not created
    assert flow._conversation_memory is None


def test_flow_default_config_has_no_compressor():
    """Test that default MemoryConfig has no compressor."""
    flow = Flow(input_type=FlowInput, output_type=FlowOutput)

    # Should have memory (enabled by default)
    assert flow._conversation_memory is not None
    # Should not have compressor (default is None)
    assert flow._conversation_memory._compressor is None


def test_flow_emit_compression_events_setting():
    """Test that emit_compression_events setting is accessible."""
    config = MemoryConfig(
        enable_conversation_memory=True,
        emit_compression_events=False,
    )

    flow = Flow(input_type=FlowInput, output_type=FlowOutput, memory_config=config)

    # Verify config is accessible
    assert flow.memory_config.emit_compression_events is False


def test_flow_compressor_isolation():
    """Test that different flows have independent compressors."""
    compressor1 = SlidingWindowCompressor(window_size=5, max_tokens=500)
    compressor2 = SlidingWindowCompressor(window_size=10, max_tokens=1000)

    config1 = MemoryConfig(enable_conversation_memory=True, compressor=compressor1)
    config2 = MemoryConfig(enable_conversation_memory=True, compressor=compressor2)

    flow1 = Flow(input_type=FlowInput, output_type=FlowOutput, memory_config=config1)
    flow2 = Flow(input_type=FlowInput, output_type=FlowOutput, memory_config=config2)

    # Verify each flow has its own compressor
    assert flow1._conversation_memory._compressor is compressor1  # type: ignore[union-attr]
    assert flow2._conversation_memory._compressor is compressor2  # type: ignore[union-attr]
    # Verify compressors are different
    comp1 = flow1._conversation_memory._compressor  # type: ignore[union-attr]
    comp2 = flow2._conversation_memory._compressor  # type: ignore[union-attr]
    assert comp1 is not comp2
