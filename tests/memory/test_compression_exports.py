"""Test that compression types are properly exported."""

from __future__ import annotations


def test_memory_package_exports() -> None:
    """Test that all compression types are exported from memory package."""
    from pydantic_flow.memory import BaseMemoryCompressor
    from pydantic_flow.memory import CompressionMetrics
    from pydantic_flow.memory import HybridCompressor
    from pydantic_flow.memory import MemoryCompressor
    from pydantic_flow.memory import SlidingWindowCompressor
    from pydantic_flow.memory import SummarizationCompressor

    assert BaseMemoryCompressor is not None
    assert CompressionMetrics is not None
    assert HybridCompressor is not None
    assert MemoryCompressor is not None
    assert SlidingWindowCompressor is not None
    assert SummarizationCompressor is not None


def test_main_package_compression_exports() -> None:
    """Test that compression types are exported from main package."""
    from pydantic_flow import BaseMemoryCompressor
    from pydantic_flow import CompressionMetrics
    from pydantic_flow import HybridCompressor
    from pydantic_flow import MemoryCompressor
    from pydantic_flow import SlidingWindowCompressor
    from pydantic_flow import SummarizationCompressor

    assert BaseMemoryCompressor is not None
    assert CompressionMetrics is not None
    assert HybridCompressor is not None
    assert MemoryCompressor is not None
    assert SlidingWindowCompressor is not None
    assert SummarizationCompressor is not None


def test_streaming_event_exports() -> None:
    """Test that compression events are exported from memory and main package."""
    from pydantic_flow import MemoryCompressionComplete
    from pydantic_flow import MemoryCompressionPending
    from pydantic_flow.memory import MemoryCompressionComplete as MemoryComplete
    from pydantic_flow.memory import MemoryCompressionPending as MemoryPending

    assert MemoryCompressionComplete is not None
    assert MemoryCompressionPending is not None
    assert MemoryCompressionComplete is MemoryComplete
    assert MemoryCompressionPending is MemoryPending


def test_all_exports_listed() -> None:
    """Test that __all__ lists include compression types."""
    import pydantic_flow
    import pydantic_flow.memory
    import pydantic_flow.streaming

    # Memory package
    assert "BaseMemoryCompressor" in pydantic_flow.memory.__all__
    assert "CompressionMetrics" in pydantic_flow.memory.__all__
    assert "HybridCompressor" in pydantic_flow.memory.__all__
    assert "MemoryCompressor" in pydantic_flow.memory.__all__
    assert "MemoryCompressionComplete" in pydantic_flow.memory.__all__
    assert "MemoryCompressionPending" in pydantic_flow.memory.__all__
    assert "SlidingWindowCompressor" in pydantic_flow.memory.__all__
    assert "SummarizationCompressor" in pydantic_flow.memory.__all__

    # Main package
    assert "BaseMemoryCompressor" in pydantic_flow.__all__
    assert "CompressionMetrics" in pydantic_flow.__all__
    assert "HybridCompressor" in pydantic_flow.__all__
    assert "MemoryCompressor" in pydantic_flow.__all__
    assert "SlidingWindowCompressor" in pydantic_flow.__all__
    assert "SummarizationCompressor" in pydantic_flow.__all__
    assert "MemoryCompressionComplete" in pydantic_flow.__all__
    assert "MemoryCompressionPending" in pydantic_flow.__all__
