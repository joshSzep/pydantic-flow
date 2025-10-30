"""Memory management for pydantic-flow.

This package provides conversation memory and configuration
for managing state across flows and agents.
"""

from __future__ import annotations

from pydantic_flow.memory.compression import BaseMemoryCompressor
from pydantic_flow.memory.compression import CompressionMetrics
from pydantic_flow.memory.compression import MemoryCompressor
from pydantic_flow.memory.compressors import HybridCompressor
from pydantic_flow.memory.compressors import SlidingWindowCompressor
from pydantic_flow.memory.compressors import SummarizationCompressor
from pydantic_flow.memory.config import MemoryConfig
from pydantic_flow.memory.events import MemoryCompressionComplete
from pydantic_flow.memory.events import MemoryCompressionPending
from pydantic_flow.memory.memory import ConversationMemory
from pydantic_flow.memory.memory import MemoryProtocol
from pydantic_flow.memory.memory import ReadOnlyConversationMemory
from pydantic_flow.memory.memory import ReadOnlyMemoryError
from pydantic_flow.memory.memory import _active_flow_memory
from pydantic_flow.memory.memory import _memory_event_emitter
from pydantic_flow.memory.modes import MemoryMode

__all__ = [
    "BaseMemoryCompressor",
    "CompressionMetrics",
    "ConversationMemory",
    "HybridCompressor",
    "MemoryCompressionComplete",
    "MemoryCompressionPending",
    "MemoryCompressor",
    "MemoryConfig",
    "MemoryMode",
    "MemoryProtocol",
    "ReadOnlyConversationMemory",
    "ReadOnlyMemoryError",
    "SlidingWindowCompressor",
    "SummarizationCompressor",
    "_active_flow_memory",
    "_memory_event_emitter",
]
