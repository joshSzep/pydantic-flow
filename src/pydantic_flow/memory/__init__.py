"""Memory management for pydantic-flow.

This package provides conversation memory and configuration
for managing state across flows and agents.
"""

from __future__ import annotations

from pydantic_flow.memory.config import MemoryConfig
from pydantic_flow.memory.memory import ConversationMemory
from pydantic_flow.memory.memory import MemoryProtocol
from pydantic_flow.memory.memory import ReadOnlyConversationMemory
from pydantic_flow.memory.memory import ReadOnlyMemoryError
from pydantic_flow.memory.memory import _active_flow_memory
from pydantic_flow.memory.modes import MemoryMode

__all__ = [
    "ConversationMemory",
    "MemoryConfig",
    "MemoryMode",
    "MemoryProtocol",
    "ReadOnlyConversationMemory",
    "ReadOnlyMemoryError",
    "_active_flow_memory",
]
