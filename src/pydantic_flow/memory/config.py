"""Memory configuration for pydantic-flow.

This module provides configuration options for conversation memory
and global memory management.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

if TYPE_CHECKING:
    from pydantic_flow.memory.compression import MemoryCompressor
else:
    # Import at runtime to avoid circular import
    from pydantic_flow.memory.compression import MemoryCompressor


class MemoryConfig(BaseModel):
    """Configuration for memory behavior in flows and agents.

    This model defines how conversation memory is managed within flows and agents.

    Supports pluggable compression strategies via the compressor field
    for automatic context management when approaching token limits.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    enable_conversation_memory: bool = Field(
        default=True,
        description="Enable conversation memory for this flow/agent",
    )

    compressor: MemoryCompressor | None = Field(
        default=None,
        description=(
            "Optional memory compressor for automatic context management. "
            "When set, the compressor will automatically compress conversation "
            "history when token limits are approached, enabling longer conversations "
            "without losing context."
        ),
    )

    emit_compression_events: bool = Field(
        default=True,
        description=(
            "Whether to emit compression events (MemoryCompressionPending, "
            "MemoryCompressionComplete) during streaming. Set to False to "
            "disable compression event emission."
        ),
    )
