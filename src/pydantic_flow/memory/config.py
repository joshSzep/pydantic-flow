"""Memory configuration for pydantic-flow.

This module provides configuration options for conversation memory
and global memory management.
"""

from __future__ import annotations

from pydantic import BaseModel
from pydantic import Field


class MemoryConfig(BaseModel):
    """Configuration for memory behavior in flows and agents.

    This model defines how conversation memory is managed within
    flows and agents, including enabling/disabling features and
    setting size limits.
    """

    enable_conversation_memory: bool = Field(
        default=True,
        description="Enable conversation memory for this flow/agent",
    )

    max_messages: int | None = Field(
        default=None,
        description="Maximum number of messages to retain (None = unlimited)",
    )

    auto_trim: bool = Field(
        default=False,
        description="Automatically trim old messages when max_messages is reached",
    )
