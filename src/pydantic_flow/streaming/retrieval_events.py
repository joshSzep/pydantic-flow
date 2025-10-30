"""Retrieval-related streaming events.

This module defines events related to information retrieval from
search systems, databases, or other data sources.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from pydantic_flow.streaming.base import ProgressItem
from pydantic_flow.streaming.base import ProgressType


class RetrievalItem(ProgressItem):
    """A single retrieved item from search or database.

    Attributes:
        item_id: Identifier for the retrieved item.
        content: The retrieved content.
        score: Optional relevance score.
        metadata: Optional additional metadata.

    """

    type: ProgressType = ProgressType.RETRIEVAL
    item_id: str = ""
    content: Any = None
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
