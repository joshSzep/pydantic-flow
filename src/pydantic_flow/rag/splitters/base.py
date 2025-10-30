"""Base types and interfaces for document splitters."""

from typing import Any
from typing import Protocol

from pydantic import BaseModel
from pydantic import Field


class ChunkMetadata(BaseModel):
    """Metadata for a document chunk.

    Attributes:
        source_id: Identifier of the source document.
        chunk_index: Sequential index of this chunk.
        byte_start: Starting byte offset in source document.
        byte_end: Ending byte offset in source document.
        token_count: Approximate token count if available.
        heading_path: List of heading levels for Markdown chunks.
        extra: Additional arbitrary metadata.

    """

    source_id: str
    chunk_index: int
    byte_start: int | None = None
    byte_end: int | None = None
    token_count: int | None = None
    heading_path: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    """A chunk of a document.

    Attributes:
        id: Unique identifier for this chunk.
        text: The chunk text content.
        metadata: Associated chunk metadata.
        embedding: Optional pre-computed embedding vector.

    """

    id: str
    text: str
    metadata: ChunkMetadata
    embedding: list[float] | None = None


class SplitConfig(BaseModel):
    """Configuration for document splitting.

    Attributes:
        splitter_type: Splitter type ('token', 'sentence', 'markdown').
        max_tokens: Maximum tokens per chunk (overrides max_chars if set).
        max_chars: Maximum characters per chunk.
        overlap: Overlap size in tokens (if max_tokens) or chars (if max_chars).
        min_chunk_chars: Minimum chunk size in characters.
        preserve_newlines: Whether to preserve paragraph boundaries.
        return_metadata: Whether to include metadata in chunks.

    """

    splitter_type: str = "token"
    max_tokens: int | None = None
    max_chars: int = 1000
    overlap: int = 100
    min_chunk_chars: int = 50
    preserve_newlines: bool = True
    return_metadata: bool = True


class TokenCounter(Protocol):
    """Protocol for token counting implementations."""

    def count(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count tokens in.

        Returns:
            Token count.

        """
        ...


class SimpleTokenCounter:
    """Simple token counter approximation.

    Estimates tokens by dividing character count by average chars per token.
    Does not require any external dependencies.

    Attributes:
        chars_per_token: Average characters per token (default: 4).

    """

    def __init__(self, chars_per_token: float = 4.0) -> None:
        """Initialize simple token counter.

        Args:
            chars_per_token: Average characters per token for estimation.

        """
        self.chars_per_token = chars_per_token

    def count(self, text: str) -> int:
        """Count tokens by character approximation.

        Args:
            text: Text to count tokens in.

        Returns:
            Estimated token count.

        """
        return max(1, int(len(text) / self.chars_per_token))


class Splitter(Protocol):
    """Protocol for document splitters."""

    def split(
        self,
        text: str,
        source_id: str,
        config: SplitConfig,
        token_counter: TokenCounter | None = None,
    ) -> list[DocumentChunk]:
        """Split document into chunks.

        Args:
            text: Document text to split.
            source_id: Identifier for the source document.
            config: Splitting configuration.
            token_counter: Optional token counter (uses SimpleTokenCounter if None).

        Returns:
            List of document chunks.

        """
        ...
