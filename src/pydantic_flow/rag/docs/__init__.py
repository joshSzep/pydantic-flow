"""Document and metadata types for RAG operations."""

from typing import Any

from pydantic import BaseModel
from pydantic import Field


class Metadata(BaseModel):
    """Metadata associated with a document.

    Attributes:
        source: Source identifier (file path, URL, etc).
        created_at: Creation timestamp.
        chunk_index: Index of this chunk within the source.
        total_chunks: Total number of chunks from this source.
        extra: Additional arbitrary metadata fields.

    """

    source: str | None = None
    created_at: str | None = None
    chunk_index: int | None = None
    total_chunks: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    """A document for RAG operations.

    Attributes:
        id: Unique identifier for this document.
        content: The document text content.
        metadata: Associated metadata.
        embedding: Optional pre-computed embedding vector.

    """

    id: str
    content: str
    metadata: Metadata = Field(default_factory=Metadata)
    embedding: list[float] | None = None
