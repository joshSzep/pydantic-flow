"""Base types and interfaces for document reranking."""

from typing import Protocol

from pydantic import BaseModel

from pydantic_flow.rag.splitters.base import DocumentChunk


class ScoredChunk(BaseModel):
    """A document chunk with relevance score.

    Attributes:
        chunk: The document chunk.
        score: Relevance score (higher is better).
        rank: Original rank position before reranking.

    """

    chunk: DocumentChunk
    score: float
    rank: int | None = None


class RerankConfig(BaseModel):
    """Configuration for reranking.

    Attributes:
        kind: Reranker type ('lexical', 'cohere', etc).
        top_n: Maximum number of results to return.
        model: Model name for provider-based rerankers.
        api_key: API key for provider-based rerankers.

    """

    kind: str = "lexical"
    top_n: int | None = None
    model: str | None = None
    api_key: str | None = None


class Reranker(Protocol):
    """Protocol for document rerankers."""

    def score(
        self,
        query: str,
        chunks: list[DocumentChunk],
    ) -> list[ScoredChunk]:
        """Score and rank chunks by relevance to query.

        Args:
            query: Query string.
            chunks: Document chunks to score.

        Returns:
            List of scored chunks sorted by relevance (highest first).

        """
        ...
