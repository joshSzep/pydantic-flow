"""Vector store abstract base class."""

from abc import ABC
from abc import abstractmethod
from typing import Any

from pydantic import BaseModel

from pydantic_flow.rag.docs import Document


class SearchResult(BaseModel):
    """A single search result from a vector store.

    Attributes:
        id: Document identifier.
        document: The retrieved document.
        score: Similarity score (higher is better).
        metadata: Additional metadata.

    """

    id: str
    document: Document
    score: float
    metadata: dict[str, Any] = {}


class VectorStore(ABC):
    """Abstract base class for vector stores.

    Implementations must provide upsert, delete, query, and embedding_dim methods.
    """

    @abstractmethod
    async def upsert(self, items: list[tuple[str, list[float], Document]]) -> None:
        """Upsert vectors and documents.

        Args:
            items: List of (id, vector, document) tuples.

        """
        ...

    @abstractmethod
    async def delete(self, ids: list[str]) -> None:
        """Delete vectors by ID.

        Args:
            ids: List of document IDs to delete.

        """
        ...

    @abstractmethod
    async def query(
        self,
        vector: list[float],
        k: int,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Query for similar vectors.

        Args:
            vector: Query embedding vector.
            k: Number of results to return.
            filter: Optional metadata filter.

        Returns:
            List of search results ordered by similarity.

        """
        ...

    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the expected embedding dimension.

        Returns:
            Embedding dimension.

        """
        ...
