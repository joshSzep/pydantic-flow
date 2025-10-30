"""Vector-based retriever."""

from typing import Any

from pydantic_flow.rag.docs import Document
from pydantic_flow.rag.embeddings.base import EmbeddingProvider
from pydantic_flow.rag.retrievers.base import Retriever
from pydantic_flow.rag.vectors.base import VectorStore


class VectorRetriever(Retriever):
    """Retriever that uses embeddings and vector store for semantic search.

    Attributes:
        embedding_provider: Provider for generating query embeddings.
        vector_store: Vector store for similarity search.
        default_k: Default number of results to return.
        filter: Optional metadata filter to apply.

    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        default_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> None:
        """Initialize vector retriever.

        Args:
            embedding_provider: Provider for embeddings.
            vector_store: Vector store instance.
            default_k: Default number of results.
            filter: Optional metadata filter.

        """
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.default_k = default_k
        self.filter = filter

    async def retrieve(self, query: str, k: int | None = None) -> list[Document]:
        """Retrieve documents using semantic search.

        Args:
            query: Query string.
            k: Number of documents to retrieve (uses default_k if None).

        Returns:
            List of retrieved documents.

        """
        k = k or self.default_k

        embeddings = await self.embedding_provider.embed([query])
        query_embedding = embeddings[0]

        results = await self.vector_store.query(
            vector=query_embedding,
            k=k,
            filter=self.filter,
        )

        return [result.document for result in results]
