"""HNSW in-memory vector store using hnswlib."""

from typing import Any

import hnswlib  # type: ignore
import numpy as np

from pydantic_flow.rag.docs import Document
from pydantic_flow.rag.vectors.base import SearchResult
from pydantic_flow.rag.vectors.base import VectorStore


class HNSWMemoryStore(VectorStore):
    """In-memory HNSW vector store.

    Fast approximate nearest neighbor search with no external dependencies.

    Attributes:
        dim: Embedding dimension.
        max_elements: Maximum number of vectors to store.
        index: hnswlib index.
        id_to_doc: Mapping from ID to document.
        next_idx: Next internal index value.
        id_to_idx: Mapping from document ID to internal index.

    """

    def __init__(
        self,
        dim: int,
        max_elements: int = 10000,
        ef_construction: int = 200,
        M: int = 16,
    ) -> None:
        """Initialize HNSW memory store.

        Args:
            dim: Embedding dimension.
            max_elements: Maximum number of vectors.
            ef_construction: Construction time parameter (higher = better recall).
            M: Number of bi-directional links (higher = better recall).

        """
        self.dim = dim
        self.max_elements = max_elements
        self.index = hnswlib.Index(space="cosine", dim=dim)
        self.index.init_index(
            max_elements=max_elements, ef_construction=ef_construction, M=M
        )
        self.index.set_ef(50)
        self.id_to_doc: dict[str, Document] = {}
        self.next_idx = 0
        self.id_to_idx: dict[str, int] = {}

    async def upsert(self, items: list[tuple[str, list[float], Document]]) -> None:
        """Upsert vectors and documents.

        Args:
            items: List of (id, vector, document) tuples.

        """
        for doc_id, vector, document in items:
            if doc_id in self.id_to_idx:
                idx = self.id_to_idx[doc_id]
            else:
                idx = self.next_idx
                self.id_to_idx[doc_id] = idx
                self.next_idx += 1

            self.index.add_items(np.array([vector]), np.array([idx]))
            self.id_to_doc[doc_id] = document

    async def delete(self, ids: list[str]) -> None:
        """Delete vectors by ID.

        Args:
            ids: List of document IDs to delete.

        """
        for doc_id in ids:
            if doc_id in self.id_to_doc:
                del self.id_to_doc[doc_id]
            if doc_id in self.id_to_idx:
                idx = self.id_to_idx[doc_id]
                self.index.mark_deleted(idx)
                del self.id_to_idx[doc_id]

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
            filter: Optional metadata filter (ignored for HNSW).

        Returns:
            List of search results ordered by similarity.

        """
        if len(self.id_to_doc) == 0:
            return []

        # Limit k to the actual number of documents to avoid HNSW errors
        actual_k = min(k, len(self.id_to_doc))

        labels, distances = self.index.knn_query(np.array([vector]), k=actual_k)

        results = []
        idx_to_id = {idx: doc_id for doc_id, idx in self.id_to_idx.items()}

        for label, distance in zip(labels[0], distances[0], strict=False):
            if label in idx_to_id:
                doc_id = idx_to_id[label]
                document = self.id_to_doc.get(doc_id)
                if document:
                    score = 1.0 - distance
                    if filter is None or self._matches_filter(document, filter):
                        results.append(
                            SearchResult(
                                id=doc_id,
                                document=document,
                                score=score,
                                metadata={},
                            )
                        )

        return results

    def _matches_filter(self, document: Document, filter: dict[str, Any]) -> bool:
        """Check if document matches filter.

        Args:
            document: Document to check.
            filter: Filter dictionary.

        Returns:
            True if document matches all filter conditions.

        """
        for key, value in filter.items():
            if key == "source" and document.metadata.source != value:
                return False
            if key in document.metadata.extra and document.metadata.extra[key] != value:
                return False
        return True

    def embedding_dim(self) -> int:
        """Return embedding dimension.

        Returns:
            Embedding dimension.

        """
        return self.dim
