"""Tests for RAG retrievers."""

import pytest

from pydantic_flow.rag.docs import Document
from pydantic_flow.rag.docs import Metadata
from pydantic_flow.rag.embeddings.base import EmbeddingProvider
from pydantic_flow.rag.retrievers.vector import VectorRetriever
from pydantic_flow.rag.vectors.hnsw import HNSWMemoryStore


class MockEmbeddings(EmbeddingProvider):
    """Mock embedding provider."""

    def __init__(self, dim: int = 64):
        """Initialize mock embeddings."""
        self.dimension = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate mock embeddings."""
        return [[0.1 * (i + 1)] * self.dimension for i, _ in enumerate(texts)]

    def dim(self) -> int:
        """Return dimension."""
        return self.dimension


@pytest.mark.asyncio
async def test_vector_retriever():
    """Test vector retriever."""
    embeddings = MockEmbeddings(dim=64)
    store = HNSWMemoryStore(dim=64)

    doc1 = Document(id="doc1", content="Python programming")
    doc2 = Document(id="doc2", content="Machine learning")
    doc3 = Document(id="doc3", content="Data science")

    vec1 = [0.1] * 64
    vec2 = [0.2] * 64
    vec3 = [0.3] * 64

    await store.upsert([
        ("doc1", vec1, doc1),
        ("doc2", vec2, doc2),
        ("doc3", vec3, doc3),
    ])

    retriever = VectorRetriever(
        embedding_provider=embeddings,
        vector_store=store,
        default_k=2,
    )

    results = await retriever.retrieve("test query", k=2)

    assert len(results) == 2
    assert all(isinstance(doc, Document) for doc in results)


@pytest.mark.asyncio
async def test_vector_retriever_default_k():
    """Test retriever with default k."""
    embeddings = MockEmbeddings(dim=32)
    store = HNSWMemoryStore(dim=32)

    for i in range(5):
        doc = Document(id=f"doc{i}", content=f"Content {i}")
        vec = [0.1 * (i + 1)] * 32
        await store.upsert([(f"doc{i}", vec, doc)])

    retriever = VectorRetriever(
        embedding_provider=embeddings,
        vector_store=store,
        default_k=3,
    )

    results = await retriever.retrieve("query")

    assert len(results) == 3


@pytest.mark.asyncio
async def test_vector_retriever_with_filter():
    """Test retriever with metadata filter."""
    embeddings = MockEmbeddings(dim=32)
    store = HNSWMemoryStore(dim=32)

    doc1 = Document(
        id="doc1",
        content="Test 1",
        metadata=Metadata(source="source1"),
    )
    doc2 = Document(
        id="doc2",
        content="Test 2",
        metadata=Metadata(source="source2"),
    )

    vec = [0.5] * 32

    await store.upsert([("doc1", vec, doc1), ("doc2", vec, doc2)])

    retriever = VectorRetriever(
        embedding_provider=embeddings,
        vector_store=store,
        filter={"source": "source1"},
    )

    results = await retriever.retrieve("query", k=5)

    assert len(results) == 1
    assert results[0].id == "doc1"
