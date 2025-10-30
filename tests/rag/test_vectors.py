"""Tests for RAG vector stores."""

import pytest

from pydantic_flow.rag.docs import Document
from pydantic_flow.rag.docs import Metadata
from pydantic_flow.rag.vectors.hnsw import HNSWMemoryStore


@pytest.mark.asyncio
async def test_hnsw_upsert_query():
    """Test HNSW upsert and query operations."""
    store = HNSWMemoryStore(dim=128)

    doc1 = Document(
        id="doc1",
        content="Hello world",
        metadata=Metadata(source="test.txt"),
    )
    doc2 = Document(
        id="doc2",
        content="Goodbye world",
        metadata=Metadata(source="test.txt"),
    )

    vec1 = [0.1] * 128
    vec2 = [0.2] * 128

    await store.upsert([("doc1", vec1, doc1), ("doc2", vec2, doc2)])

    results = await store.query(vector=vec1, k=2)

    assert len(results) == 2
    assert results[0].id == "doc1"
    assert results[0].document.content == "Hello world"
    assert results[0].score > 0.9


@pytest.mark.asyncio
async def test_hnsw_delete():
    """Test HNSW delete operation."""
    store = HNSWMemoryStore(dim=64)

    doc1 = Document(id="doc1", content="Test 1")
    doc2 = Document(id="doc2", content="Test 2")
    doc3 = Document(id="doc3", content="Test 3")

    vec = [0.5] * 64

    await store.upsert([
        ("doc1", vec, doc1),
        ("doc2", vec, doc2),
        ("doc3", vec, doc3),
    ])

    await store.delete(["doc2"])

    results = await store.query(vector=vec, k=5)
    ids = {r.id for r in results}

    assert "doc2" not in ids
    assert "doc1" in ids
    assert "doc3" in ids


@pytest.mark.asyncio
async def test_hnsw_empty_query():
    """Test querying empty store."""
    store = HNSWMemoryStore(dim=128)

    results = await store.query(vector=[0.1] * 128, k=5)
    assert len(results) == 0


@pytest.mark.asyncio
async def test_hnsw_metadata_filter():
    """Test HNSW with metadata filtering."""
    store = HNSWMemoryStore(dim=32)

    doc1 = Document(
        id="doc1",
        content="Test 1",
        metadata=Metadata(source="file1.txt"),
    )
    doc2 = Document(
        id="doc2",
        content="Test 2",
        metadata=Metadata(source="file2.txt"),
    )

    vec = [0.3] * 32

    await store.upsert([("doc1", vec, doc1), ("doc2", vec, doc2)])

    results = await store.query(vector=vec, k=5, filter={"source": "file1.txt"})

    assert len(results) == 1
    assert results[0].id == "doc1"


@pytest.mark.asyncio
async def test_embedding_dim():
    """Test embedding dimension retrieval."""
    store = HNSWMemoryStore(dim=256)
    assert store.embedding_dim() == 256


@pytest.mark.asyncio
async def test_pgvector_store():
    """Test PostgreSQL pgvector store."""
    from unittest.mock import AsyncMock
    from unittest.mock import MagicMock
    from unittest.mock import patch

    from pydantic_flow.rag.vectors.pgvector import PostgresPGVectorStore

    # Mock asyncpg connection
    mock_row = {
        "id": "doc1",
        "content": "Test document",
        "metadata": "{}",
        "score": 0.9,
    }

    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock()
    mock_conn.fetch = AsyncMock(return_value=[mock_row])

    # Mock pool acquire context manager
    mock_acquire = AsyncMock()
    mock_acquire.__aenter__.return_value = mock_conn
    mock_acquire.__aexit__.return_value = None

    mock_pool = MagicMock()
    mock_pool.acquire.return_value = mock_acquire
    mock_pool.close = AsyncMock()

    mock_create_pool = AsyncMock(return_value=mock_pool)

    with patch(
        "pydantic_flow.rag.vectors.pgvector.asyncpg.create_pool", mock_create_pool
    ):
        conn_string = "postgresql://localhost/testdb"
        store = PostgresPGVectorStore(connection_string=conn_string, dim=128)

        await store.initialize()

        doc = Document(id="doc1", content="Test document")
        vec = [0.5] * 128

        await store.upsert([("doc1", vec, doc)])

        results = await store.query(vector=vec, k=1)

    assert len(results) == 1
    assert results[0].id == "doc1"

    await store.delete(["doc1"])
    await store.close()
