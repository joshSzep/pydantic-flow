"""Tests for RAG nodes."""

import pytest

from pydantic_flow.rag.docs import Document
from pydantic_flow.rag.embeddings.base import EmbeddingProvider
from pydantic_flow.rag.nodes.embedding import EmbeddingInput
from pydantic_flow.rag.nodes.embedding import EmbeddingNode
from pydantic_flow.rag.nodes.embedding import EmbeddingOutput
from pydantic_flow.rag.nodes.retriever import QueryInput
from pydantic_flow.rag.nodes.retriever import RetrievalResult
from pydantic_flow.rag.nodes.retriever import VectorRetrieverNode
from pydantic_flow.rag.retrievers.base import Retriever
from pydantic_flow.streaming.core_events import StreamEnd
from pydantic_flow.streaming.core_events import StreamStart
from pydantic_flow.streaming.retrieval_events import RetrievalItem
from tests.conftest import extract_result_from_stream


class MockRetriever(Retriever):
    """Mock retriever for testing."""

    async def retrieve(self, query: str, k: int) -> list[Document]:
        """Return mock documents."""
        return [
            Document(id=f"doc{i}", content=f"Result {i} for {query}") for i in range(k)
        ]


class MockEmbeddings(EmbeddingProvider):
    """Mock embedding provider."""

    def __init__(self, dim: int = 128):
        """Initialize mock embeddings."""
        self.dimension = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate mock embeddings."""
        return [[0.1] * self.dimension for _ in texts]

    def dim(self) -> int:
        """Return dimension."""
        return self.dimension


@pytest.mark.asyncio
async def test_vector_retriever_node():
    """Test VectorRetrieverNode streaming."""
    retriever = MockRetriever()
    node = VectorRetrieverNode(
        retriever=retriever,
        name="test-retriever",
        run_id="test-run",
    )

    query = QueryInput(query="test search", k=3)
    items = []

    async for item in node.astream(query):
        items.append(item)

    # StreamStart + 3 RetrievalItems + ToolResult + StreamEnd = 6 items
    assert len(items) == 6

    assert isinstance(items[0], StreamStart)
    assert items[0].node_id == "test-retriever"

    retrieval_items = [item for item in items if isinstance(item, RetrievalItem)]
    assert len(retrieval_items) == 3
    assert all(item.node_id == "test-retriever" for item in retrieval_items)

    assert isinstance(items[-1], StreamEnd)
    assert items[-1].node_id == "test-retriever"
    assert items[-1].result is not None
    assert isinstance(items[-1].result, RetrievalResult)
    assert len(items[-1].result.documents) == 3


@pytest.mark.asyncio
async def test_vector_retriever_node_run():
    """Test VectorRetrieverNode run method."""
    retriever = MockRetriever()
    node = VectorRetrieverNode(retriever=retriever)

    query = QueryInput(query="test", k=2)
    result = await extract_result_from_stream(node.astream(query))

    assert len(result.documents) == 2
    assert result.query == "test"


@pytest.mark.asyncio
async def test_embedding_node():
    """Test EmbeddingNode."""
    provider = MockEmbeddings(dim=256)
    node = EmbeddingNode(
        embedding_provider=provider,
        name="test-embedder",
        run_id="test-run",
    )

    input_data = EmbeddingInput(texts=["hello", "world"])
    items = []

    async for item in node.astream(input_data):
        items.append(item)

    # StreamStart + ToolResult + StreamEnd = 3 items
    assert len(items) == 3
    assert isinstance(items[0], StreamStart)
    assert isinstance(items[-1], StreamEnd)

    assert items[-1].result is not None
    assert isinstance(items[-1].result, EmbeddingOutput)
    assert items[-1].result.dimensions == 256
    assert len(items[-1].result.embeddings) == 2


@pytest.mark.asyncio
async def test_embedding_node_run():
    """Test EmbeddingNode run method."""
    provider = MockEmbeddings(dim=128)
    node = EmbeddingNode(embedding_provider=provider)

    input_data = EmbeddingInput(texts=["test"])
    result = await extract_result_from_stream(node.astream(input_data))

    assert len(result.embeddings) == 1
    assert len(result.embeddings[0]) == 128
    assert result.dimensions == 128
