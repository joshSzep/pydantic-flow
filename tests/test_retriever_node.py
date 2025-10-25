"""Tests for RetrieverNode."""

from pydantic import BaseModel
import pytest

from pydantic_flow.nodes.retriever import RetrieverNode
from pydantic_flow.streaming.events import RetrievalItem
from pydantic_flow.streaming.events import StreamEnd
from pydantic_flow.streaming.events import StreamStart


class SearchQuery(BaseModel):
    """Test query model."""

    query: str


class SearchResult(BaseModel):
    """Test result model."""

    results: list[dict]


@pytest.mark.asyncio
async def test_retriever_node_basic():
    """Test basic RetrieverNode streaming."""

    async def mock_retriever(query: SearchQuery):
        """Mock retriever that yields search results."""
        yield {"id": "doc1", "content": "Result 1", "score": 0.9}
        yield {"id": "doc2", "content": "Result 2", "score": 0.8}
        yield {"id": "doc3", "content": "Result 3", "score": 0.7}

    node = RetrieverNode(
        retriever_fn=mock_retriever,
        name="test-retriever",
        run_id="test-run-123",
    )

    query = SearchQuery(query="test search")
    items = []

    async for item in node.astream(query):
        items.append(item)

    assert len(items) == 5  # Start + 3 retrieval items + End

    # Check StreamStart
    assert isinstance(items[0], StreamStart)
    assert items[0].run_id == "test-run-123"
    assert items[0].node_id == "test-retriever"
    assert items[0].input_preview == {"query": "test search"}

    # Check RetrievalItems
    assert isinstance(items[1], RetrievalItem)
    assert items[1].item_id == "doc1"
    assert items[1].content == "Result 1"
    assert items[1].score == 0.9
    assert items[1].run_id == "test-run-123"
    assert items[1].node_id == "test-retriever"

    assert isinstance(items[2], RetrievalItem)
    assert items[2].item_id == "doc2"
    assert items[2].content == "Result 2"
    assert items[2].score == 0.8

    assert isinstance(items[3], RetrievalItem)
    assert items[3].item_id == "doc3"
    assert items[3].content == "Result 3"
    assert items[3].score == 0.7

    # Check StreamEnd
    assert isinstance(items[4], StreamEnd)
    assert items[4].run_id == "test-run-123"
    assert items[4].node_id == "test-retriever"
    assert items[4].result_preview == {
        "results": [
            {"id": "doc1", "content": "Result 1", "score": 0.9},
            {"id": "doc2", "content": "Result 2", "score": 0.8},
            {"id": "doc3", "content": "Result 3", "score": 0.7},
        ]
    }


@pytest.mark.asyncio
async def test_retriever_node_generates_run_id():
    """Test RetrieverNode generates run_id if not provided."""

    async def mock_retriever(query: SearchQuery):
        yield {"id": "doc1", "content": "Result 1"}

    node = RetrieverNode(retriever_fn=mock_retriever, name="retriever")

    query = SearchQuery(query="test")
    items = []

    async for item in node.astream(query):
        items.append(item)

    # All items should have the same run_id
    run_ids = {item.run_id for item in items}
    assert len(run_ids) == 1


@pytest.mark.asyncio
async def test_retriever_node_with_metadata():
    """Test RetrieverNode handles metadata correctly."""

    async def mock_retriever(query: SearchQuery):
        yield {
            "id": "doc1",
            "content": "Result with metadata",
            "score": 0.95,
            "metadata": {"source": "database", "timestamp": "2024-01-01"},
        }

    node = RetrieverNode(retriever_fn=mock_retriever, name="retriever")

    query = SearchQuery(query="test")
    items = []

    async for item in node.astream(query):
        items.append(item)

    # Check the retrieval item has metadata
    retrieval_item = items[1]
    assert isinstance(retrieval_item, RetrievalItem)
    assert retrieval_item.metadata == {"source": "database", "timestamp": "2024-01-01"}


@pytest.mark.asyncio
async def test_retriever_node_without_metadata():
    """Test RetrieverNode handles missing metadata field."""

    async def mock_retriever(query: SearchQuery):
        yield {"id": "doc1", "content": "Result without metadata", "score": 0.9}

    node = RetrieverNode(retriever_fn=mock_retriever, name="retriever")

    query = SearchQuery(query="test")
    items = []

    async for item in node.astream(query):
        items.append(item)

    # Check the retrieval item has empty metadata
    retrieval_item = items[1]
    assert isinstance(retrieval_item, RetrievalItem)
    assert retrieval_item.metadata == {}


@pytest.mark.asyncio
async def test_retriever_node_missing_id():
    """Test RetrieverNode generates UUID when id is missing."""

    async def mock_retriever(query: SearchQuery):
        yield {"content": "Result without ID", "score": 0.9}

    node = RetrieverNode(retriever_fn=mock_retriever, name="retriever")

    query = SearchQuery(query="test")
    items = []

    async for item in node.astream(query):
        items.append(item)

    # Check the retrieval item has a generated UUID
    retrieval_item = items[1]
    assert isinstance(retrieval_item, RetrievalItem)
    assert retrieval_item.item_id  # Should have some ID


@pytest.mark.asyncio
async def test_retriever_node_empty_results():
    """Test RetrieverNode handles empty results."""

    async def mock_retriever(query: SearchQuery):
        # Yield nothing
        if False:
            yield

    node = RetrieverNode(retriever_fn=mock_retriever, name="retriever")

    query = SearchQuery(query="no results")
    items = []

    async for item in node.astream(query):
        items.append(item)

    # Should have Start and End, but no retrieval items
    assert len(items) == 2
    assert isinstance(items[0], StreamStart)
    assert isinstance(items[1], StreamEnd)
    assert items[1].result_preview == {"results": []}


@pytest.mark.asyncio
async def test_retriever_node_with_input_dependency():
    """Test RetrieverNode with input from another node."""

    async def mock_retriever(query: SearchQuery):
        yield {"id": "doc1", "content": f"Result for {query.query}"}

    # Simulate input from another node
    class MockOutputNode:
        output = SearchQuery(query="dependency query")

    node = RetrieverNode(
        retriever_fn=mock_retriever,
        input=MockOutputNode.output,
        name="retriever",
    )

    assert node.input == MockOutputNode.output


@pytest.mark.asyncio
async def test_retriever_node_without_model_dump():
    """Test RetrieverNode handles input without model_dump."""

    async def mock_retriever(query):
        yield {"id": "doc1", "content": "Result"}

    node = RetrieverNode(retriever_fn=mock_retriever, name="retriever")

    # Use a plain dict as input
    items = []
    async for item in node.astream({"query": "test"}):
        items.append(item)

    # Check StreamStart with None input_preview
    assert isinstance(items[0], StreamStart)
    assert items[0].input_preview is None
