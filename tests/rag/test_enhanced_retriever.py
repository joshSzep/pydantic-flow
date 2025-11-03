"""Tests for EnhancedRetrieverNode."""

from pydantic_flow.rag import Document
from pydantic_flow.rag import Metadata
from pydantic_flow.rag.diversify import DiversifyConfig
from pydantic_flow.rag.nodes.enhanced_retriever import EnhancedQueryInput
from pydantic_flow.rag.nodes.enhanced_retriever import EnhancedRetrievalResult
from pydantic_flow.rag.nodes.enhanced_retriever import EnhancedRetrieverNode
from pydantic_flow.rag.rerankers import RerankConfig
from pydantic_flow.rag.retrievers.base import Retriever
from pydantic_flow.rag.splitters import SplitConfig
from pydantic_flow.streaming.core_events import StreamEnd
from pydantic_flow.streaming.core_events import StreamStart
from pydantic_flow.streaming.retrieval_events import RetrievalItem


class MockRetriever(Retriever):
    """Mock retriever for testing."""

    def __init__(self, docs: list[Document]) -> None:
        """Initialize mock retriever."""
        self.docs = docs

    async def retrieve(self, query: str, k: int = 5) -> list[Document]:
        """Return mock documents."""
        return self.docs[:k]


class TestEnhancedRetrieverNode:
    """Test suite for EnhancedRetrieverNode."""

    async def test_basic_retrieval(self) -> None:
        """Test basic retrieval without enhancement."""
        docs = [
            Document(id="doc1", content="Test document 1", metadata=Metadata()),
            Document(id="doc2", content="Test document 2", metadata=Metadata()),
        ]

        retriever = MockRetriever(docs)
        node = EnhancedRetrieverNode(retriever=retriever)

        query = EnhancedQueryInput(query="test", k=2, top_n=5)

        items = []
        async for item in node.astream(query):
            items.append(item)

        assert any(isinstance(i, StreamStart) for i in items)
        assert any(isinstance(i, StreamEnd) for i in items)
        retrieval_items = [i for i in items if isinstance(i, RetrievalItem)]
        assert len(retrieval_items) == 2

    async def test_with_token_splitting(self) -> None:
        """Test with token-based splitting."""
        long_text = "sentence. " * 50
        docs = [
            Document(id="doc1", content=long_text, metadata=Metadata()),
        ]

        retriever = MockRetriever(docs)
        split_config = SplitConfig(
            splitter_type="token",
            max_tokens=20,
            overlap=5,
            min_chunk_chars=10,
        )
        node = EnhancedRetrieverNode(retriever=retriever, split_config=split_config)

        query = EnhancedQueryInput(query="test", k=3, top_n=5)

        items = []
        async for item in node.astream(query):
            items.append(item)

        stream_end = next(i for i in items if isinstance(i, StreamEnd))
        result = EnhancedRetrievalResult.model_validate(stream_end.result)

        assert result.chunks
        assert any(stage["name"] == "split" for stage in result.stats["stages"])

    async def test_with_sentence_splitting(self) -> None:
        """Test with sentence-based splitting."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        docs = [
            Document(id="doc1", content=text, metadata=Metadata()),
        ]

        retriever = MockRetriever(docs)
        split_config = SplitConfig(
            splitter_type="sentence",
            max_chars=50,
            overlap=10,
            min_chunk_chars=5,
        )
        node = EnhancedRetrieverNode(retriever=retriever, split_config=split_config)

        query = EnhancedQueryInput(query="test", k=2, top_n=5)

        items = []
        async for item in node.astream(query):
            items.append(item)

        stream_end = next(i for i in items if isinstance(i, StreamEnd))
        result = EnhancedRetrievalResult.model_validate(stream_end.result)

        assert result.chunks
        assert any(stage["name"] == "split" for stage in result.stats["stages"])

    async def test_with_markdown_splitting(self) -> None:
        """Test with markdown heading splitting."""
        text = """# Title
Content under title.

## Section
Content under section."""
        docs = [
            Document(id="doc1", content=text, metadata=Metadata()),
        ]

        retriever = MockRetriever(docs)
        split_config = SplitConfig(
            splitter_type="markdown",
            max_chars=200,
            overlap=0,
            min_chunk_chars=5,
        )
        node = EnhancedRetrieverNode(retriever=retriever, split_config=split_config)

        query = EnhancedQueryInput(query="test", k=2, top_n=5)

        items = []
        async for item in node.astream(query):
            items.append(item)

        stream_end = next(i for i in items if isinstance(i, StreamEnd))
        result = EnhancedRetrievalResult.model_validate(stream_end.result)

        assert result.chunks
        assert any(stage["name"] == "split" for stage in result.stats["stages"])

    async def test_with_lexical_reranking(self) -> None:
        """Test with lexical reranking."""
        docs = [
            Document(
                id="doc1", content="Python programming language", metadata=Metadata()
            ),
            Document(
                id="doc2", content="Java programming language", metadata=Metadata()
            ),
            Document(id="doc3", content="Python is popular", metadata=Metadata()),
        ]

        retriever = MockRetriever(docs)
        rerank_config = RerankConfig(kind="lexical")
        node = EnhancedRetrieverNode(retriever=retriever, rerank_config=rerank_config)

        query = EnhancedQueryInput(query="python programming", k=2, top_n=5)

        items = []
        async for item in node.astream(query):
            items.append(item)

        stream_end = next(i for i in items if isinstance(i, StreamEnd))
        result = EnhancedRetrievalResult.model_validate(stream_end.result)

        assert result.chunks
        assert any(stage["name"] == "rerank" for stage in result.stats["stages"])

    async def test_with_mmr_diversification(self) -> None:
        """Test with MMR diversification."""
        docs = [
            Document(id="doc1", content="Python is a language", metadata=Metadata()),
            Document(
                id="doc2", content="Python is a popular language", metadata=Metadata()
            ),
            Document(id="doc3", content="Java is also a language", metadata=Metadata()),
        ]

        retriever = MockRetriever(docs)
        diversify_config = DiversifyConfig(lambda_mult=0.5)
        node = EnhancedRetrieverNode(
            retriever=retriever,
            diversify_config=diversify_config,
        )

        query = EnhancedQueryInput(query="programming", k=2, top_n=5)

        items = []
        async for item in node.astream(query):
            items.append(item)

        stream_end = next(i for i in items if isinstance(i, StreamEnd))
        result = EnhancedRetrievalResult.model_validate(stream_end.result)

        assert result.chunks
        assert any(stage["name"] == "diversify" for stage in result.stats["stages"])

    async def test_with_source_diversification(self) -> None:
        """Test with source-based diversification."""
        docs = [
            Document(id="doc1", content="Content A", metadata=Metadata(source="src1")),
            Document(id="doc2", content="Content B", metadata=Metadata(source="src1")),
            Document(id="doc3", content="Content C", metadata=Metadata(source="src2")),
        ]

        retriever = MockRetriever(docs)
        diversify_config = DiversifyConfig(
            max_per_source=1,
            round_robin_by_source=True,
        )
        node = EnhancedRetrieverNode(
            retriever=retriever,
            diversify_config=diversify_config,
        )

        query = EnhancedQueryInput(query="content", k=3, top_n=5)

        items = []
        async for item in node.astream(query):
            items.append(item)

        stream_end = next(i for i in items if isinstance(i, StreamEnd))
        result = EnhancedRetrievalResult.model_validate(stream_end.result)

        assert result.chunks
        assert any(stage["name"] == "diversify" for stage in result.stats["stages"])

    async def test_full_pipeline(self) -> None:
        """Test complete pipeline with all features."""
        text = "First sentence about Python. Second sentence about Java. " * 10
        docs = [
            Document(id="doc1", content=text, metadata=Metadata()),
        ]

        retriever = MockRetriever(docs)
        split_config = SplitConfig(
            splitter_type="sentence",
            max_chars=100,
            overlap=10,
        )
        rerank_config = RerankConfig(kind="lexical")
        diversify_config = DiversifyConfig(lambda_mult=0.6)

        node = EnhancedRetrieverNode(
            retriever=retriever,
            split_config=split_config,
            rerank_config=rerank_config,
            diversify_config=diversify_config,
        )

        query = EnhancedQueryInput(query="python", k=3, top_n=5)

        items = []
        async for item in node.astream(query):
            items.append(item)

        stream_end = next(i for i in items if isinstance(i, StreamEnd))
        result = EnhancedRetrievalResult.model_validate(stream_end.result)

        assert result.chunks
        assert len(result.stats["stages"]) >= 3
        stage_names = {stage["name"] for stage in result.stats["stages"]}
        assert "retrieve" in stage_names
        assert "split" in stage_names
        assert "rerank" in stage_names
        assert "diversify" in stage_names

    async def test_unknown_splitter_defaults_to_token(self) -> None:
        """Test that unknown splitter type defaults to token splitter."""
        docs = [
            Document(id="doc1", content="Test content " * 50, metadata=Metadata()),
        ]

        retriever = MockRetriever(docs)
        split_config = SplitConfig(
            splitter_type="unknown_type",
            max_tokens=10,
            min_chunk_chars=5,
        )
        node = EnhancedRetrieverNode(retriever=retriever, split_config=split_config)

        query = EnhancedQueryInput(query="test", k=2, top_n=5)

        items = []
        async for item in node.astream(query):
            items.append(item)

        stream_end = next(i for i in items if isinstance(i, StreamEnd))
        result = EnhancedRetrievalResult.model_validate(stream_end.result)

        assert result.chunks
        assert any(stage["name"] == "split" for stage in result.stats["stages"])

    async def test_result_contains_metadata(self) -> None:
        """Test that result chunks contain full metadata."""
        docs = [
            Document(id="doc1", content="Test content", metadata=Metadata()),
        ]

        retriever = MockRetriever(docs)
        node = EnhancedRetrieverNode(retriever=retriever)

        query = EnhancedQueryInput(query="test", k=1, top_n=5)

        items = []
        async for item in node.astream(query):
            items.append(item)

        stream_end = next(i for i in items if isinstance(i, StreamEnd))
        result = EnhancedRetrievalResult.model_validate(stream_end.result)

        assert result.chunks[0]["id"]
        assert result.chunks[0]["text"]
        assert result.chunks[0]["score"]
        assert result.chunks[0]["metadata"]
        assert result.chunks[0]["metadata"]["source_id"]
