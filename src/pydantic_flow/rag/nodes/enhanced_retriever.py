"""Enhanced retriever node with splitting, reranking, and diversification."""

from collections.abc import AsyncIterator
import time
import uuid

from pydantic import BaseModel

from pydantic_flow.nodes.base import NodeOutput
from pydantic_flow.nodes.base import NodeWithInput
from pydantic_flow.rag.diversify import DiversifyConfig
from pydantic_flow.rag.diversify import diversify_by_source
from pydantic_flow.rag.diversify import mmr_select
from pydantic_flow.rag.docs import Document
from pydantic_flow.rag.rerankers import CohereReranker
from pydantic_flow.rag.rerankers import LexicalReranker
from pydantic_flow.rag.rerankers import RerankConfig
from pydantic_flow.rag.rerankers import Reranker
from pydantic_flow.rag.rerankers import ScoredChunk
from pydantic_flow.rag.retrievers.base import Retriever
from pydantic_flow.rag.splitters import DocumentChunk
from pydantic_flow.rag.splitters import MarkdownHeadingSplitter
from pydantic_flow.rag.splitters import SentenceSplitter
from pydantic_flow.rag.splitters import SplitConfig
from pydantic_flow.rag.splitters import Splitter
from pydantic_flow.rag.splitters import TokenSplitter
from pydantic_flow.rag.splitters.base import ChunkMetadata
from pydantic_flow.streaming.base import ProgressItem
from pydantic_flow.streaming.core_events import StreamEnd
from pydantic_flow.streaming.core_events import StreamStart
from pydantic_flow.streaming.retrieval_events import RetrievalItem


class EnhancedQueryInput(BaseModel):
    """Query input for enhanced retriever node.

    Attributes:
        query: Query string.
        k: Number of final results to return.
        top_n: Number of raw documents to retrieve before processing.

    """

    query: str
    k: int = 5
    top_n: int = 20


class EnhancedRetrievalResult(BaseModel):
    """Enhanced retrieval result output.

    Attributes:
        chunks: List of retrieved and processed chunks.
        query: Original query string.
        stats: Processing statistics.

    """

    chunks: list[dict]
    query: str
    stats: dict


class EnhancedRetrieverNode(NodeWithInput[EnhancedQueryInput, EnhancedRetrievalResult]):
    """Enhanced retriever with splitting, reranking, and diversification.

    This node extends the basic retriever with:
    - Optional document splitting strategies
    - Optional reranking (lexical or provider-based)
    - Optional MMR diversification
    - Source-aware selection

    Attributes:
        retriever: Retriever instance for semantic search.
        split_config: Optional splitting configuration.
        rerank_config: Optional reranking configuration.
        diversify_config: Optional diversification configuration.

    """

    def __init__(  # noqa: PLR0913
        self,
        retriever: Retriever,
        split_config: SplitConfig | None = None,
        rerank_config: RerankConfig | None = None,
        diversify_config: DiversifyConfig | None = None,
        *,
        input: NodeOutput[EnhancedQueryInput] | None = None,
        name: str | None = None,
        run_id: str | None = None,
    ) -> None:
        """Initialize EnhancedRetrieverNode.

        Args:
            retriever: Retriever instance.
            split_config: Optional splitting configuration.
            rerank_config: Optional reranking configuration.
            diversify_config: Optional diversification configuration.
            input: Optional input from another node's output.
            name: Optional unique identifier for this node.
            run_id: Optional run identifier for tracking execution.

        """
        super().__init__(input, name, run_id)
        self.retriever = retriever
        self.split_config = split_config
        self.rerank_config = rerank_config
        self.diversify_config = diversify_config

        self._splitter = self._create_splitter()
        self._reranker = self._create_reranker()

    def _create_splitter(self) -> Splitter | None:
        """Create splitter based on config."""
        if not self.split_config:
            return None

        splitter_type = self.split_config.splitter_type

        if splitter_type == "token":
            return TokenSplitter()
        elif splitter_type == "sentence":
            return SentenceSplitter()
        elif splitter_type == "markdown":
            return MarkdownHeadingSplitter()
        else:
            return TokenSplitter()

    def _create_reranker(self) -> Reranker | None:
        """Create reranker based on config."""
        if not self.rerank_config:
            return None

        if self.rerank_config.kind == "cohere":
            return CohereReranker(
                api_key=self.rerank_config.api_key,
                model=self.rerank_config.model or "rerank-english-v3.0",
                top_n=self.rerank_config.top_n,
            )
        else:
            return LexicalReranker()

    async def astream(
        self, input_data: EnhancedQueryInput
    ) -> AsyncIterator[ProgressItem]:
        """Stream enhanced retrieval results.

        Pipeline:
        1. Retrieve top_n raw documents
        2. Optional: Split documents into chunks
        3. Optional: Rerank chunks
        4. Optional: Diversify using MMR
        5. Return top k chunks

        Yields:
            StreamStart, RetrievalItem for each chunk, and StreamEnd.

        """
        actual_run_id = self.run_id or str(uuid.uuid4())
        stats = {"stages": []}

        yield StreamStart(
            run_id=actual_run_id,
            node_id=self.name,
            input_preview=input_data.model_dump(),
        )

        start_time = time.time()
        documents = await self.retriever.retrieve(
            query=input_data.query,
            k=input_data.top_n,
        )
        stats["stages"].append({
            "name": "retrieve",
            "count": len(documents),
            "duration_ms": int((time.time() - start_time) * 1000),
        })

        chunks = self._documents_to_chunks(documents)

        if self._splitter and self.split_config:
            start_time = time.time()
            chunks = self._apply_splitting(documents)
            stats["stages"].append({
                "name": "split",
                "count": len(chunks),
                "duration_ms": int((time.time() - start_time) * 1000),
            })

        if self._reranker:
            start_time = time.time()
            scored_chunks = self._reranker.score(input_data.query, chunks)
            stats["stages"].append({
                "name": "rerank",
                "count": len(scored_chunks),
                "duration_ms": int((time.time() - start_time) * 1000),
            })
        else:
            scored_chunks = [
                ScoredChunk(chunk=c, score=1.0, rank=i) for i, c in enumerate(chunks)
            ]

        if self.diversify_config:
            start_time = time.time()

            if self.diversify_config.lambda_mult < 1.0:
                scored_chunks = mmr_select(
                    scored_chunks,
                    k=input_data.k,
                    lambda_mult=self.diversify_config.lambda_mult,
                )

            if (
                self.diversify_config.max_per_source
                or self.diversify_config.round_robin_by_source
            ):
                scored_chunks = diversify_by_source(
                    scored_chunks,
                    k=input_data.k,
                    max_per_source=self.diversify_config.max_per_source,
                    round_robin=self.diversify_config.round_robin_by_source,
                )

            stats["stages"].append({
                "name": "diversify",
                "count": len(scored_chunks),
                "duration_ms": int((time.time() - start_time) * 1000),
            })
        else:
            scored_chunks = scored_chunks[: input_data.k]

        chunk_dicts = []
        for scored in scored_chunks:
            chunk_dict = {
                "id": scored.chunk.id,
                "text": scored.chunk.text,
                "score": scored.score,
                "metadata": scored.chunk.metadata.model_dump(),
            }
            chunk_dicts.append(chunk_dict)

            yield RetrievalItem(
                item_id=scored.chunk.id,
                content=scored.chunk.text,
                score=scored.score,
                metadata=scored.chunk.metadata.model_dump(),
                run_id=actual_run_id,
                node_id=self.name,
            )

        result = EnhancedRetrievalResult(
            chunks=chunk_dicts,
            query=input_data.query,
            stats=stats,
        )

        # Yield the actual result object
        from pydantic_flow.streaming.tool_events import ToolResult  # noqa: PLC0415

        yield ToolResult(
            run_id=actual_run_id,
            node_id=self.name,
            tool_name="enhanced_retrieve",
            result=result,
        )

        yield StreamEnd(
            run_id=actual_run_id,
            node_id=self.name,
            result=result,
        )

    def _documents_to_chunks(self, documents: list[Document]) -> list[DocumentChunk]:
        """Convert documents to chunks without splitting."""
        chunks = []
        for doc in documents:
            chunk = DocumentChunk(
                id=doc.id,
                text=doc.content,
                metadata=ChunkMetadata(
                    source_id=doc.id,
                    chunk_index=0,
                ),
            )
            chunks.append(chunk)
        return chunks

    def _apply_splitting(self, documents: list[Document]) -> list[DocumentChunk]:
        """Apply splitting to documents."""
        if not self._splitter or not self.split_config:
            return self._documents_to_chunks(documents)

        all_chunks = []
        for doc in documents:
            chunks = self._splitter.split(
                text=doc.content,
                source_id=doc.id,
                config=self.split_config,
            )
            all_chunks.extend(chunks)

        return all_chunks
