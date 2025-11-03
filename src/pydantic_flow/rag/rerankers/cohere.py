"""Cohere reranker adapter with lazy import."""

from typing import Any

from opentelemetry import trace

from pydantic_flow.rag.rerankers.base import ScoredChunk
from pydantic_flow.rag.splitters.base import DocumentChunk

tracer = trace.get_tracer(__name__)


class CohereReranker:
    """Cohere Rerank API adapter.

    Lazily imports cohere library. If not available, raises clear error
    with installation instructions.

    Attributes:
        client: Cohere client instance.
        model: Rerank model name.
        top_n: Maximum number of results to return.

    """

    def __init__(
        self,
        api_key: str | None = None,
        client: Any = None,
        model: str = "rerank-english-v3.0",
        top_n: int | None = None,
    ) -> None:
        """Initialize Cohere reranker.

        Args:
            api_key: Cohere API key (required if client not provided).
            client: Pre-configured Cohere client instance.
            model: Rerank model name.
            top_n: Maximum number of results to return.

        Raises:
            ImportError: If cohere library is not installed.

        """
        if client is None:
            if api_key is None:
                msg = "api_key is required when client is not provided"
                raise ValueError(msg)

            try:
                import cohere  # noqa: PLC0415
            except ImportError as e:
                msg = (
                    "cohere library is required for CohereReranker. "
                    "Install it with: pip install cohere"
                )
                raise ImportError(msg) from e

            self.client = cohere.Client(api_key)
        else:
            self.client = client

        self.model = model
        self.top_n = top_n

    def score(
        self,
        query: str,
        chunks: list[DocumentChunk],
    ) -> list[ScoredChunk]:
        """Score chunks using Cohere Rerank API.

        Args:
            query: Query string.
            chunks: Document chunks to score.

        Returns:
            List of scored chunks sorted by relevance (highest first).

        """
        with tracer.start_as_current_span("rag.rerank.cohere") as span:
            if not chunks:
                return []

            documents = [chunk.text for chunk in chunks]

            response = self.client.rerank(
                model=self.model,
                query=query,
                documents=documents,
                top_n=self.top_n or len(documents),
            )

            span.set_attribute("model", self.model)
            span.set_attribute("top_n", len(response.results))

            scored: list[ScoredChunk] = []
            for result in response.results:
                original_chunk = chunks[result.index]
                scored.append(
                    ScoredChunk(
                        chunk=original_chunk,
                        score=result.relevance_score,
                        rank=result.index,
                    )
                )

            return scored
