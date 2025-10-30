"""Lexical baseline reranker using hashed term frequencies."""

from collections import Counter
import math

from opentelemetry import trace

from pydantic_flow.rag.rerankers.base import ScoredChunk
from pydantic_flow.rag.splitters.base import DocumentChunk

tracer = trace.get_tracer(__name__)


class LexicalReranker:
    """Baseline lexical reranker using hashed TF vectors.

    Uses normalized term frequencies and cosine similarity for scoring.
    Completely dependency-free and deterministic.

    """

    def __init__(self, normalize: bool = True) -> None:
        """Initialize lexical reranker.

        Args:
            normalize: Whether to normalize term frequencies.

        """
        self.normalize = normalize

    def score(
        self,
        query: str,
        chunks: list[DocumentChunk],
    ) -> list[ScoredChunk]:
        """Score chunks using lexical similarity to query.

        Args:
            query: Query string.
            chunks: Document chunks to score.

        Returns:
            List of scored chunks sorted by relevance (highest first).

        """
        with tracer.start_as_current_span("rag.rerank.lexical") as span:
            query_vec = self._vectorize(query)
            scored: list[ScoredChunk] = []

            for rank, chunk in enumerate(chunks):
                doc_vec = self._vectorize(chunk.text)
                similarity = self._cosine_similarity(query_vec, doc_vec)

                scored.append(
                    ScoredChunk(
                        chunk=chunk,
                        score=similarity,
                        rank=rank,
                    )
                )

            scored.sort(key=lambda x: x.score, reverse=True)

            span.set_attribute("top_k", len(scored))
            span.set_attribute("max_score", scored[0].score if scored else 0.0)

            return scored

    def _vectorize(self, text: str) -> dict[str, float]:
        """Convert text to normalized term frequency vector.

        Args:
            text: Text to vectorize.

        Returns:
            Dictionary mapping terms to frequencies.

        """
        tokens = self._tokenize(text)
        counter = Counter(tokens)

        if not self.normalize:
            return {term: float(count) for term, count in counter.items()}

        total = sum(counter.values())
        if total == 0:
            return {}

        return {term: count / total for term, count in counter.items()}

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text using whitespace and lowercasing.

        Args:
            text: Text to tokenize.

        Returns:
            List of tokens.

        """
        return text.lower().split()

    def _cosine_similarity(
        self, vec1: dict[str, float], vec2: dict[str, float]
    ) -> float:
        """Calculate cosine similarity between two sparse vectors.

        Args:
            vec1: First vector.
            vec2: Second vector.

        Returns:
            Cosine similarity score [0, 1].

        """
        if not vec1 or not vec2:
            return 0.0

        common_keys = set(vec1.keys()) & set(vec2.keys())
        if not common_keys:
            return 0.0

        dot_product = sum(vec1[k] * vec2[k] for k in common_keys)

        norm1 = math.sqrt(sum(v * v for v in vec1.values()))
        norm2 = math.sqrt(sum(v * v for v in vec2.values()))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def get_vector(self, text: str) -> dict[str, float]:
        """Get term frequency vector for text.

        Exposed for use by MMR diversification.

        Args:
            text: Text to vectorize.

        Returns:
            Dictionary mapping terms to frequencies.

        """
        return self._vectorize(text)
