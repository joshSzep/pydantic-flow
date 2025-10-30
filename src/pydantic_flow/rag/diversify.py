"""MMR diversification and source-aware selection."""

from collections import defaultdict
from collections.abc import Callable

from opentelemetry import trace
from pydantic import BaseModel

from pydantic_flow.rag.rerankers.base import ScoredChunk
from pydantic_flow.rag.rerankers.lexical import LexicalReranker

tracer = trace.get_tracer(__name__)


class DiversifyConfig(BaseModel):
    """Configuration for diversification.

    Attributes:
        k: Number of final results to return.
        lambda_mult: Balance between relevance and diversity [0, 1].
        max_per_source: Maximum chunks per source_id.
        round_robin_by_source: Use round-robin selection by source.

    """

    k: int = 10
    lambda_mult: float = 0.5
    max_per_source: int | None = None
    round_robin_by_source: bool = False


def mmr_select(
    scored_chunks: list[ScoredChunk],
    k: int,
    lambda_mult: float = 0.5,
    similarity: Callable[[ScoredChunk, ScoredChunk], float] | None = None,
) -> list[ScoredChunk]:
    """Apply Maximal Marginal Relevance selection.

    Selects k chunks that balance relevance and diversity.

    Args:
        scored_chunks: List of scored chunks sorted by relevance.
        k: Number of chunks to select.
        lambda_mult: Balance between relevance (1.0) and diversity (0.0).
        similarity: Optional similarity function (uses lexical by default).

    Returns:
        List of k diverse chunks.

    """
    with tracer.start_as_current_span("rag.diversify.mmr") as span:
        span.set_attribute("lambda_mult", lambda_mult)
        span.set_attribute("input_size", len(scored_chunks))

        if len(scored_chunks) <= k:
            span.set_attribute("final_k", len(scored_chunks))
            return scored_chunks

        sim_func = similarity or _default_similarity
        selected: list[ScoredChunk] = []
        remaining = scored_chunks.copy()

        if remaining:
            selected.append(remaining.pop(0))

        while len(selected) < k and remaining:
            best_idx = -1
            best_score = float("-inf")

            for idx, candidate in enumerate(remaining):
                relevance = candidate.score

                if selected:
                    max_similarity = max(sim_func(candidate, sel) for sel in selected)
                else:
                    max_similarity = 0.0

                mmr_score = lambda_mult * relevance - (1 - lambda_mult) * max_similarity

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx >= 0:
                selected.append(remaining.pop(best_idx))

        span.set_attribute("final_k", len(selected))
        return selected


def diversify_by_source(
    scored_chunks: list[ScoredChunk],
    k: int,
    max_per_source: int | None = None,
    round_robin: bool = False,
) -> list[ScoredChunk]:
    """Apply source-aware diversification.

    Args:
        scored_chunks: List of scored chunks sorted by relevance.
        k: Number of chunks to select.
        max_per_source: Maximum chunks per source_id.
        round_robin: Use round-robin selection across sources.

    Returns:
        List of k diverse chunks respecting source constraints.

    """
    with tracer.start_as_current_span("rag.diversify.source") as span:
        span.set_attribute("max_per_source", max_per_source or 0)
        span.set_attribute("round_robin", round_robin)

        if not round_robin and not max_per_source:
            return scored_chunks[:k]

        if round_robin:
            result = _round_robin_by_source(scored_chunks, k)
        else:
            result = _enforce_max_per_source(scored_chunks, k, max_per_source or k)

        span.set_attribute("final_k", len(result))
        return result


def _default_similarity(chunk1: ScoredChunk, chunk2: ScoredChunk) -> float:
    """Compute similarity between two chunks using lexical vectors."""
    reranker = LexicalReranker()
    vec1 = reranker.get_vector(chunk1.chunk.text)
    vec2 = reranker.get_vector(chunk2.chunk.text)
    return reranker._cosine_similarity(vec1, vec2)


def _round_robin_by_source(
    scored_chunks: list[ScoredChunk], k: int
) -> list[ScoredChunk]:
    """Select chunks using round-robin across sources."""
    by_source: dict[str, list[ScoredChunk]] = defaultdict(list)

    for chunk in scored_chunks:
        source_id = chunk.chunk.metadata.source_id
        by_source[source_id].append(chunk)

    result: list[ScoredChunk] = []
    sources = list(by_source.keys())
    source_idx = 0

    while len(result) < k and by_source:
        source = sources[source_idx % len(sources)]

        if by_source[source]:
            result.append(by_source[source].pop(0))
            if not by_source[source]:
                sources.remove(source)
        else:
            sources.remove(source)

        if sources:
            source_idx = (source_idx + 1) % len(sources)
        else:
            break

    return result


def _enforce_max_per_source(
    scored_chunks: list[ScoredChunk], k: int, max_per_source: int
) -> list[ScoredChunk]:
    """Enforce maximum chunks per source."""
    source_counts: dict[str, int] = defaultdict(int)
    result: list[ScoredChunk] = []

    for chunk in scored_chunks:
        if len(result) >= k:
            break

        source_id = chunk.chunk.metadata.source_id

        if source_counts[source_id] < max_per_source:
            result.append(chunk)
            source_counts[source_id] += 1

    return result
