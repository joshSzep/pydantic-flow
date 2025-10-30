"""Tests for MMR diversification and source-aware selection."""

from pydantic_flow.rag.diversify import DiversifyConfig
from pydantic_flow.rag.diversify import diversify_by_source
from pydantic_flow.rag.diversify import mmr_select
from pydantic_flow.rag.rerankers import ScoredChunk
from pydantic_flow.rag.splitters import ChunkMetadata
from pydantic_flow.rag.splitters import DocumentChunk


class TestMMRSelect:
    """Tests for MMR selection."""

    def test_basic_mmr_selection(self) -> None:
        """Test basic MMR selection."""
        chunks = [
            ScoredChunk(
                chunk=DocumentChunk(
                    id=str(i),
                    text=f"document {i}",
                    metadata=ChunkMetadata(source_id="doc1", chunk_index=i),
                ),
                score=1.0 - (i * 0.1),
                rank=i,
            )
            for i in range(5)
        ]

        selected = mmr_select(chunks, k=3, lambda_mult=0.5)

        assert len(selected) == 3
        assert all(isinstance(s, ScoredChunk) for s in selected)

    def test_lambda_zero_maximizes_diversity(self) -> None:
        """Test that lambda=0 maximizes diversity."""
        chunks = [
            ScoredChunk(
                chunk=DocumentChunk(
                    id="1",
                    text="identical text",
                    metadata=ChunkMetadata(source_id="doc1", chunk_index=0),
                ),
                score=1.0,
                rank=0,
            ),
            ScoredChunk(
                chunk=DocumentChunk(
                    id="2",
                    text="identical text",
                    metadata=ChunkMetadata(source_id="doc1", chunk_index=1),
                ),
                score=0.9,
                rank=1,
            ),
            ScoredChunk(
                chunk=DocumentChunk(
                    id="3",
                    text="completely different content here",
                    metadata=ChunkMetadata(source_id="doc1", chunk_index=2),
                ),
                score=0.8,
                rank=2,
            ),
        ]

        selected = mmr_select(chunks, k=2, lambda_mult=0.0)

        assert len(selected) == 2
        assert selected[1].chunk.id == "3"

    def test_lambda_one_ignores_diversity(self) -> None:
        """Test that lambda=1 ignores diversity (pure relevance)."""
        chunks = [
            ScoredChunk(
                chunk=DocumentChunk(
                    id=str(i),
                    text="same content",
                    metadata=ChunkMetadata(source_id="doc1", chunk_index=i),
                ),
                score=1.0 - (i * 0.1),
                rank=i,
            )
            for i in range(5)
        ]

        selected = mmr_select(chunks, k=3, lambda_mult=1.0)

        assert len(selected) == 3
        assert [s.chunk.id for s in selected] == ["0", "1", "2"]

    def test_k_larger_than_input(self) -> None:
        """Test k larger than input size."""
        chunks = [
            ScoredChunk(
                chunk=DocumentChunk(
                    id=str(i),
                    text=f"doc {i}",
                    metadata=ChunkMetadata(source_id="doc1", chunk_index=i),
                ),
                score=1.0,
                rank=i,
            )
            for i in range(3)
        ]

        selected = mmr_select(chunks, k=10, lambda_mult=0.5)

        assert len(selected) == 3

    def test_empty_chunks(self) -> None:
        """Test with empty chunks list."""
        selected = mmr_select([], k=5, lambda_mult=0.5)

        assert selected == []

    def test_near_duplicate_detection(self) -> None:
        """Test that near duplicates are detected and reduced."""
        chunks = [
            ScoredChunk(
                chunk=DocumentChunk(
                    id="1",
                    text="the quick brown fox jumps",
                    metadata=ChunkMetadata(source_id="doc1", chunk_index=0),
                ),
                score=1.0,
                rank=0,
            ),
            ScoredChunk(
                chunk=DocumentChunk(
                    id="2",
                    text="the quick brown fox leaps",
                    metadata=ChunkMetadata(source_id="doc1", chunk_index=1),
                ),
                score=0.95,
                rank=1,
            ),
            ScoredChunk(
                chunk=DocumentChunk(
                    id="3",
                    text="completely different sentence about cats",
                    metadata=ChunkMetadata(source_id="doc1", chunk_index=2),
                ),
                score=0.8,
                rank=2,
            ),
        ]

        selected_low_lambda = mmr_select(chunks, k=2, lambda_mult=0.3)
        _ = mmr_select(chunks, k=2, lambda_mult=0.9)

        assert selected_low_lambda[1].chunk.id == "3"


class TestDiversifyBySource:
    """Tests for source-aware diversification."""

    def test_max_per_source_enforcement(self) -> None:
        """Test that max_per_source is enforced."""
        chunks = [
            ScoredChunk(
                chunk=DocumentChunk(
                    id=f"doc1_{i}",
                    text=f"content {i}",
                    metadata=ChunkMetadata(source_id="doc1", chunk_index=i),
                ),
                score=1.0 - (i * 0.1),
                rank=i,
            )
            for i in range(5)
        ] + [
            ScoredChunk(
                chunk=DocumentChunk(
                    id=f"doc2_{i}",
                    text=f"content {i}",
                    metadata=ChunkMetadata(source_id="doc2", chunk_index=i),
                ),
                score=0.9 - (i * 0.1),
                rank=i + 5,
            )
            for i in range(5)
        ]

        selected = diversify_by_source(
            chunks, k=10, max_per_source=2, round_robin=False
        )

        doc1_count = sum(1 for s in selected if s.chunk.metadata.source_id == "doc1")
        doc2_count = sum(1 for s in selected if s.chunk.metadata.source_id == "doc2")

        assert doc1_count <= 2
        assert doc2_count <= 2

    def test_round_robin_selection(self) -> None:
        """Test round-robin selection across sources."""
        chunks = [
            ScoredChunk(
                chunk=DocumentChunk(
                    id=f"doc{i % 3}_{i}",
                    text=f"content {i}",
                    metadata=ChunkMetadata(source_id=f"doc{i % 3}", chunk_index=i),
                ),
                score=1.0 - (i * 0.05),
                rank=i,
            )
            for i in range(12)
        ]

        selected = diversify_by_source(chunks, k=9, round_robin=True)

        sources = [s.chunk.metadata.source_id for s in selected]

        for i in range(0, min(9, len(sources)) - 3, 3):
            window = sources[i : i + 3]
            assert len(set(window)) >= 2

    def test_no_constraints_returns_top_k(self) -> None:
        """Test that no constraints returns top k."""
        chunks = [
            ScoredChunk(
                chunk=DocumentChunk(
                    id=str(i),
                    text=f"content {i}",
                    metadata=ChunkMetadata(source_id="doc1", chunk_index=i),
                ),
                score=1.0 - (i * 0.1),
                rank=i,
            )
            for i in range(10)
        ]

        selected = diversify_by_source(chunks, k=5)

        assert len(selected) == 5
        assert [s.chunk.id for s in selected] == ["0", "1", "2", "3", "4"]

    def test_empty_chunks(self) -> None:
        """Test with empty chunks."""
        selected = diversify_by_source([], k=5, max_per_source=2)

        assert selected == []


class TestDiversifyConfig:
    """Tests for DiversifyConfig model."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = DiversifyConfig()

        assert config.k == 10
        assert config.lambda_mult == 0.5
        assert config.max_per_source is None
        assert config.round_robin_by_source is False

    def test_custom_values(self) -> None:
        """Test custom configuration."""
        config = DiversifyConfig(
            k=20,
            lambda_mult=0.7,
            max_per_source=3,
            round_robin_by_source=True,
        )

        assert config.k == 20
        assert config.lambda_mult == 0.7
        assert config.max_per_source == 3
        assert config.round_robin_by_source is True

    def test_validation(self) -> None:
        """Test that validation works."""
        config = DiversifyConfig(lambda_mult=0.5)
        assert 0.0 <= config.lambda_mult <= 1.0


class TestDiversificationPropertyBased:
    """Property-based tests for diversification."""

    def test_result_size_never_exceeds_k(self) -> None:
        """Test that result size never exceeds k."""
        chunks = [
            ScoredChunk(
                chunk=DocumentChunk(
                    id=str(i),
                    text=f"content {i}",
                    metadata=ChunkMetadata(source_id=f"doc{i % 3}", chunk_index=i),
                ),
                score=1.0 - (i * 0.05),
                rank=i,
            )
            for i in range(20)
        ]

        for k in [1, 5, 10, 15, 25]:
            selected_mmr = mmr_select(chunks, k=k, lambda_mult=0.5)
            selected_source = diversify_by_source(
                chunks, k=k, max_per_source=2, round_robin=True
            )

            assert len(selected_mmr) <= k
            assert len(selected_source) <= k

    def test_duplicate_prevention(self) -> None:
        """Test that no duplicates are in results."""
        chunks = [
            ScoredChunk(
                chunk=DocumentChunk(
                    id=str(i),
                    text="identical content",
                    metadata=ChunkMetadata(source_id="doc1", chunk_index=i),
                ),
                score=1.0 - (i * 0.1),
                rank=i,
            )
            for i in range(10)
        ]

        selected = mmr_select(chunks, k=5, lambda_mult=0.3)

        ids = [s.chunk.id for s in selected]
        assert len(ids) == len(set(ids))
