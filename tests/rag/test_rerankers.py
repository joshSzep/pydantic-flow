"""Tests for document rerankers."""

import os

import pytest

from pydantic_flow.rag.rerankers import CohereReranker
from pydantic_flow.rag.rerankers import LexicalReranker
from pydantic_flow.rag.rerankers import ScoredChunk
from pydantic_flow.rag.splitters import ChunkMetadata
from pydantic_flow.rag.splitters import DocumentChunk


class TestLexicalReranker:
    """Tests for LexicalReranker."""

    def test_exact_match_ranks_highest(self) -> None:
        """Test that exact matches rank highest."""
        reranker = LexicalReranker()

        chunks = [
            DocumentChunk(
                id="1",
                text="completely different content",
                metadata=ChunkMetadata(source_id="doc1", chunk_index=0),
            ),
            DocumentChunk(
                id="2",
                text="the quick brown fox",
                metadata=ChunkMetadata(source_id="doc1", chunk_index=1),
            ),
            DocumentChunk(
                id="3",
                text="another unrelated text",
                metadata=ChunkMetadata(source_id="doc1", chunk_index=2),
            ),
        ]

        scored = reranker.score("quick brown fox", chunks)

        assert scored[0].chunk.id == "2"
        assert scored[0].score > scored[1].score
        assert scored[0].score > scored[2].score

    def test_partial_match_scoring(self) -> None:
        """Test scoring with partial matches."""
        reranker = LexicalReranker()

        chunks = [
            DocumentChunk(
                id="1",
                text="python programming language",
                metadata=ChunkMetadata(source_id="doc1", chunk_index=0),
            ),
            DocumentChunk(
                id="2",
                text="python snake reptile",
                metadata=ChunkMetadata(source_id="doc1", chunk_index=1),
            ),
            DocumentChunk(
                id="3",
                text="java programming language",
                metadata=ChunkMetadata(source_id="doc1", chunk_index=2),
            ),
        ]

        scored = reranker.score("python programming", chunks)

        assert scored[0].chunk.id == "1"
        assert scored[0].score > 0.5

    def test_deterministic_scoring(self) -> None:
        """Test that scoring is deterministic."""
        reranker = LexicalReranker()

        chunks = [
            DocumentChunk(
                id="1",
                text="test content",
                metadata=ChunkMetadata(source_id="doc1", chunk_index=0),
            ),
        ]

        scored1 = reranker.score("test", chunks)
        scored2 = reranker.score("test", chunks)

        assert scored1[0].score == scored2[0].score

    def test_empty_query(self) -> None:
        """Test behavior with empty query."""
        reranker = LexicalReranker()

        chunks = [
            DocumentChunk(
                id="1",
                text="some content",
                metadata=ChunkMetadata(source_id="doc1", chunk_index=0),
            ),
        ]

        scored = reranker.score("", chunks)

        assert len(scored) == 1
        assert scored[0].score == 0.0

    def test_empty_chunks(self) -> None:
        """Test behavior with empty chunks list."""
        reranker = LexicalReranker()

        scored = reranker.score("test query", [])

        assert scored == []

    def test_rank_preservation(self) -> None:
        """Test that original rank is preserved."""
        reranker = LexicalReranker()

        chunks = [
            DocumentChunk(
                id=str(i),
                text=f"content {i}",
                metadata=ChunkMetadata(source_id="doc1", chunk_index=i),
            )
            for i in range(5)
        ]

        scored = reranker.score("content", chunks)

        for scored_chunk in scored:
            assert scored_chunk.rank is not None
            assert 0 <= scored_chunk.rank < 5

    def test_normalization(self) -> None:
        """Test with and without normalization."""
        reranker_norm = LexicalReranker(normalize=True)
        reranker_no_norm = LexicalReranker(normalize=False)

        chunks = [
            DocumentChunk(
                id="1",
                text="test " * 100,
                metadata=ChunkMetadata(source_id="doc1", chunk_index=0),
            ),
            DocumentChunk(
                id="2",
                text="test",
                metadata=ChunkMetadata(source_id="doc1", chunk_index=1),
            ),
        ]

        scored_norm = reranker_norm.score("test", chunks)
        scored_no_norm = reranker_no_norm.score("test", chunks)

        assert len(scored_norm) == 2
        assert len(scored_no_norm) == 2

    def test_get_vector_exposed(self) -> None:
        """Test that get_vector is exposed for MMR."""
        reranker = LexicalReranker()

        vec = reranker.get_vector("test document")

        assert isinstance(vec, dict)
        assert len(vec) > 0
        assert "test" in vec
        assert "document" in vec


class TestCohereReranker:
    """Tests for CohereReranker."""

    def test_missing_cohere_raises_import_error(self) -> None:
        """Test that missing cohere library raises clear error."""
        import sys
        from unittest.mock import patch

        with (
            patch.dict(sys.modules, {"cohere": None}),
            pytest.raises(ImportError, match="cohere library is required"),
        ):
            CohereReranker(api_key="test_key")

    def test_missing_api_key_raises_value_error(self) -> None:
        """Test that missing API key raises ValueError."""
        with pytest.raises(ValueError, match="api_key is required"):
            CohereReranker(api_key=None, client=None)

    @pytest.mark.skipif(
        not os.getenv("COHERE_API_KEY"),
        reason="COHERE_API_KEY not set",
    )
    def test_cohere_integration(self) -> None:
        """Integration test with real Cohere API."""
        try:
            import importlib.util

            if not importlib.util.find_spec("cohere"):
                pytest.skip("cohere not installed")
        except ImportError:
            pytest.skip("cohere not installed")

        reranker = CohereReranker(
            api_key=os.getenv("COHERE_API_KEY"),
            model="rerank-english-v3.0",
        )

        chunks = [
            DocumentChunk(
                id="1",
                text="How to reset your password",
                metadata=ChunkMetadata(source_id="doc1", chunk_index=0),
            ),
            DocumentChunk(
                id="2",
                text="How to create a new account",
                metadata=ChunkMetadata(source_id="doc1", chunk_index=1),
            ),
            DocumentChunk(
                id="3",
                text="Password reset not receiving email",
                metadata=ChunkMetadata(source_id="doc1", chunk_index=2),
            ),
        ]

        scored = reranker.score("reset password email not working", chunks)

        assert len(scored) > 0
        assert all(isinstance(s, ScoredChunk) for s in scored)
        assert scored[0].score >= 0.0


class TestRerankerPropertyBased:
    """Property-based tests for rerankers."""

    def test_score_ordering(self) -> None:
        """Test that scores are returned in descending order."""
        reranker = LexicalReranker()

        chunks = [
            DocumentChunk(
                id=str(i),
                text=f"document {i} with various content",
                metadata=ChunkMetadata(source_id="doc1", chunk_index=i),
            )
            for i in range(10)
        ]

        scored = reranker.score("document content", chunks)

        scores = [s.score for s in scored]
        assert scores == sorted(scores, reverse=True)

    def test_all_chunks_returned(self) -> None:
        """Test that all chunks are returned."""
        reranker = LexicalReranker()

        chunks = [
            DocumentChunk(
                id=str(i),
                text=f"content {i}",
                metadata=ChunkMetadata(source_id="doc1", chunk_index=i),
            )
            for i in range(5)
        ]

        scored = reranker.score("query", chunks)

        assert len(scored) == len(chunks)

        chunk_ids = {c.id for c in chunks}
        scored_ids = {s.chunk.id for s in scored}
        assert chunk_ids == scored_ids
