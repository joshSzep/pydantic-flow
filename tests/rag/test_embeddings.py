"""Tests for RAG embeddings."""

import pytest

from pydantic_flow.rag.embeddings.base import EmbeddingProvider


class MockEmbeddings(EmbeddingProvider):
    """Mock embedding provider for testing."""

    def __init__(self, dim: int = 128):
        """Initialize mock embeddings.

        Args:
            dim: Embedding dimension.

        """
        self.dimension = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate mock embeddings.

        Args:
            texts: List of text strings.

        Returns:
            List of mock embeddings.

        """
        return [[0.1] * self.dimension for _ in texts]

    def dim(self) -> int:
        """Return embedding dimension.

        Returns:
            Embedding dimension.

        """
        return self.dimension


@pytest.mark.asyncio
async def test_embedding_dimension_consistency():
    """Test that embedding dimensions are consistent."""
    provider = MockEmbeddings(dim=256)

    texts = ["hello", "world", "test"]
    embeddings = await provider.embed(texts)

    assert len(embeddings) == len(texts)
    assert provider.dim() == 256
    for emb in embeddings:
        assert len(emb) == 256


@pytest.mark.asyncio
async def test_embedding_batch_behavior():
    """Test batch embedding behavior."""
    provider = MockEmbeddings(dim=128)

    single = await provider.embed(["test"])
    assert len(single) == 1
    assert len(single[0]) == 128

    batch = await provider.embed(["one", "two", "three", "four", "five"])
    assert len(batch) == 5
    for emb in batch:
        assert len(emb) == 128


@pytest.mark.asyncio
async def test_empty_batch():
    """Test embedding empty batch."""
    provider = MockEmbeddings(dim=64)

    result = await provider.embed([])
    assert len(result) == 0


@pytest.mark.asyncio
async def test_openai_embeddings():
    """Test OpenAI embeddings integration."""
    from unittest.mock import AsyncMock
    from unittest.mock import MagicMock
    from unittest.mock import patch

    from pydantic_flow.rag.embeddings.openai import OpenAIEmbeddings

    # Mock the OpenAI client
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.1] * 512)]

    with patch("pydantic_flow.rag.embeddings.openai.AsyncOpenAI") as mock_client:
        mock_client.return_value.embeddings.create = AsyncMock(
            return_value=mock_response
        )

        provider = OpenAIEmbeddings(dimensions=512)

        texts = ["hello world"]
        embeddings = await provider.embed(texts)

        assert len(embeddings) == 1
        assert len(embeddings[0]) == 512
        assert provider.dim() == 512


@pytest.mark.asyncio
async def test_huggingface_embeddings():
    """Test HuggingFace embeddings."""
    from unittest.mock import MagicMock
    from unittest.mock import patch

    import numpy as np

    # Mock sentence_transformers
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1] * 384, [0.2] * 384])
    mock_model.get_sentence_embedding_dimension.return_value = 384

    with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
        from pydantic_flow.rag.embeddings.huggingface import HuggingFaceEmbeddings

        provider = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        texts = ["hello world", "test sentence"]
        embeddings = await provider.embed(texts)

        assert len(embeddings) == 2
        dim = provider.dim()
        assert dim == 384
        for emb in embeddings:
            assert len(emb) == dim
