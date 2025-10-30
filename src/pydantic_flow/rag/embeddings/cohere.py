"""Cohere embeddings provider."""

import httpx

from pydantic_flow.rag.embeddings.base import EmbeddingProvider


class CohereEmbeddings(EmbeddingProvider):
    """Cohere embeddings using httpx client.

    Attributes:
        api_key: Cohere API key.
        model: Model name (default: embed-english-v3.0).
        dimensions: Embedding dimension.

    """

    def __init__(
        self,
        api_key: str,
        model: str = "embed-english-v3.0",
        dimensions: int = 1024,
    ) -> None:
        """Initialize Cohere embeddings provider.

        Args:
            api_key: Cohere API key.
            model: Cohere embedding model name.
            dimensions: Embedding dimension.

        """
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.base_url = "https://api.cohere.ai/v1"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using Cohere API.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.

        """
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/embed",
                json={
                    "texts": texts,
                    "model": self.model,
                    "input_type": "search_document",
                },
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
            return data["embeddings"]

    def dim(self) -> int:
        """Return embedding dimension.

        Returns:
            Embedding dimension.

        """
        return self.dimensions
