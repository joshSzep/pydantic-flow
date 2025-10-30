"""Ollama embeddings provider."""

import httpx

from pydantic_flow.rag.embeddings.base import EmbeddingProvider


class OllamaEmbeddings(EmbeddingProvider):
    """Ollama embeddings using httpx client.

    Attributes:
        model: Model name (default: llama2).
        base_url: Ollama API base URL.
        dimensions: Embedding dimension.

    """

    def __init__(
        self,
        model: str = "llama2",
        base_url: str = "http://localhost:11434",
        dimensions: int = 4096,
    ) -> None:
        """Initialize Ollama embeddings provider.

        Args:
            model: Ollama model name.
            base_url: Ollama server base URL.
            dimensions: Embedding dimension.

        """
        self.model = model
        self.base_url = base_url
        self.dimensions = dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using Ollama API.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.

        """
        embeddings = []
        async with httpx.AsyncClient() as client:
            for text in texts:
                response = await client.post(
                    f"{self.base_url}/api/embeddings",
                    json={
                        "model": self.model,
                        "prompt": text,
                    },
                    timeout=30.0,
                )
                response.raise_for_status()
                data = response.json()
                embeddings.append(data["embedding"])
        return embeddings

    def dim(self) -> int:
        """Return embedding dimension.

        Returns:
            Embedding dimension.

        """
        return self.dimensions
