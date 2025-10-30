"""OpenAI embeddings provider."""

from openai import AsyncOpenAI

from pydantic_flow.rag.embeddings.base import EmbeddingProvider


class OpenAIEmbeddings(EmbeddingProvider):
    """OpenAI embeddings using AsyncOpenAI client.

    Attributes:
        client: AsyncOpenAI client instance.
        model: Model name (default: text-embedding-3-small).
        dimensions: Embedding dimension for the model.

    """

    def __init__(
        self,
        client: AsyncOpenAI | None = None,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
    ) -> None:
        """Initialize OpenAI embeddings provider.

        Args:
            client: Optional AsyncOpenAI client. If not provided, creates default.
            model: OpenAI embedding model name.
            dimensions: Embedding dimension.

        """
        self.client = client or AsyncOpenAI()
        self.model = model
        self.dimensions = dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using OpenAI API.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.

        """
        response = await self.client.embeddings.create(
            input=texts,
            model=self.model,
        )
        return [item.embedding for item in response.data]

    def dim(self) -> int:
        """Return embedding dimension.

        Returns:
            Embedding dimension.

        """
        return self.dimensions
