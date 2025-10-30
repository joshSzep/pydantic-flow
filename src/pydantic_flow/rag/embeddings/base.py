"""Embedding provider abstract base class."""

from abc import ABC
from abc import abstractmethod


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers.

    Implementations must provide embed() and dim() methods.
    """

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors, one per input text.

        """
        ...

    @abstractmethod
    def dim(self) -> int:
        """Return the dimensionality of embeddings produced by this provider.

        Returns:
            Embedding dimension.

        """
        ...
