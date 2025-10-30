"""HuggingFace embeddings provider using sentence-transformers."""

from pydantic_flow.rag.embeddings.base import EmbeddingProvider


class HuggingFaceEmbeddings(EmbeddingProvider):
    """HuggingFace embeddings via sentence-transformers.

    Attributes:
        model: SentenceTransformer model instance.
        model_name: Model identifier.

    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        """Initialize HuggingFace embeddings provider.

        Args:
            model_name: SentenceTransformer model name or path.

        """
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using sentence-transformers.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.

        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return [emb.tolist() for emb in embeddings]

    def dim(self) -> int:
        """Return embedding dimension.

        Returns:
            Embedding dimension.

        """
        return self.model.get_sentence_embedding_dimension()
