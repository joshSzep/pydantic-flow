"""EmbeddingNode for materializing embeddings."""

from collections.abc import AsyncIterator
import uuid

from pydantic import BaseModel

from pydantic_flow.nodes.base import NodeOutput
from pydantic_flow.nodes.base import NodeWithInput
from pydantic_flow.rag.embeddings.base import EmbeddingProvider
from pydantic_flow.streaming.base import ProgressItem
from pydantic_flow.streaming.core_events import StreamEnd
from pydantic_flow.streaming.core_events import StreamStart


class EmbeddingInput(BaseModel):
    """Input for embedding node.

    Attributes:
        texts: List of texts to embed.

    """

    texts: list[str]


class EmbeddingOutput(BaseModel):
    """Output from embedding node.

    Attributes:
        embeddings: List of embedding vectors.
        dimensions: Embedding dimension.

    """

    embeddings: list[list[float]]
    dimensions: int


class EmbeddingNode(NodeWithInput[EmbeddingInput, EmbeddingOutput]):
    """Node that materializes embeddings for downstream use.

    This node takes an EmbeddingProvider and generates embeddings
    for input texts, making them available to downstream nodes.

    Attributes:
        embedding_provider: Provider for generating embeddings.

    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        *,
        input: NodeOutput[EmbeddingInput] | None = None,
        name: str | None = None,
        run_id: str | None = None,
    ) -> None:
        """Initialize EmbeddingNode.

        Args:
            embedding_provider: Embedding provider instance.
            input: Optional input from another node's output.
            name: Optional unique identifier for this node.
            run_id: Optional run identifier for tracking execution.

        """
        super().__init__(input, name, run_id)
        self.embedding_provider = embedding_provider

    async def astream(self, input_data: EmbeddingInput) -> AsyncIterator[ProgressItem]:
        """Generate embeddings and stream result.

        Yields:
            StreamStart, StreamEnd with embeddings.

        """
        actual_run_id = self.run_id or str(uuid.uuid4())

        yield StreamStart(
            run_id=actual_run_id,
            node_id=self.name,
            input_preview={"text_count": len(input_data.texts)},
        )

        embeddings = await self.embedding_provider.embed(input_data.texts)
        dimensions = self.embedding_provider.dim()

        output = EmbeddingOutput(embeddings=embeddings, dimensions=dimensions)

        # Yield the actual result object
        from pydantic_flow.streaming.tool_events import ToolResult  # noqa: PLC0415

        yield ToolResult(
            run_id=actual_run_id,
            node_id=self.name,
            tool_name="embed",
            result=output,
        )

        yield StreamEnd(
            run_id=actual_run_id,
            node_id=self.name,
            result_preview=output.model_dump(),
        )
