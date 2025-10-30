"""VectorRetrieverNode for streaming RAG retrieval."""

from collections.abc import AsyncIterator
import uuid

from pydantic import BaseModel

from pydantic_flow.nodes.base import NodeOutput
from pydantic_flow.nodes.base import NodeWithInput
from pydantic_flow.rag.retrievers.base import Retriever
from pydantic_flow.streaming.events import ProgressItem
from pydantic_flow.streaming.events import RetrievalItem
from pydantic_flow.streaming.events import StreamEnd
from pydantic_flow.streaming.events import StreamStart


class QueryInput(BaseModel):
    """Query input for retriever node.

    Attributes:
        query: Query string.
        k: Number of results to retrieve.

    """

    query: str
    k: int = 5


class RetrievalResult(BaseModel):
    """Retrieval result output.

    Attributes:
        documents: List of retrieved documents.
        query: Original query string.

    """

    documents: list[dict]
    query: str


class VectorRetrieverNode(NodeWithInput[QueryInput, RetrievalResult]):
    """Node that streams retrieval results using a Retriever.

    This node takes a Retriever and yields RetrievalItem events
    for each retrieved document, maintaining compatibility with
    the pydantic-flow streaming vocabulary.

    Attributes:
        retriever: Retriever instance for semantic search.

    """

    def __init__(
        self,
        retriever: Retriever,
        *,
        input: NodeOutput[QueryInput] | None = None,
        name: str | None = None,
        run_id: str | None = None,
    ) -> None:
        """Initialize VectorRetrieverNode.

        Args:
            retriever: Retriever instance.
            input: Optional input from another node's output.
            name: Optional unique identifier for this node.
            run_id: Optional run identifier for tracking execution.

        """
        super().__init__(input, name, run_id)
        self.retriever = retriever

    async def astream(self, input_data: QueryInput) -> AsyncIterator[ProgressItem]:
        """Stream retrieval results.

        Yields:
            StreamStart, RetrievalItem for each document, and StreamEnd.

        """
        actual_run_id = self.run_id or str(uuid.uuid4())

        yield StreamStart(
            run_id=actual_run_id,
            node_id=self.name,
            input_preview=input_data.model_dump(),
        )

        documents = await self.retriever.retrieve(
            query=input_data.query,
            k=input_data.k,
        )

        doc_dicts = []
        for document in documents:
            doc_dict = {
                "id": document.id,
                "content": document.content,
                "metadata": document.metadata.model_dump(),
            }
            doc_dicts.append(doc_dict)

            yield RetrievalItem(
                item_id=document.id,
                content=document.content,
                score=None,
                metadata=document.metadata.model_dump(),
                run_id=actual_run_id,
                node_id=self.name,
            )

        result = RetrievalResult(documents=doc_dicts, query=input_data.query)

        yield StreamEnd(
            run_id=actual_run_id,
            node_id=self.name,
            result_preview=result.model_dump(),
        )
