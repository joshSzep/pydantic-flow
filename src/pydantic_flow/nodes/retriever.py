"""Retriever node for incremental search results streaming."""

from collections.abc import AsyncIterator
from typing import Any
import uuid

from pydantic import BaseModel

from pydantic_flow.nodes.base import NodeWithInput
from pydantic_flow.streaming.events import ProgressItem
from pydantic_flow.streaming.events import RetrievalItem
from pydantic_flow.streaming.events import StreamEnd
from pydantic_flow.streaming.events import StreamStart


class RetrieverNode[QueryModel: BaseModel, ResultModel: BaseModel](
    NodeWithInput[QueryModel, ResultModel]
):
    """Streaming retriever that yields search results progressively.

    This demonstrates the pattern for incremental retrieval where
    downstream nodes can react before all results are gathered.
    """

    def __init__(
        self,
        retriever_fn: Any,
        *,
        input: Any = None,
        name: str | None = None,
        run_id: str | None = None,
    ) -> None:
        """Initialize a RetrieverNode.

        Args:
            retriever_fn: Async function that yields retrieval results.
            input: Optional input from another node's output.
            name: Optional unique identifier for this node.
            run_id: Optional run identifier for tracking execution.

        """
        super().__init__(input, name, run_id)
        self.retriever_fn = retriever_fn

    async def astream(self, input_data: QueryModel) -> AsyncIterator[ProgressItem]:
        """Stream retrieval items as they are found.

        Yields:
            StreamStart, RetrievalItem for each result, and StreamEnd with
            aggregated results.

        """
        actual_run_id = self.run_id or str(uuid.uuid4())

        yield StreamStart(
            run_id=actual_run_id,
            node_id=self.name,
            input_preview=input_data.model_dump()
            if hasattr(input_data, "model_dump")
            else None,
        )

        results = []

        # Stream results from retriever function
        async for item in self.retriever_fn(input_data):
            # Emit retrieval item
            yield RetrievalItem(
                item_id=str(item.get("id", uuid.uuid4())),
                content=item.get("content"),
                score=item.get("score"),
                metadata=item.get("metadata", {}),
                run_id=actual_run_id,
                node_id=self.name,
            )
            results.append(item)

        # Emit end with aggregated results
        yield StreamEnd(
            run_id=actual_run_id,
            node_id=self.name,
            result_preview={"results": results},
        )
