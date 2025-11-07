"""Retriever node for incremental search results streaming."""

from collections.abc import AsyncIterator
from typing import Any
import uuid

from pydantic import BaseModel

from pydantic_flow.cache.base import CachePolicy
from pydantic_flow.nodes.base import BaseNode
from pydantic_flow.nodes.base import NodeOutput
from pydantic_flow.streaming.base import ProgressItem
from pydantic_flow.streaming.core_events import GenericResult
from pydantic_flow.streaming.core_events import StreamEnd
from pydantic_flow.streaming.core_events import StreamStart
from pydantic_flow.streaming.retrieval_events import RetrievalItem


class RetrieverNode[QueryModel: BaseModel, ResultModel: BaseModel](
    BaseNode[QueryModel, ResultModel]
):
    """Streaming retriever that yields search results progressively.

    This demonstrates the pattern for incremental retrieval where
    downstream nodes can react before all results are gathered.
    """

    def __init__(
        self,
        retriever_fn: Any,
        *,
        inputs: tuple[NodeOutput, ...] | None = None,
        name: str | None = None,
        run_id: str | None = None,
        cache_policy: CachePolicy | None = None,
    ) -> None:
        """Initialize a RetrieverNode.

        Args:
            retriever_fn: Async function that yields retrieval results.
            inputs: Optional tuple of inputs from other nodes.
            name: Optional unique identifier for this node.
            run_id: Optional run identifier for tracking execution.
            cache_policy: Optional cache policy for this node.

        """
        super().__init__(inputs, name, run_id, cache_policy)
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
            result=GenericResult(value=results),
        )
