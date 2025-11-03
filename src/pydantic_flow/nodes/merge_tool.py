"""MergeToolNode implementation for multi-input tool execution."""

from collections.abc import AsyncIterator
from collections.abc import Awaitable
from collections.abc import Callable
import inspect
from typing import Any
import uuid

from pydantic import BaseModel

from pydantic_flow.cache.base import CachePolicy
from pydantic_flow.nodes.base import MergeNode
from pydantic_flow.nodes.base import NodeOutput
from pydantic_flow.streaming.base import ProgressItem
from pydantic_flow.streaming.core_events import GenericResult
from pydantic_flow.streaming.core_events import StreamEnd
from pydantic_flow.streaming.core_events import StreamStart
from pydantic_flow.streaming.tool_events import ToolCall
from pydantic_flow.streaming.tool_events import ToolResult


class MergeToolNode[*InputTs, OutputModel: BaseModel](MergeNode[*InputTs, OutputModel]):
    """A tool node that merges multiple inputs before processing.

    This node enables fan-in patterns where a tool needs to combine
    outputs from multiple upstream nodes.

    Example:
        node_a = ToolNode[Input, DataA](tool_func=get_data_a, name="A")
        node_b = ToolNode[Input, DataB](tool_func=get_data_b, name="B")

        async def combine(data_a: DataA, data_b: DataB) -> Result:
            return Result(combined=f"{data_a} + {data_b}")

        merge_node = MergeToolNode[DataA, DataB, Result](
            inputs=(node_a.output, node_b.output),
            tool_func=combine,
            name="merge"
        )

    """

    def __init__(
        self,
        tool_func: Callable[..., Awaitable[OutputModel]],
        *,
        inputs: tuple[NodeOutput[Any], ...],
        name: str | None = None,
        cache_policy: CachePolicy | None = None,
    ) -> None:
        """Initialize a MergeToolNode.

        Args:
            tool_func: Async function that combines multiple inputs into output.
                      Should accept arguments matching the input types.
            inputs: Tuple of NodeOutput references from upstream nodes
            name: Optional unique identifier for this node
            cache_policy: Optional cache policy for this node

        Raises:
            ValueError: If tool_func is not an async function

        """
        super().__init__(inputs, name, cache_policy=cache_policy)

        # Validate that tool_func is async
        if not inspect.iscoroutinefunction(tool_func):
            func_name = getattr(tool_func, "__name__", repr(tool_func))
            msg = (
                f"MergeToolNode requires an async function, but {func_name} "
                "is not async. Please define your function with 'async def'."
            )
            raise ValueError(msg)

        self.tool_func = tool_func

    async def astream(self, input_data: tuple[Any, ...]) -> AsyncIterator[ProgressItem]:
        """Stream progress items while executing the merge tool.

        Yields:
            StreamStart, ToolCall, ToolResult, and StreamEnd.

        """
        call_id = str(uuid.uuid4())
        run_id = self.run_id or ""
        node_id = self.name

        yield StreamStart(run_id=run_id, node_id=node_id)

        # Emit tool call intent
        yield ToolCall(
            run_id=run_id,
            node_id=node_id,
            tool_name=self.tool_func.__name__,
            call_id=call_id,
        )

        # Execute the tool with unpacked inputs (await the async function)
        try:
            result = await self.tool_func(*input_data)
            yield ToolResult(
                run_id=run_id,
                node_id=node_id,
                tool_name=self.tool_func.__name__,
                call_id=call_id,
                result=result,
            )

            # Prepare result as BaseModel
            if isinstance(result, BaseModel):
                result_model = result
            else:
                result_model = GenericResult(value=result)

            yield StreamEnd(run_id=run_id, node_id=node_id, result=result_model)
        except Exception as e:
            yield ToolResult(
                run_id=run_id,
                node_id=node_id,
                tool_name=self.tool_func.__name__,
                call_id=call_id,
                error=str(e),
            )
            raise
