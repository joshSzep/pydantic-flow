"""ToolNode implementation for custom function execution."""

from collections.abc import AsyncIterator
from collections.abc import Awaitable
from collections.abc import Callable
import inspect
import uuid

from pydantic import BaseModel

from pydantic_flow.cache.base import CachePolicy
from pydantic_flow.nodes.base import Node
from pydantic_flow.nodes.base import NodeOutput
from pydantic_flow.streaming.base import ProgressItem
from pydantic_flow.streaming.core_events import GenericResult
from pydantic_flow.streaming.core_events import StreamEnd
from pydantic_flow.streaming.core_events import StreamStart
from pydantic_flow.streaming.tool_events import ToolCall
from pydantic_flow.streaming.tool_events import ToolResult


class ToolNode[InputModel: BaseModel, OutputModel: BaseModel](
    Node[InputModel, OutputModel]
):
    """A node that calls an external tool using an async function.

    This node enables integration with external APIs, databases, or other services.
    As an async-first framework, only async functions are supported.
    """

    def __init__(
        self,
        tool_func: Callable[[InputModel], Awaitable[OutputModel]],
        *,
        input: NodeOutput[InputModel] | None = None,
        name: str | None = None,
        cache_policy: CachePolicy | None = None,
    ) -> None:
        """Initialize a ToolNode.

        Args:
            tool_func: Async function that implements the tool call
            input: Optional input from another node's output
            name: Optional unique identifier for this node
            cache_policy: Optional cache policy for this node

        Raises:
            ValueError: If tool_func is not an async function

        """
        super().__init__(input, name, cache_policy=cache_policy)

        # Validate that tool_func is async
        if not inspect.iscoroutinefunction(tool_func):
            func_name = getattr(tool_func, "__name__", repr(tool_func))
            msg = (
                f"ToolNode requires an async function, but {func_name} "
                "is not async. Please define your function with 'async def'."
            )
            raise ValueError(msg)

        self.tool_func = tool_func

    async def astream(self, input_data: InputModel) -> AsyncIterator[ProgressItem]:
        """Stream progress items while executing the tool.

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

        # Execute the tool (await the async function)
        try:
            result = await self.tool_func(input_data)
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

            yield StreamEnd(
                run_id=run_id,
                node_id=node_id,
                result=result_model,
            )
        except Exception as e:
            yield ToolResult(
                run_id=run_id,
                node_id=node_id,
                tool_name=self.tool_func.__name__,
                call_id=call_id,
                error=str(e),
            )
            raise
