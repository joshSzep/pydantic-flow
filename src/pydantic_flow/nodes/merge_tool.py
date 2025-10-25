"""MergeToolNode implementation for multi-input tool execution."""

from collections.abc import AsyncIterator
from collections.abc import Callable
from typing import Any
import uuid

from pydantic import BaseModel

from pydantic_flow.nodes.base import MergeNode
from pydantic_flow.nodes.base import NodeOutput
from pydantic_flow.streaming.events import ProgressItem
from pydantic_flow.streaming.events import StreamEnd
from pydantic_flow.streaming.events import StreamStart
from pydantic_flow.streaming.events import ToolCall
from pydantic_flow.streaming.events import ToolResult


class MergeToolNode[*InputTs, OutputModel: BaseModel](MergeNode[*InputTs, OutputModel]):
    """A tool node that merges multiple inputs before processing.

    This node enables fan-in patterns where a tool needs to combine
    outputs from multiple upstream nodes.

    Example:
        node_a = ToolNode[Input, DataA](tool_func=get_data_a, name="A")
        node_b = ToolNode[Input, DataB](tool_func=get_data_b, name="B")

        def combine(data_a: DataA, data_b: DataB) -> Result:
            return Result(combined=f"{data_a} + {data_b}")

        merge_node = MergeToolNode[DataA, DataB, Result](
            inputs=(node_a.output, node_b.output),
            tool_func=combine,
            name="merge"
        )

    """

    def __init__(
        self,
        tool_func: Callable[..., OutputModel],
        *,
        inputs: tuple[NodeOutput[Any], ...],
        name: str | None = None,
    ) -> None:
        """Initialize a MergeToolNode.

        Args:
            tool_func: Function that combines multiple inputs into output.
                      Should accept arguments matching the input types.
            inputs: Tuple of NodeOutput references from upstream nodes
            name: Optional unique identifier for this node

        """
        super().__init__(inputs, name)
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

        # Execute the tool with unpacked inputs
        try:
            result = self.tool_func(*input_data)
            yield ToolResult(
                run_id=run_id,
                node_id=node_id,
                tool_name=self.tool_func.__name__,
                call_id=call_id,
                result=result,
            )

            # Prepare result preview
            result_preview = None
            if hasattr(result, "model_dump"):
                result_preview = result.model_dump()
            elif result is not None:
                result_preview = {"value": str(result)}

            yield StreamEnd(
                run_id=run_id, node_id=node_id, result_preview=result_preview
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
