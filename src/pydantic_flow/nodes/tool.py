"""ToolNode implementation for custom function execution."""

from collections.abc import AsyncIterator
from collections.abc import Callable
import uuid

from pydantic import BaseModel

from pydantic_flow.nodes.base import NodeOutput
from pydantic_flow.nodes.base import NodeWithInput
from pydantic_flow.streaming.events import ProgressItem
from pydantic_flow.streaming.events import StreamEnd
from pydantic_flow.streaming.events import StreamStart
from pydantic_flow.streaming.events import ToolCall
from pydantic_flow.streaming.events import ToolResult


class ToolNode[InputModel: BaseModel, OutputModel: BaseModel](
    NodeWithInput[InputModel, OutputModel]
):
    """A node that calls an external tool using a user-defined function.

    This node enables integration with external APIs, databases, or other services.
    """

    def __init__(
        self,
        tool_func: Callable[[InputModel], OutputModel],
        *,
        input: NodeOutput[InputModel] | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize a ToolNode.

        Args:
            tool_func: Function that implements the tool call
            input: Optional input from another node's output
            name: Optional unique identifier for this node

        """
        super().__init__(input, name)
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

        # Execute the tool
        try:
            result = self.tool_func(input_data)
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

            # Also store the actual result object in a special field
            # so run() can extract it directly without reconstruction
            yield StreamEnd(
                run_id=run_id,
                node_id=node_id,
                result_preview=result_preview or {"__result__": result},
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
