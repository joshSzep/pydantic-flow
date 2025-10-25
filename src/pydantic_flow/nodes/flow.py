"""FlowNode implementation for composable sub-flows."""

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING
from typing import Any

from pydantic import BaseModel

from pydantic_flow.nodes.base import BaseNode
from pydantic_flow.nodes.base import NodeOutput
from pydantic_flow.nodes.base import NodeWithInput
from pydantic_flow.streaming.events import ProgressItem
from pydantic_flow.streaming.events import StreamEnd
from pydantic_flow.streaming.events import StreamStart
from pydantic_flow.streaming.events import ToolResult

if TYPE_CHECKING:
    from pydantic_flow.flow.flow import Flow


class FlowNode[InputModel: BaseModel, OutputModel: BaseModel](
    NodeWithInput[InputModel, OutputModel]
):
    """A node that wraps a Flow, enabling sub-flows within larger workflows.

    This node allows for hierarchical composition of flows, where a complete
    Flow can be used as a single node within another Flow. This enables
    building complex workflows from simpler, reusable sub-flows.
    """

    def __init__(
        self,
        flow: Flow[InputModel, OutputModel],
        *,
        input: NodeOutput[InputModel] | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize a FlowNode with a wrapped Flow.

        Args:
            flow: The Flow to wrap as a node. The flow's input and output types
                 must match the FlowNode's type parameters.
            input: Optional input from another node's output
            name: Optional unique identifier for this node. If not provided,
                 will use the format "FlowNode_{flow_repr}"

        """
        # Generate a meaningful default name that includes the wrapped flow info
        if name is None:
            flow_repr = repr(flow)
            name = f"FlowNode_{flow_repr}"

        super().__init__(input, name)
        self.flow = flow

    async def astream(self, input_data: InputModel) -> AsyncIterator[ProgressItem]:
        """Stream progress items while executing the wrapped flow.

        Yields:
            StreamStart, progress from the wrapped flow, and StreamEnd.

        """
        run_id = self.run_id or ""
        node_id = self.name

        yield StreamStart(run_id=run_id, node_id=node_id)

        # Check if the flow has an astream method (future enhancement)
        if hasattr(self.flow, "astream"):
            # Stream from the flow
            result = None
            result_preview = None
            async for item in self.flow.astream(input_data):  # type: ignore
                # Don't forward StreamStart/StreamEnd from wrapped flow
                if isinstance(item, StreamStart):
                    continue
                elif isinstance(item, StreamEnd):
                    # Capture result
                    result_preview = item.result_preview
                elif isinstance(item, ToolResult) and item.result:
                    # Capture actual result if available
                    result = item.result
                else:
                    # Forward other progress items
                    yield item

            # Emit ToolResult with actual result if we have it
            if result is not None:
                yield ToolResult(
                    run_id=run_id,
                    node_id=node_id,
                    tool_name="flow",
                    call_id="",
                    result=result,
                    error=None,
                )

            # Emit our own StreamEnd with the result
            yield StreamEnd(
                run_id=run_id, node_id=node_id, result_preview=result_preview
            )
        else:
            # Fall back to run() and wrap result
            result = await self.flow.run(input_data)

            # Emit ToolResult with the actual result
            yield ToolResult(
                run_id=run_id,
                node_id=node_id,
                tool_name="flow",
                call_id="",
                result=result,
                error=None,
            )

            result_preview = None
            if hasattr(result, "model_dump"):
                result_preview = result.model_dump()
            elif result is not None:
                result_preview = {"value": str(result)}

            yield StreamEnd(
                run_id=run_id, node_id=node_id, result_preview=result_preview
            )

    @property
    def dependencies(self) -> list[BaseNode[Any, Any]]:
        """Get the list of nodes this FlowNode depends on.

        Returns:
            List containing the input node if this FlowNode takes input from
            another node, otherwise an empty list.

        Note:
            The wrapped flow's internal dependencies are not exposed here
            since they are encapsulated within the flow execution.

        """
        return super().dependencies

    def __repr__(self) -> str:
        """Return a string representation of the FlowNode."""
        return f"FlowNode(name='{self.name}', flow={self.flow!r})"
