"""MergeToolNode implementation for multi-input tool execution."""

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from pydantic_flow.nodes.base import MergeNode
from pydantic_flow.nodes.base import NodeOutput


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

    async def run(self, input_data: tuple[Any, ...]) -> OutputModel:
        """Execute the tool function with merged inputs.

        Args:
            input_data: Tuple of data from all dependency nodes

        Returns:
            The tool's output data

        """
        return self.tool_func(*input_data)
