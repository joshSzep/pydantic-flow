"""ToolNode implementation for custom function execution."""

from collections.abc import Callable

from pydantic import BaseModel

from pydantic_flow.nodes.base import NodeOutput
from pydantic_flow.nodes.base import NodeWithInput


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

    async def run(self, input_data: InputModel) -> OutputModel:
        """Execute the tool function with the given input.

        Args:
            input_data: The input data for this node

        Returns:
            The tool's output data

        """
        return self.tool_func(input_data)
