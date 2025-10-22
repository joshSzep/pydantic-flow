"""ParserNode implementation for data transformation."""

from collections.abc import Callable

from pydantic import BaseModel

from pflow.nodes.base import NodeOutput
from pflow.nodes.base import NodeWithInput


class ParserNode[InputT, OutputModel: BaseModel](NodeWithInput[InputT, OutputModel]):
    """A node that applies a Python function to transform input data.

    This node allows for custom transformation logic between workflow steps.
    """

    def __init__(
        self,
        parser_func: Callable[[InputT], OutputModel],
        *,
        input: NodeOutput[InputT] | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize a ParserNode.

        Args:
            parser_func: Function to transform input to output
            input: Optional input from another node's output
            name: Optional unique identifier for this node

        """
        super().__init__(input, name)
        self.parser_func = parser_func

    async def run(self, input_data: InputT) -> OutputModel:
        """Execute the parser function with the given input.

        Args:
            input_data: The input data for this node

        Returns:
            The transformed output data

        """
        return self.parser_func(input_data)
