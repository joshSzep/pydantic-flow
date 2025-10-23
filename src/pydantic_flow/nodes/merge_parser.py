"""MergeParserNode implementation for multi-input parsing."""

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from pydantic_flow.nodes.base import MergeNode
from pydantic_flow.nodes.base import NodeOutput


class MergeParserNode[*InputTs, OutputModel: BaseModel](
    MergeNode[*InputTs, OutputModel]
):
    """A parser node that merges multiple inputs before processing.

    This node enables fan-in patterns where a parser needs to combine
    outputs from multiple upstream nodes.

    Example:
        node_a = PromptNode[Input, str](prompt="Get A", name="A")
        node_b = PromptNode[Input, str](prompt="Get B", name="B")

        def parse_both(text_a: str, text_b: str) -> Result:
            return Result(combined=f"{text_a} and {text_b}")

        merge_node = MergeParserNode[str, str, Result](
            inputs=(node_a.output, node_b.output),
            parser_func=parse_both,
            name="merge"
        )

    """

    def __init__(
        self,
        parser_func: Callable[..., OutputModel],
        *,
        inputs: tuple[NodeOutput[Any], ...],
        name: str | None = None,
    ) -> None:
        """Initialize a MergeParserNode.

        Args:
            parser_func: Function that combines and parses multiple inputs.
                        Should accept arguments matching the input types.
            inputs: Tuple of NodeOutput references from upstream nodes
            name: Optional unique identifier for this node

        """
        super().__init__(inputs, name)
        self.parser_func = parser_func

    async def run(self, input_data: tuple[Any, ...]) -> OutputModel:
        """Execute the parser function with merged inputs.

        Args:
            input_data: Tuple of data from all dependency nodes

        Returns:
            The parsed output data

        """
        return self.parser_func(*input_data)
