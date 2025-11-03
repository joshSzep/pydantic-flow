"""MergeParserNode implementation for multi-input parsing."""

from collections.abc import AsyncIterator
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from pydantic_flow.cache.base import CachePolicy
from pydantic_flow.nodes.base import MergeNode
from pydantic_flow.nodes.base import NodeOutput
from pydantic_flow.streaming.base import ProgressItem
from pydantic_flow.streaming.core_events import GenericResult
from pydantic_flow.streaming.core_events import StreamEnd
from pydantic_flow.streaming.core_events import StreamStart


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
        cache_policy: CachePolicy | None = None,
    ) -> None:
        """Initialize a MergeParserNode.

        Args:
            parser_func: Function that combines and parses multiple inputs.
                        Should accept arguments matching the input types.
            inputs: Tuple of NodeOutput references from upstream nodes
            name: Optional unique identifier for this node
            cache_policy: Optional cache policy for this node

        """
        super().__init__(inputs, name, cache_policy=cache_policy)
        self.parser_func = parser_func

    async def astream(self, input_data: tuple[Any, ...]) -> AsyncIterator[ProgressItem]:
        """Stream progress items while executing the merge parser.

        Yields:
            StreamStart, and StreamEnd with the parsed result.

        """
        run_id = self.run_id or ""
        node_id = self.name

        yield StreamStart(run_id=run_id, node_id=node_id)

        # Execute the parser function with unpacked inputs
        result = self.parser_func(*input_data)

        # Yield the actual result object
        from pydantic_flow.streaming.tool_events import ToolResult  # noqa: PLC0415

        yield ToolResult(
            run_id=run_id,
            node_id=node_id,
            tool_name="merge_parser",
            result=result,
        )

        # Prepare result as BaseModel
        if isinstance(result, BaseModel):
            result_model = result
        else:
            result_model = GenericResult(value=result)

        yield StreamEnd(run_id=run_id, node_id=node_id, result=result_model)
