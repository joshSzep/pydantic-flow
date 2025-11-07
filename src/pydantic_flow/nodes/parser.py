"""ParserNode implementation for custom output transformation."""

from collections.abc import AsyncIterator
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from pydantic_flow.cache.base import CachePolicy
from pydantic_flow.nodes.base import BaseNode
from pydantic_flow.nodes.base import NodeOutput
from pydantic_flow.streaming.base import ProgressItem
from pydantic_flow.streaming.core_events import GenericResult
from pydantic_flow.streaming.core_events import StreamEnd
from pydantic_flow.streaming.core_events import StreamStart
from pydantic_flow.streaming.tool_events import ToolResult


class ParserNode[InputT, OutputModel: BaseModel](BaseNode[InputT, OutputModel]):
    """A node that applies a Python function to transform input data.

    This node allows for custom transformation logic between workflow steps.
    Now supports caching like all other nodes.

    Supports single or multiple inputs via the inputs parameter:
    - Single input: inputs=node.output, parser_func takes one argument
    - Multiple inputs: inputs=(node1.output, node2.output, ...),
                      parser_func takes multiple arguments (unpacked from tuple)
    - Entry node: inputs=None
    """

    def __init__(
        self,
        parser_func: Callable[[InputT], OutputModel] | Callable[..., OutputModel],
        *,
        inputs: tuple[NodeOutput, ...] | None = None,
        name: str | None = None,
        cache_policy: CachePolicy | None = None,
    ) -> None:
        """Initialize a ParserNode.

        Args:
            parser_func: Function to transform input(s) to output.
                        For single input: takes one argument.
                        For multiple inputs: takes multiple arguments
                        (will be unpacked).
            inputs: Optional tuple of inputs from other nodes:
                   - None: Entry node with no dependencies
                   - (node.output,): Single input dependency
                   - (node1.output, node2.output, ...): Multiple inputs (fan-in)
            name: Optional unique identifier for this node
            cache_policy: Optional cache policy for this node

        """
        super().__init__(inputs, name, cache_policy=cache_policy)
        self.parser_func = parser_func

    async def astream(
        self, input_data: InputT | tuple[Any, ...]
    ) -> AsyncIterator[ProgressItem]:
        """Stream progress items while executing the parser.

        Yields:
            StreamStart, and StreamEnd with the transformed result.

        """
        run_id = self.run_id or ""
        node_id = self.name

        yield StreamStart(run_id=run_id, node_id=node_id)

        # Execute the parser function
        if isinstance(input_data, tuple):
            # Multiple inputs: unpack them
            result = self.parser_func(*input_data)
        else:
            # Single input
            result = self.parser_func(input_data)

        # Emit result (actual object for run() to extract)
        yield ToolResult(
            run_id=run_id,
            node_id=node_id,
            tool_name="parser",
            call_id="",
            result=result,
            error=None,
        )

        # Prepare result as BaseModel
        if isinstance(result, BaseModel):
            result_model = result
        else:
            result_model = GenericResult(value=result)

        yield StreamEnd(run_id=run_id, node_id=node_id, result=result_model)
