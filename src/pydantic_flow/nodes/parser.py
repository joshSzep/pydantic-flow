"""ParserNode implementation for custom output transformation."""

from collections.abc import AsyncIterator
from collections.abc import Callable

from pydantic import BaseModel

from pydantic_flow.nodes.base import NodeOutput
from pydantic_flow.nodes.base import NodeWithInput
from pydantic_flow.streaming.base import ProgressItem
from pydantic_flow.streaming.core_events import GenericResult
from pydantic_flow.streaming.core_events import StreamEnd
from pydantic_flow.streaming.core_events import StreamStart
from pydantic_flow.streaming.tool_events import ToolResult


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

    async def astream(self, input_data: InputT) -> AsyncIterator[ProgressItem]:
        """Stream progress items while executing the parser.

        Yields:
            StreamStart, and StreamEnd with the transformed result.

        """
        run_id = self.run_id or ""
        node_id = self.name

        yield StreamStart(run_id=run_id, node_id=node_id)

        # Execute the parser function
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
