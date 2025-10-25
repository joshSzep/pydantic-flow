"""IfNode implementation for conditional branching."""

from collections.abc import AsyncIterator
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from pydantic_flow.nodes.base import NodeOutput
from pydantic_flow.nodes.base import NodeWithInput
from pydantic_flow.streaming.events import ProgressItem
from pydantic_flow.streaming.events import StreamEnd
from pydantic_flow.streaming.events import StreamStart


class IfNode[OutputModel: BaseModel](NodeWithInput[Any, OutputModel]):
    """A node that evaluates a predicate and branches to different nodes.

    This node enables conditional execution paths in workflows.
    """

    def __init__(
        self,
        predicate: Callable[[Any], bool],
        if_true: NodeWithInput[Any, OutputModel],
        if_false: NodeWithInput[Any, OutputModel],
        *,
        input: NodeOutput[Any] | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize an IfNode.

        Args:
            predicate: Function that evaluates the condition
            if_true: Node to execute if predicate returns True
            if_false: Node to execute if predicate returns False
            input: Optional input from another node's output
            name: Optional unique identifier for this node

        """
        super().__init__(input, name)
        self.predicate = predicate
        self.if_true = if_true
        self.if_false = if_false

    @property
    def dependencies(self) -> list[Any]:
        """Get the list of nodes this node depends on."""
        deps = []
        if self.input:
            deps.extend(super().dependencies)
        deps.extend(self.if_true.dependencies)
        deps.extend(self.if_false.dependencies)
        return deps

    async def astream(self, input_data: Any) -> AsyncIterator[ProgressItem]:
        """Stream progress items while executing the conditional branch.

        Yields:
            StreamStart, progress from the chosen branch, and StreamEnd.

        """
        run_id = self.run_id or ""
        node_id = self.name

        yield StreamStart(run_id=run_id, node_id=node_id)

        # Evaluate predicate and choose branch
        chosen_branch = self.if_true if self.predicate(input_data) else self.if_false

        # Stream from the chosen branch
        result_preview = None
        async for item in chosen_branch.astream(input_data):
            # Forward all items from the branch, but don't forward its StreamEnd
            if not isinstance(item, StreamEnd):
                yield item
            else:
                # Save the result from the branch's StreamEnd
                result_preview = item.result_preview

        # Emit our own StreamEnd with the branch's result
        yield StreamEnd(run_id=run_id, node_id=node_id, result_preview=result_preview)
