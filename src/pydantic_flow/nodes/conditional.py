"""IfNode implementation for conditional branching."""

from collections.abc import AsyncIterator
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from pydantic_flow.cache.base import CachePolicy
from pydantic_flow.nodes.base import BaseNode
from pydantic_flow.nodes.base import NodeOutput
from pydantic_flow.streaming.base import ProgressItem
from pydantic_flow.streaming.core_events import StreamEnd
from pydantic_flow.streaming.core_events import StreamStart


class IfNode[OutputModel: BaseModel](BaseNode[Any, OutputModel]):
    """A node that evaluates a predicate and branches to different nodes.

    This node enables conditional execution paths in workflows.
    """

    def __init__(
        self,
        predicate: Callable[[Any], bool],
        if_true: BaseNode[Any, OutputModel],
        if_false: BaseNode[Any, OutputModel],
        *,
        inputs: tuple[NodeOutput, ...] | None = None,
        name: str | None = None,
        cache_policy: CachePolicy | None = None,
    ) -> None:
        """Initialize an IfNode.

        Args:
            predicate: Function that evaluates the condition
            if_true: Node to execute if predicate returns True
            if_false: Node to execute if predicate returns False
            inputs: Optional tuple of inputs from other nodes
            name: Optional unique identifier for this node
            cache_policy: Optional cache policy for this node

        """
        super().__init__(inputs, name, cache_policy=cache_policy)
        self.predicate = predicate
        self.if_true = if_true
        self.if_false = if_false

    @property
    def dependencies(self) -> list[Any]:
        """Get the list of nodes this node depends on."""
        deps = []
        if self._inputs:
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

        # Import ToolResult for result capture
        from pydantic_flow.streaming.tool_events import ToolResult  # noqa: PLC0415

        # Stream from the chosen branch
        result = None
        async for item in chosen_branch.astream(input_data):
            # Forward all items from the branch, but don't forward its StreamEnd
            if not isinstance(item, StreamEnd):
                yield item
                # Capture result from ToolResult if available
                if isinstance(item, ToolResult) and item.result is not None:
                    result = item.result
            else:
                # Save the result from the branch's StreamEnd
                result = item.result

        # Yield ToolResult with actual result if we have it
        if result is not None:
            yield ToolResult(
                run_id=run_id,
                node_id=node_id,
                tool_name="conditional",
                result=result,
            )

        # Emit our own StreamEnd with the branch's result
        yield StreamEnd(run_id=run_id, node_id=node_id, result=result)
