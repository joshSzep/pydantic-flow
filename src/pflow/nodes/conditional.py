"""IfNode implementation for conditional branching."""

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from pflow.nodes.base import NodeOutput
from pflow.nodes.base import NodeWithInput


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

    async def run(self, input_data: Any) -> OutputModel:
        """Execute the appropriate branch based on the predicate.

        Args:
            input_data: The input data for this node

        Returns:
            The output from the chosen branch

        """
        if self.predicate(input_data):
            return await self.if_true.run(input_data)
        else:
            return await self.if_false.run(input_data)
