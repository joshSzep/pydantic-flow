"""FlowNode implementation for composable sub-flows."""

from typing import TYPE_CHECKING
from typing import Any

from pydantic import BaseModel

from pflow.nodes.base import BaseNode
from pflow.nodes.base import NodeOutput
from pflow.nodes.base import NodeWithInput

if TYPE_CHECKING:
    from pflow.flow.flow import Flow


class FlowNode[InputModel: BaseModel, OutputModel: BaseModel](
    NodeWithInput[InputModel, OutputModel]
):
    """A node that wraps a Flow, enabling sub-flows within larger workflows.

    This node allows for hierarchical composition of flows, where a complete
    Flow can be used as a single node within another Flow. This enables
    building complex workflows from simpler, reusable sub-flows.
    """

    def __init__(
        self,
        flow: Flow[InputModel, OutputModel],
        *,
        input: NodeOutput[InputModel] | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize a FlowNode with a wrapped Flow.

        Args:
            flow: The Flow to wrap as a node. The flow's input and output types
                 must match the FlowNode's type parameters.
            input: Optional input from another node's output
            name: Optional unique identifier for this node. If not provided,
                 will use the format "FlowNode_{flow_repr}"

        """
        # Generate a meaningful default name that includes the wrapped flow info
        if name is None:
            flow_repr = repr(flow)
            name = f"FlowNode_{flow_repr}"

        super().__init__(input, name)
        self.flow = flow

    async def run(self, input_data: InputModel) -> OutputModel:
        """Execute the wrapped flow with the given input.

        Args:
            input_data: The input data for the wrapped flow

        Returns:
            The output from the wrapped flow execution

        Raises:
            FlowError: If the wrapped flow execution fails
            TypeError: If input_data doesn't match the flow's expected input type

        """
        return await self.flow.run(input_data)

    @property
    def dependencies(self) -> list[BaseNode[Any, Any]]:
        """Get the list of nodes this FlowNode depends on.

        Returns:
            List containing the input node if this FlowNode takes input from
            another node, otherwise an empty list.

        Note:
            The wrapped flow's internal dependencies are not exposed here
            since they are encapsulated within the flow execution.

        """
        return super().dependencies

    def __repr__(self) -> str:
        """Return a string representation of the FlowNode."""
        return f"FlowNode(name='{self.name}', flow={self.flow!r})"
