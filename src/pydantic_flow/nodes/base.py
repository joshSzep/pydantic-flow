"""Core node abstractions for the pydantic-flow framework.

This module provides the foundational building blocks for creating type-safe,
composable AI workflows using Pydantic models with streaming-native execution.
"""

from abc import ABC
from abc import abstractmethod
from collections.abc import AsyncIterator
from typing import Any
from typing import Protocol
from typing import cast

from pydantic import BaseModel

from pydantic_flow.streaming.events import ProgressItem
from pydantic_flow.streaming.events import StreamEnd
from pydantic_flow.streaming.events import StreamStart
from pydantic_flow.streaming.events import ToolResult


class NodeOutput[OutputT](BaseModel):
    """Represents a typed output reference from a node.

    This class enables type-safe wiring between nodes by providing
    a strongly-typed reference to another node's output.
    """

    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    node: BaseNode[Any, OutputT]

    @property
    def type_hint(self) -> type[OutputT]:
        """Get the output type hint for this node output."""
        return self.node._output_type


class BaseNode[InputT, OutputT](ABC):
    """Abstract base class for all workflow nodes.

    Nodes are streaming-native: the primary interface is astream() which
    yields progress items, with run() as a convenience wrapper that assembles
    the final result.
    """

    def __init__(self, name: str | None = None, run_id: str | None = None) -> None:
        """Initialize the base node.

        Args:
            name: Optional unique identifier for this node. If not provided,
                  will be auto-generated based on the class name.
            run_id: Optional run identifier for tracking execution.

        """
        self.name = name or f"{self.__class__.__name__}_{id(self):x}"
        self.run_id = run_id
        self._output: NodeOutput[OutputT] = NodeOutput(node=self)
        # Store type information for runtime inspection
        self._input_type: type[InputT] = self.__class__.__orig_bases__[0].__args__[0]  # type: ignore
        self._output_type: type[OutputT] = self.__class__.__orig_bases__[0].__args__[1]  # type: ignore

    @property
    def output(self) -> NodeOutput[OutputT]:
        """Get the typed output reference for this node."""
        return self._output

    @abstractmethod
    async def astream(self, input_data: InputT) -> AsyncIterator[ProgressItem]:
        """Stream progress items while executing the node's logic.

        This is the primary interface for node execution. It yields a
        coherent sequence: start, useful progress, clean end.

        Args:
            input_data: The input data for this node

        Yields:
            Progress items representing execution progress.

        """
        # Emit start marker
        yield StreamStart(
            run_id=self.run_id or "",
            node_id=self.name,
            input_preview=self._preview_input(input_data),
        )

        # Subclass implements actual streaming logic here
        yield  # type: ignore

        # Emit end marker (subclass should do this)
        yield StreamEnd(
            run_id=self.run_id or "",
            node_id=self.name,
        )

    async def run(self, input_data: InputT) -> OutputT:
        """Execute the node and return the final validated result.

        This is a convenience method that consumes the astream() and
        assembles the final output.

        Args:
            input_data: The input data for this node

        Returns:
            The final validated output data

        """
        final_result: OutputT | None = None
        tool_result: Any = None

        async for item in self.astream(input_data):
            # Try to extract result from ToolResult first (has the actual object)
            if isinstance(item, ToolResult) and item.result is not None:
                tool_result = item.result
            # StreamEnd carries the final result preview as fallback
            elif isinstance(item, StreamEnd) and item.result_preview:
                # Reconstruct the output from the preview
                # Try Pydantic validation first
                try:
                    if hasattr(self._output_type, "model_validate"):
                        final_result = self._output_type.model_validate(  # type: ignore
                            item.result_preview
                        )
                    else:
                        final_result = item.result_preview  # type: ignore
                except Exception:
                    # Fall back to direct assignment
                    final_result = item.result_preview  # type: ignore

        # Prefer the actual result from ToolResult if available
        if tool_result is not None:
            final_result = tool_result  # type: ignore

        if final_result is None:
            msg = f"Node {self.name} did not produce a result"
            raise RuntimeError(msg)

        return final_result

    def _preview_input(self, input_data: InputT) -> dict[str, Any] | None:
        """Create a preview dict of input data for progress events.

        Args:
            input_data: The input data to preview.

        Returns:
            Dict preview or None if preview cannot be created.

        """
        if hasattr(input_data, "model_dump"):
            return input_data.model_dump()  # type: ignore
        if isinstance(input_data, dict):
            return cast(dict[str, Any], input_data)
        return {"value": str(input_data)[:100]}

    def __repr__(self) -> str:
        """Return a string representation of the node."""
        return f"{self.__class__.__name__}(name='{self.name}')"


class NodeWithInput[InputT, OutputT](BaseNode[InputT, OutputT]):
    """Base class for nodes that take input from other nodes.

    This class handles the common pattern of nodes that depend on
    the output of other nodes in the workflow.
    """

    def __init__(
        self,
        input: NodeOutput[InputT] | None = None,
        name: str | None = None,
        run_id: str | None = None,
    ) -> None:
        """Initialize a node with optional input dependency.

        Args:
            input: Optional input from another node's output
            name: Optional unique identifier for this node
            run_id: Optional run identifier for tracking execution

        """
        super().__init__(name, run_id)
        self.input = input

    @property
    def dependencies(self) -> list[BaseNode[Any, Any]]:
        """Get the list of nodes this node depends on."""
        if self.input is None:
            return []
        return [self.input.node]


class MergeNode[*InputTs, OutputT](BaseNode[tuple[*InputTs], OutputT]):
    """Base class for nodes that merge multiple inputs.

    This class enables fan-in patterns where a node needs to combine
    outputs from multiple upstream nodes.

    Uses PEP 646 TypeVarTuple for arbitrary input types, allowing
    full type safety across multiple inputs.

    Example:
        MergeNode[DataA, DataB, DataC, Result] represents a node that
        takes three inputs (DataA, DataB, DataC) and produces Result.

    """

    def __init__(
        self,
        inputs: tuple[NodeOutput[Any], ...],
        name: str | None = None,
        run_id: str | None = None,
    ) -> None:
        """Initialize a merge node with multiple input dependencies.

        Args:
            inputs: Tuple of NodeOutput references from upstream nodes
            name: Optional unique identifier for this node
            run_id: Optional run identifier for tracking execution

        """
        super().__init__(name, run_id)
        self.inputs = inputs

    @property
    def dependencies(self) -> list[BaseNode[Any, Any]]:
        """Get all dependency nodes from multiple inputs."""
        return [node_output.node for node_output in self.inputs]


# Protocol classes for type safety
class NodeProtocol[InputT, OutputT](Protocol):
    """Protocol defining the interface all nodes must implement."""

    name: str
    output: NodeOutput[OutputT]

    async def astream(self, input_data: InputT) -> AsyncIterator[ProgressItem]:
        """Stream progress items during execution."""
        ...

    async def run(self, input_data: InputT) -> OutputT:
        """Execute the node's logic and return final result."""
        ...


class RunnableNode[InputT, OutputT](Protocol):
    """Protocol for nodes that can be executed."""

    name: str

    async def astream(self, input_data: InputT) -> AsyncIterator[ProgressItem]:
        """Stream progress items during execution."""
        ...

    async def run(self, input_data: InputT) -> OutputT:
        """Execute the node's logic and return final result."""
        ...
