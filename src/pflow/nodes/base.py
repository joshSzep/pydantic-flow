"""Core node abstractions for the pflow framework.

This module provides the foundational building blocks for creating type-safe,
composable AI workflows using Pydantic models.
"""

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any
from typing import Protocol


@dataclass(frozen=True)
class NodeOutput[OutputT]:
    """Represents a typed output reference from a node.

    This class enables type-safe wiring between nodes by providing
    a strongly-typed reference to another node's output.
    """

    node: BaseNode[Any, OutputT]

    @property
    def type_hint(self) -> type[OutputT]:
        """Get the output type hint for this node output."""
        return self.node._output_type


class BaseNode[InputT, OutputT](ABC):
    """Abstract base class for all workflow nodes.

    Provides the core interface that all nodes must implement, including
    type-safe input/output handling and unique naming.
    """

    def __init__(self, name: str | None = None) -> None:
        """Initialize the base node.

        Args:
            name: Optional unique identifier for this node. If not provided,
                  will be auto-generated based on the class name.

        """
        self.name = name or f"{self.__class__.__name__}_{id(self):x}"
        self._output: NodeOutput[OutputT] = NodeOutput(self)
        # Store type information for runtime inspection
        self._input_type: type[InputT] = self.__class__.__orig_bases__[0].__args__[0]  # type: ignore
        self._output_type: type[OutputT] = self.__class__.__orig_bases__[0].__args__[1]  # type: ignore

    @property
    def output(self) -> NodeOutput[OutputT]:
        """Get the typed output reference for this node."""
        return self._output

    @abstractmethod
    async def run(self, input_data: InputT) -> OutputT:
        """Execute the node's logic with the given input.

        Args:
            input_data: The input data for this node

        Returns:
            The processed output data

        """

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
    ) -> None:
        """Initialize a node with optional input dependency.

        Args:
            input: Optional input from another node's output
            name: Optional unique identifier for this node

        """
        super().__init__(name)
        self.input = input

    @property
    def dependencies(self) -> list[BaseNode[Any, Any]]:
        """Get the list of nodes this node depends on."""
        if self.input is None:
            return []
        return [self.input.node]


# Protocol classes for type safety
class NodeProtocol[InputT, OutputT](Protocol):
    """Protocol defining the interface all nodes must implement."""

    name: str
    output: NodeOutput[OutputT]

    async def run(self, input_data: InputT) -> OutputT:
        """Execute the node's logic."""
        ...


class RunnableNode[InputT, OutputT](Protocol):
    """Protocol for nodes that can be executed."""

    async def run(self, input_data: InputT) -> OutputT:
        """Execute the node's logic."""
        ...
