"""Flow orchestration for the pydantic-flow framework.

This module provides the Flow class that manages workflow execution,
dependency resolution, and DAG validation.
"""

from collections import deque
from typing import Any
from typing import TypeVar

from pydantic import BaseModel

from pydantic_flow.flow.exceptions import CyclicDependencyError
from pydantic_flow.flow.exceptions import FlowError
from pydantic_flow.nodes import BaseNode

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class Flow[InputT: BaseModel, OutputT: BaseModel]:
    """A workflow orchestrator that manages node execution and dependencies.

    The Flow class provides DAG validation, dependency resolution, and
    execution coordination for workflows built from connected nodes.

    Type Parameters:
        InputT: The input type for the flow, must be a BaseModel subclass
        OutputT: The output type for the flow, must be a BaseModel subclass
    """

    def __init__(self, *, input_type: type[InputT], output_type: type[OutputT]) -> None:
        """Initialize a flow with the required input and output types.

        Args:
            input_type: The BaseModel class that this flow accepts as input.
            output_type: The BaseModel class to construct from flow results.

        """
        self.nodes: list[BaseNode[Any, Any]] = []
        self._execution_order: list[BaseNode[Any, Any]] = []
        self._results: dict[str, Any] = {}
        self._input_type = input_type
        self._output_type = output_type

    def add_nodes(self, *nodes: BaseNode[InputT | Any, Any]) -> None:
        """Add one or more nodes to the flow.

        Args:
            *nodes: Variable number of nodes to add to the flow.
                   Nodes that take direct flow input should accept InputT,
                   others can accept Any (from other nodes).

        """
        for node in nodes:
            if node not in self.nodes:
                self.nodes.append(node)
        # Recalculate execution order when nodes are added
        self._calculate_execution_order()

    def _calculate_execution_order(self) -> None:
        """Calculate the execution order using topological sorting.

        Raises:
            CyclicDependencyError: If a cycle is detected in the dependencies

        """
        # Build dependency graph
        in_degree = dict.fromkeys(self.nodes, 0)
        adjacency = {node: [] for node in self.nodes}

        for node in self.nodes:
            for dep in getattr(node, "dependencies", []):
                if dep in self.nodes:
                    adjacency[dep].append(node)
                    in_degree[node] += 1

        # Kahn's algorithm for topological sorting
        queue = deque([node for node in self.nodes if in_degree[node] == 0])
        execution_order = []

        while queue:
            current = queue.popleft()
            execution_order.append(current)

            for neighbor in adjacency[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Check for cycles
        if len(execution_order) != len(self.nodes):
            msg = "Cyclic dependency detected in the flow"
            raise CyclicDependencyError(msg)

        self._execution_order = execution_order

    async def run(self, inputs: InputT) -> OutputT:
        """Execute the flow with the given inputs.

        Args:
            inputs: The input data for the flow (must match the flow's InputT type)

        Returns:
            The flow results with the specified OutputT type

        Raises:
            FlowError: If the flow execution fails
            TypeError: If the input type doesn't match the expected input_type

        """
        # Runtime validation of input type
        if not isinstance(inputs, self._input_type):
            expected_name = self._input_type.__name__
            actual_name = type(inputs).__name__
            msg = f"Input type mismatch: expected {expected_name}, got {actual_name}"
            raise TypeError(msg)

        self._results = {}

        try:
            for node in self._execution_order:
                # Check if multi-input node (MergeNode pattern)
                node_inputs = getattr(node, "inputs", None)
                if node_inputs is not None and isinstance(node_inputs, tuple):
                    # Multi-input node: gather all dependency results as tuple
                    input_data = tuple(
                        self._results[dep.node.name] for dep in node_inputs
                    )
                else:
                    # Single-input or no-input node (existing behavior)
                    node_input = getattr(node, "input", None)
                    if node_input is not None:
                        # Node takes input from another node
                        input_node = node_input.node
                        if input_node.name not in self._results:
                            msg = f"Input node {input_node.name} has not been executed"
                            raise FlowError(msg)
                        input_data = self._results[input_node.name]
                    else:
                        # Node takes input from flow inputs
                        input_data = inputs

                # Execute the node
                result = await node.run(input_data)
                self._results[node.name] = result

            # Construct the output BaseModel from the results
            return self._output_type(**self._results)

        except Exception as e:
            # Wrap any other exception in a FlowError for consistency
            if isinstance(e, FlowError):
                raise
            msg = f"Flow execution failed: {e}"
            raise FlowError(msg) from e

    def get_execution_order(self) -> list[str]:
        """Get the names of nodes in execution order.

        Returns:
            List of node names in the order they will be executed

        """
        return [node.name for node in self._execution_order]

    def validate(self) -> bool:
        """Validate the flow structure.

        Returns:
            True if the flow is valid

        Raises:
            CyclicDependencyError: If cycles are detected
            FlowError: If other validation errors are found

        """
        try:
            self._calculate_execution_order()
            return True
        except CyclicDependencyError:
            raise
        except Exception as e:
            msg = f"Flow validation failed: {e}"
            raise FlowError(msg) from e

    def __repr__(self) -> str:
        """Return a string representation of the flow."""
        node_count = len(self.nodes)

        # Get input and output type names
        input_type_name = getattr(self._input_type, "__name__", str(self._input_type))
        output_type_name = getattr(
            self._output_type, "__name__", str(self._output_type)
        )

        return f"Flow[{input_type_name}, {output_type_name}](nodes={node_count})"
