"""Flow orchestration for the pydantic-flow framework.

This module provides the Flow class that manages workflow execution,
dependency resolution, and DAG validation.
"""

from collections import deque
from collections.abc import Callable
from enum import StrEnum
from typing import Any
from typing import TypeVar

from pydantic import BaseModel

from pydantic_flow.core.errors import FlowError
from pydantic_flow.core.routing import T_Route
from pydantic_flow.core.run_config import RunConfig
from pydantic_flow.engine.stepper import ConditionalEdge
from pydantic_flow.engine.stepper import EngineConfig
from pydantic_flow.engine.stepper import StepperEngine
from pydantic_flow.flow.exceptions import CyclicDependencyError
from pydantic_flow.nodes import BaseNode
from pydantic_flow.nodes.protocols import has_input_dependency
from pydantic_flow.nodes.protocols import has_multiple_inputs

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class ExecutionMode(StrEnum):
    """Execution engine selection for flow compilation.

    Attributes:
        AUTO: Automatically detect based on flow structure (cycles, conditional edges).
        DAG: Use legacy topological sort execution (no cycles or conditional edges).
        STEPPER: Use loop-capable stepper engine (supports cycles and routing).

    """

    AUTO = "auto"
    DAG = "dag"
    STEPPER = "stepper"


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
        self._edges: dict[str, list[str]] = {}
        self._conditional_edges: list[ConditionalEdge[Any]] = []
        self._entry_nodes: list[str] | None = None

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
                # Determine input data for the node
                if has_multiple_inputs(node):
                    # Multi-input node: gather all dependency results as tuple
                    input_data: Any = tuple(
                        self._results[dep.node.name] for dep in node.inputs
                    )
                elif has_input_dependency(node):
                    # Single-input node: node takes input from another node
                    input_node = node.input.node
                    if input_node.name not in self._results:
                        msg = f"Input node {input_node.name} has not been executed"
                        raise FlowError(msg)
                    input_data = self._results[input_node.name]
                else:
                    # No-input node: takes input from flow inputs
                    input_data = inputs

                # Execute the node (node is still BaseNode regardless of type guard)
                result = await node.run(input_data)  # type: ignore[union-attr]
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

    def add_edge(self, from_node: str, to_node: str) -> None:
        """Add a static edge between two nodes.

        Args:
            from_node: Source node name.
            to_node: Target node name.

        """
        if from_node not in self._edges:
            self._edges[from_node] = []
        self._edges[from_node].append(to_node)

    def add_conditional_edges(
        self,
        from_node: str,
        router: Callable[[BaseModel], T_Route | list[T_Route]],
        mapping: dict[Any, str] | None = None,
    ) -> None:
        """Add conditional routing edges from a node.

        Args:
            from_node: Source node name.
            router: Function that receives state and returns routing target(s).
            mapping: Optional dict to map router outputs to node names.

        """
        edge = ConditionalEdge(from_node, router, mapping)
        self._conditional_edges.append(edge)

    def set_entry_nodes(self, *node_names: str) -> None:
        """Set the entry nodes for loop-capable execution.

        Args:
            *node_names: Names of nodes to execute first.

        Raises:
            ValueError: If no nodes specified or if any node name doesn't exist.

        """
        if not node_names:
            msg = "Must specify at least one entry node"
            raise ValueError(msg)

        existing_names = {node.name for node in self.nodes}
        unknown = set(node_names) - existing_names

        if unknown:
            msg = (
                f"Unknown entry nodes: {sorted(unknown)}. "
                f"Available nodes: {sorted(existing_names)}"
            )
            raise ValueError(msg)

        self._entry_nodes = list(node_names)

    def compile(
        self, *, mode: ExecutionMode = ExecutionMode.AUTO
    ) -> CompiledFlow[InputT, OutputT]:
        """Compile the flow into an executable form.

        Args:
            mode: Execution engine to use. AUTO detects based on flow structure,
                  DAG uses topological sort, STEPPER uses loop-capable engine.

        Returns:
            CompiledFlow instance ready for execution.

        Raises:
            FlowError: If flow structure is invalid or incompatible with mode.

        """
        # Determine which engine to use
        if mode == ExecutionMode.AUTO:
            use_stepper = self._should_use_stepper()
        elif mode == ExecutionMode.STEPPER:
            use_stepper = True
        else:  # ExecutionMode.DAG
            use_stepper = False
            # Validate no cycles or conditional edges for DAG mode
            if self._conditional_edges:
                msg = (
                    "Cannot use DAG mode with conditional edges. "
                    "Use ExecutionMode.STEPPER or ExecutionMode.AUTO."
                )
                raise FlowError(msg)
            if self._detect_cycles_efficiently():
                msg = (
                    "Cannot use DAG mode with cyclic dependencies. "
                    "Use ExecutionMode.STEPPER or ExecutionMode.AUTO."
                )
                raise FlowError(msg)

        if use_stepper:
            entry_nodes = self._entry_nodes or self._infer_entry_nodes()
            engine_config = EngineConfig(
                nodes=self.nodes,
                edges=self._edges,
                conditional_edges=self._conditional_edges,
                entry_nodes=entry_nodes,
                input_type=self._input_type,
                output_type=self._output_type,
            )
            engine = StepperEngine(engine_config)
            return CompiledFlow(engine=engine, use_stepper=True)

        self._calculate_execution_order()
        return CompiledFlow(flow=self, use_stepper=False)

    def _has_cycles(self) -> bool:
        """Check if the flow has cycles."""
        try:
            self._calculate_execution_order()
            return False
        except CyclicDependencyError:
            return True

    def _should_use_stepper(self) -> bool:
        """Determine if stepper engine is needed based on flow structure.

        Returns:
            True if stepper engine should be used, False for DAG execution.

        """
        # Has conditional edges -> need stepper
        if self._conditional_edges:
            return True

        # Check for cycles
        return self._detect_cycles_efficiently()

    def _detect_cycles_efficiently(self) -> bool:
        """Detect cycles using DFS with color marking (no exceptions).

        Uses three-color marking:
        - WHITE (0): unvisited
        - GRAY (1): currently being processed (on stack)
        - BLACK (2): finished processing

        Returns:
            True if cycles detected, False otherwise.

        """
        # Build adjacency list from explicit edges
        adj: dict[str, list[str]] = {}
        for node in self.nodes:
            adj[node.name] = self._edges.get(node.name, [])

        # Add implicit edges from node dependencies
        for node in self.nodes:
            if has_input_dependency(node):
                input_node_name = node.input.node.name
                if input_node_name not in adj:
                    adj[input_node_name] = []
                if node.name not in adj[input_node_name]:
                    adj[input_node_name].append(node.name)

            # Handle multi-input nodes
            if has_multiple_inputs(node):
                for dep in node.inputs:
                    dep_node_name = dep.node.name
                    if dep_node_name not in adj:
                        adj[dep_node_name] = []
                    if node.name not in adj[dep_node_name]:
                        adj[dep_node_name].append(node.name)

        # Three-color DFS
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {node.name: WHITE for node in self.nodes}

        def has_cycle_from(node_name: str) -> bool:
            """DFS helper to detect cycles from a starting node."""
            color[node_name] = GRAY
            for neighbor in adj.get(node_name, []):
                if color[neighbor] == GRAY:
                    # Back edge found - cycle detected
                    return True
                if color[neighbor] == WHITE and has_cycle_from(neighbor):
                    return True
            color[node_name] = BLACK
            return False

        # Check all nodes (handles disconnected components)
        for node in self.nodes:
            if color[node.name] == WHITE and has_cycle_from(node.name):
                return True

        return False

    def _infer_entry_nodes(self) -> list[str]:
        """Infer entry nodes from nodes with no dependencies."""
        entry = []
        for node in self.nodes:
            deps = getattr(node, "dependencies", [])
            if not deps:
                entry.append(node.name)
        return entry if entry else [self.nodes[0].name] if self.nodes else []

    def __repr__(self) -> str:
        """Return a string representation of the flow."""
        node_count = len(self.nodes)

        # Get input and output type names
        input_type_name = getattr(self._input_type, "__name__", str(self._input_type))
        output_type_name = getattr(
            self._output_type, "__name__", str(self._output_type)
        )

        return f"Flow[{input_type_name}, {output_type_name}](nodes={node_count})"


class CompiledFlow[InputT: BaseModel, OutputT: BaseModel]:
    """Compiled flow ready for execution.

    This class wraps either the legacy DAG runner or the new stepper engine.
    """

    def __init__(
        self,
        flow: Flow[InputT, OutputT] | None = None,
        engine: StepperEngine[InputT, OutputT] | None = None,
        use_stepper: bool = False,
    ) -> None:
        """Initialize compiled flow.

        Args:
            flow: Legacy flow for DAG execution.
            engine: Stepper engine for loop-capable execution.
            use_stepper: Whether to use the stepper engine.

        """
        self.flow = flow
        self.engine = engine
        self.use_stepper = use_stepper

    async def invoke(self, inputs: InputT, config: RunConfig | None = None) -> OutputT:
        """Execute the compiled flow.

        Args:
            inputs: Input data.
            config: Optional execution configuration (only for stepper).

        Returns:
            Flow output.

        """
        if self.use_stepper and self.engine is not None:
            return await self.engine.invoke(inputs, config)
        if self.flow is not None:
            return await self.flow.run(inputs)
        msg = "CompiledFlow has neither flow nor engine configured"
        raise FlowError(msg)
