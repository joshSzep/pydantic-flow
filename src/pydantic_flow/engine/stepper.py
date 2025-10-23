"""Stepper-based execution engine for loop-capable flows.

This module provides a frontier-based execution engine that supports cycles,
conditional routing, and dynamic control flow.
"""

import asyncio
from collections.abc import Callable
import time
from typing import Any
from typing import cast

from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator

from pydantic_flow.core.errors import FlowError
from pydantic_flow.core.errors import FlowTimeoutError
from pydantic_flow.core.errors import RecursionLimitError
from pydantic_flow.core.errors import RoutingError
from pydantic_flow.core.routing import Route
from pydantic_flow.core.routing import T_Route
from pydantic_flow.core.run_config import RunConfig
from pydantic_flow.nodes import BaseNode
from pydantic_flow.nodes.protocols import NodeWithInput
from pydantic_flow.nodes.protocols import NodeWithInputs
from pydantic_flow.nodes.protocols import has_input_dependency
from pydantic_flow.nodes.protocols import has_multiple_inputs


class IterationEvent(BaseModel):
    """Event emitted for each execution superstep.

    Attributes:
        iteration: The superstep number (0-indexed).
        frontier: List of node names executed in this iteration.
        routed_to: List of node names selected for the next iteration.
        ended: Whether the flow terminated with Route.END.
        elapsed_ms: Milliseconds elapsed since flow start.

    """

    iteration: int
    frontier: list[str]
    routed_to: list[str]
    ended: bool
    elapsed_ms: float


class ConditionalEdge[StateT: BaseModel]:
    """Represents a conditional routing edge.

    Attributes:
        from_node: The source node name.
        router: Function that takes state and returns routing outcome(s).
            Can be a typed RouterFunction[StateT] for better IDE support.
        mapping: Optional dict to map router output to target node names.

    """

    def __init__(
        self,
        from_node: str,
        router: Callable[[StateT], T_Route | list[T_Route]],
        mapping: dict[Any, str] | None = None,
    ) -> None:
        """Initialize a conditional edge.

        Args:
            from_node: Source node name.
            router: Callable that returns routing targets. For best type safety,
                use a function with explicit state type annotation.
            mapping: Optional mapping from router return values to node names.

        """
        self.from_node = from_node
        self.router = router
        self.mapping = mapping


class EngineConfig[InputT: BaseModel, OutputT: BaseModel](BaseModel):
    """Configuration for stepper engine initialization.

    Attributes:
        nodes: List of all nodes in the flow.
        edges: Static edge mapping from node name to target node names.
        conditional_edges: List of conditional routing edges.
        entry_nodes: Names of nodes to execute first.
        input_type: Expected input type for the flow.
        output_type: Expected output type for the flow.

    """

    nodes: list[BaseNode[Any, Any]]
    edges: dict[str, list[str]] = Field(default_factory=dict)
    conditional_edges: list[ConditionalEdge[Any]] = Field(default_factory=list)
    entry_nodes: list[str]
    input_type: type[InputT]
    output_type: type[OutputT]

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def validate_references(self) -> EngineConfig[InputT, OutputT]:
        """Validate that all edge references point to existing nodes."""
        node_names = {node.name for node in self.nodes}

        unknown_entry = set(self.entry_nodes) - node_names
        if unknown_entry:
            msg = f"Unknown entry nodes: {sorted(unknown_entry)}"
            raise ValueError(msg)

        for from_node, targets in self.edges.items():
            unknown_targets = set(targets) - node_names
            if unknown_targets:
                msg = (
                    f"Unknown edge targets from '{from_node}': "
                    f"{sorted(unknown_targets)}"
                )
                raise ValueError(msg)

        return self


class StepperEngine[InputT: BaseModel, OutputT: BaseModel]:
    """Loop-capable execution engine using frontier-based stepping.

    This engine supports cycles and conditional routing by executing nodes
    in supersteps and dynamically determining the next frontier based on
    edge configuration and router outputs.
    """

    def __init__(self, config: EngineConfig[InputT, OutputT]) -> None:
        """Initialize the stepper engine.

        Args:
            config: Engine configuration containing nodes, edges, and types.

        """
        self.nodes_by_name = {node.name: node for node in config.nodes}
        self.edges = config.edges
        self.conditional_edges = config.conditional_edges
        self.entry_nodes = config.entry_nodes
        self.input_type = config.input_type
        self.output_type = config.output_type

    async def invoke(
        self,
        inputs: InputT,
        config: RunConfig | None = None,
    ) -> OutputT:
        """Execute the flow with the given inputs.

        Args:
            inputs: Input data matching InputT.
            config: Optional execution configuration.

        Returns:
            Output data matching OutputT.

        Raises:
            RecursionLimitError: If max_steps is exceeded.
            FlowTimeoutError: If timeout_seconds is exceeded.
            RoutingError: If routing targets are invalid.
            FlowError: For other execution errors.

        """
        if config is None:
            config = RunConfig()

        self._validate_input_type(inputs)

        start_time = time.time()
        results: dict[str, Any] = {}
        frontier = set(self.entry_nodes)
        iteration = 0
        events: list[IterationEvent] = []

        try:
            while frontier:
                self._check_limits(iteration, config, start_time, events)

                current_frontier = list(frontier)
                frontier = set()

                await self._execute_frontier(current_frontier, inputs, results)
                next_frontier, ended = await self._route_next(current_frontier, results)

                elapsed_ms = (time.time() - start_time) * 1000
                if config.trace_iterations:
                    event = IterationEvent(
                        iteration=iteration,
                        frontier=current_frontier,
                        routed_to=list(next_frontier),
                        ended=ended,
                        elapsed_ms=elapsed_ms,
                    )
                    events.append(event)

                if ended:
                    break

                frontier = next_frontier
                iteration += 1

            return self.output_type(**results)

        except FlowError:
            raise
        except Exception as e:
            msg = f"Flow execution failed: {e}"
            raise FlowError(msg) from e

    def _validate_input_type(self, inputs: InputT) -> None:
        """Validate input type matches expected type."""
        if not isinstance(inputs, self.input_type):
            expected_name = self.input_type.__name__
            actual_name = type(inputs).__name__
            msg = f"Input type mismatch: expected {expected_name}, got {actual_name}"
            raise TypeError(msg)

    def _check_limits(
        self,
        iteration: int,
        config: RunConfig,
        start_time: float,
        events: list[IterationEvent],
    ) -> None:
        """Check recursion and timeout limits."""
        if iteration >= config.max_steps:
            recent_count = config.recent_events_count
            recent = events[-recent_count:] if len(events) >= recent_count else events
            msg = (
                f"Exceeded max_steps={config.max_steps} "
                f"at iteration {iteration}. "
                f"Recent iterations: {[e.model_dump() for e in recent]}"
            )
            raise RecursionLimitError(msg)

        if config.timeout_seconds is not None:
            elapsed = time.time() - start_time
            if elapsed > config.timeout_seconds:
                msg = (
                    f"Exceeded timeout of {config.timeout_seconds}s "
                    f"at iteration {iteration}"
                )
                raise FlowTimeoutError(msg)

    async def _execute_frontier(
        self,
        frontier: list[str],
        inputs: InputT,
        results: dict[str, Any],
    ) -> None:
        """Execute all nodes in the current frontier in parallel."""

        async def execute_node(node_name: str) -> tuple[str, Any]:
            """Execute a single node and return its name and result."""
            node = self.nodes_by_name[node_name]
            input_data = self._get_node_input(node, inputs, results)
            result = await node.run(input_data)
            return node_name, result

        tasks = [execute_node(node_name) for node_name in frontier]
        node_results = await asyncio.gather(*tasks)

        for node_name, result in node_results:
            results[node_name] = result

    def _get_node_input(
        self,
        node: BaseNode[Any, Any],
        inputs: InputT,
        results: dict[str, Any],
    ) -> Any:
        """Determine input data for a node.

        Uses type guards to check node patterns instead of getattr().
        """
        # Check for multi-input nodes (e.g., MergeNode)
        if has_multiple_inputs(node):
            node_with_inputs = cast(NodeWithInputs, node)
            return tuple(results[dep.node.name] for dep in node_with_inputs.inputs)

        # Check for single-input nodes
        if has_input_dependency(node):
            node_with_input = cast(NodeWithInput, node)
            input_node = node_with_input.input.node
            if input_node.name not in results:
                msg = f"Input node {input_node.name} has not been executed"
                raise FlowError(msg)
            return results[input_node.name]

        # For nodes with no explicit dependencies, check if they have a previous result
        # (for loops). If so, use that. Otherwise use flow inputs.
        if node.name in results:
            return results[node.name]

        return inputs

    async def _route_next(
        self,
        current_frontier: list[str],
        results: dict[str, Any],
    ) -> tuple[set[str], bool]:
        """Route to next frontier based on edges and conditional routers."""
        next_frontier: set[str] = set()
        ended = False

        for node_name in current_frontier:
            static_targets = self.edges.get(node_name, [])
            next_frontier.update(static_targets)

            for cond_edge in self.conditional_edges:
                if cond_edge.from_node == node_name:
                    targets, edge_ended = self._apply_conditional_edge(
                        cond_edge, results
                    )
                    next_frontier.update(targets)
                    ended = ended or edge_ended

        return next_frontier, ended

    def _apply_conditional_edge(
        self,
        cond_edge: ConditionalEdge[Any],
        results: dict[str, Any],
    ) -> tuple[set[str], bool]:
        """Apply a conditional edge and return targets and ended status."""
        state = self._build_state(results)
        raw_outcome = cond_edge.router(state)

        outcomes = raw_outcome if isinstance(raw_outcome, list) else [raw_outcome]

        targets: set[str] = set()
        ended = False

        for outcome in outcomes:
            target, outcome_ended = self._resolve_outcome(outcome, cond_edge)
            if target is not None:
                targets.add(target)
            ended = ended or outcome_ended

        return targets, ended

    def _resolve_outcome(
        self,
        outcome: T_Route,
        cond_edge: ConditionalEdge[Any],
    ) -> tuple[str | None, bool]:
        """Resolve a single routing outcome to target and ended status."""
        if cond_edge.mapping is not None:
            if outcome not in cond_edge.mapping:
                msg = (
                    f"Router outcome {outcome!r} not in mapping: "
                    f"{list(cond_edge.mapping.keys())}"
                )
                raise RoutingError(msg)
            target = cond_edge.mapping[outcome]
        else:
            target = outcome

        if target == Route.END:
            return None, True
        if isinstance(target, str):
            if target not in self.nodes_by_name:
                msg = f"Router target {target!r} is not a valid node name"
                raise RoutingError(msg)
            return target, False

        msg = f"Invalid routing target: {target!r}"
        raise RoutingError(msg)

    def _build_state(self, results: dict[str, Any]) -> BaseModel:
        """Build a state model from results."""
        state_dict = dict(results)
        annotations = {k: type(v) for k, v in state_dict.items()}
        state_model = type("State", (BaseModel,), {"__annotations__": annotations})
        return state_model(**state_dict)
