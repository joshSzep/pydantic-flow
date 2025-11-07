"""Dataflow execution engine for eager, dependency-driven node execution.

This module provides an execution engine that eliminates wave-based synchronization
barriers in favor of eager execution: nodes start as soon as their dependencies
are satisfied, maximizing parallelism and minimizing total execution time.
"""

import asyncio
from collections.abc import AsyncIterator
from collections.abc import Callable
import time
from typing import Any

from pydantic import BaseModel

from pydantic_flow.checkpoints import CheckpointConfig
from pydantic_flow.checkpoints import CheckpointManager
from pydantic_flow.checkpoints.types import RunId as CheckpointRunId
from pydantic_flow.checkpoints.types import RunMetadata
from pydantic_flow.core.errors import FlowError
from pydantic_flow.core.errors import FlowTimeoutError
from pydantic_flow.core.errors import RecursionLimitError
from pydantic_flow.core.errors import RoutingError
from pydantic_flow.core.routing import Route
from pydantic_flow.core.routing import T_Route
from pydantic_flow.core.run_config import RunConfig
from pydantic_flow.nodes import BaseNode
from pydantic_flow.streaming import ProgressItem
from pydantic_flow.streaming import StreamEnd
from pydantic_flow.streaming import ToolResult
from pydantic_flow.streaming.core_events import FlowResult
from pydantic_flow.streaming.core_events import GenericResult


class DataflowConfig[InputT: BaseModel, OutputT: BaseModel](BaseModel):
    """Configuration for dataflow engine initialization.

    Groups together parameters for cleaner __init__ signature.
    """

    model_config = {"arbitrary_types_allowed": True}

    nodes: list[BaseNode[Any, Any]]
    edges: dict[str, list[str]]
    conditional_edges: list[
        tuple[str, Callable[[BaseModel], Any], dict[Any, str] | None]
    ]
    entry_nodes: list[str]
    input_type: type[InputT]
    output_type: type[OutputT]
    cache_backend: Any = None
    default_cache_policy: Any = None
    flow_id: str | None = None


class NodeExecution(BaseModel):
    """Tracks the execution state of a single node.

    Attributes:
        node_name: Name of the node being executed.
        started_at: Timestamp when execution started (seconds since epoch).
        completed_at: Timestamp when execution completed, None if still running.
        result: The execution result, None if not yet complete.
        error: Exception if execution failed, None otherwise.

    """

    model_config = {"arbitrary_types_allowed": True}

    node_name: str
    started_at: float
    completed_at: float | None = None
    result: Any = None
    error: Exception | None = None

    @property
    def is_complete(self) -> bool:
        """Check if this execution has completed."""
        return self.completed_at is not None

    @property
    def duration_ms(self) -> float | None:
        """Calculate duration in milliseconds, or None if not complete."""
        if self.completed_at is None:
            return None
        return (self.completed_at - self.started_at) * 1000


class DataflowEngine[InputT: BaseModel, OutputT: BaseModel]:
    """Dataflow execution engine with eager, dependency-driven scheduling.

    This engine executes nodes as soon as their dependencies are satisfied,
    without artificial wave/frontier synchronization barriers. It automatically
    identifies and exploits all parallelism opportunities in the flow graph.

    Key characteristics:
    - Eager execution: nodes start immediately when dependencies are met
    - Optimal parallelism: independent nodes run concurrently
    - Real-time streaming: progress items emitted immediately
    - Smart error handling: failures cancel only dependent work
    - Type-safe: full generic type checking throughout

    """

    def __init__(self, config: DataflowConfig[InputT, OutputT]) -> None:
        """Initialize the dataflow engine.

        Args:
            config: Configuration containing all initialization parameters.

        """
        self.nodes_list = config.nodes
        self.edges = config.edges
        self.conditional_edges = config.conditional_edges
        self.entry_nodes = config.entry_nodes
        self.input_type = config.input_type
        self.output_type = config.output_type
        self.cache_backend = config.cache_backend
        self.default_cache_policy = config.default_cache_policy
        self.flow_id = config.flow_id

        # Build node lookup
        self.nodes_by_name: dict[str, BaseNode[Any, Any]] = {
            node.name: node for node in config.nodes
        }

        # Build reverse dependency graph (which nodes depend on each node)
        self.dependents: dict[str, set[str]] = {
            node.name: set() for node in config.nodes
        }
        for source, targets in config.edges.items():
            for target in targets:
                self.dependents[source].add(target)

        # Track conditional edge sources for routing
        self.conditional_sources: set[str] = {
            src for src, _, _ in config.conditional_edges
        }

    async def _initialize_checkpoint_manager(
        self, config: RunConfig, run_id: str
    ) -> CheckpointManager | None:
        """Initialize checkpoint manager if backend is configured."""
        backend = config.checkpoint_backend
        if backend is None:
            return None

        checkpoint_cfg = config.checkpoint_config or CheckpointConfig()
        checkpoint_manager = CheckpointManager(
            config=checkpoint_cfg,
            storage=backend,
            flow_id=self.flow_id or "default_flow",
            run_id=CheckpointRunId(run_id),
        )
        await checkpoint_manager.initialize_run()
        return checkpoint_manager

    def _check_execution_limits(
        self, step_count: int, start_time: float, config: RunConfig
    ) -> None:
        """Check if execution limits have been exceeded."""
        if config.max_steps and step_count >= config.max_steps:
            msg = f"Max steps ({config.max_steps}) exceeded"
            raise RecursionLimitError(msg)

        if config.timeout_seconds:
            elapsed = time.time() - start_time
            if elapsed > config.timeout_seconds:
                msg = f"Flow execution timeout ({config.timeout_seconds}s) exceeded"
                raise FlowTimeoutError(msg)

    def _determine_node_input(
        self,
        node: BaseNode[Any, Any],
        node_name: str,
        results: dict[str, Any],
        routing_sources: dict[str, str],
        entry_input: InputT,
    ) -> Any:
        """Determine input for a node based on dependencies and routing."""
        # Check if node has explicit input dependencies
        if node.dependencies:
            return self._get_node_input(node, results)

        # Check if this node was routed to conditionally
        if node_name in routing_sources:
            source_node = routing_sources[node_name]
            result = results[source_node]
            del routing_sources[node_name]
            # Unwrap GenericResult to get the actual value
            if isinstance(result, GenericResult):
                return result.value
            return result

        # Entry node - use flow input or previous result for self-loops
        if node_name in self.entry_nodes:
            return results.get(node_name, entry_input)

        # Self-loop: use previous result
        if node_name in results:
            return results[node_name]

        # Fallback: use flow input
        return entry_input

    async def _handle_checkpoint_after_node(
        self,
        checkpoint_manager: CheckpointManager | None,
        node_name: str,
        results: dict[str, Any],
        completed: set[str],
        running: set[str],
    ) -> tuple[set[str], bool]:
        """Handle checkpoint saving and routing after node completion.

        Returns:
            Tuple of (routed_nodes, flow_ended)

        """
        if checkpoint_manager is None:
            # No checkpoint manager, just apply routing
            return await self._apply_routing(node_name, results)

        # Determine next frontier based on static edges and routing
        next_frontier_nodes: set[str] = set()

        # Add nodes that become ready due to static edges
        for candidate_node in self.nodes_by_name:
            if (
                candidate_node not in completed
                and candidate_node not in running
                and self._are_dependencies_satisfied(candidate_node, completed)
            ):
                next_frontier_nodes.add(candidate_node)

        # Apply conditional routing to get routed nodes
        routed_nodes, ended = await self._apply_routing(node_name, results)
        next_frontier_nodes.update(routed_nodes)

        # Save checkpoint with current state and next frontier
        await checkpoint_manager.save_wave_checkpoint(
            current_state=results,
            next_frontier=list(next_frontier_nodes),
            routing_ended=ended,
        )
        return routed_nodes, ended

    async def _process_routing_queue(
        self,
        routing_queue: asyncio.Queue[str],
        completed: set[str],
        running: set[str],
        tasks: dict[str, asyncio.Task[None]],
        node_executor: Any,
    ) -> None:
        """Process the routing queue and schedule routed nodes."""
        while not routing_queue.empty():
            routed_node = await routing_queue.get()
            completed.discard(routed_node)
            if routed_node not in running and routed_node not in tasks:
                task = asyncio.create_task(node_executor(routed_node))
                tasks[routed_node] = task
                running.add(routed_node)

    async def _handle_completed_tasks(
        self, tasks: dict[str, asyncio.Task[None]]
    ) -> None:
        """Handle completed asyncio tasks and propagate exceptions."""
        done, _pending = await asyncio.wait(
            tasks.values(),
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in done:
            for name, t in list(tasks.items()):
                if t == task:
                    del tasks[name]
                    task.result()  # Raises if task failed
                    break

    async def _yield_queued_items(
        self, item_queue: asyncio.Queue[ProgressItem | None]
    ) -> AsyncIterator[ProgressItem]:
        """Yield all queued progress items."""
        while not item_queue.empty():
            item = await item_queue.get()
            if item is not None:
                yield item

    async def _finalize_flow(
        self,
        results: dict[str, Any],
        checkpoint_manager: CheckpointManager | None,
    ) -> FlowResult:
        """Finalize the flow and return the final result."""
        final_result = self._construct_output(results)

        if checkpoint_manager is not None:
            await checkpoint_manager.finalize_run(status=RunMetadata.Status.COMPLETED)

        return FlowResult(result=final_result)

    def _validate_input_type(self, inputs: InputT) -> None:
        """Validate that input matches expected type.

        Args:
            inputs: Input data to validate.

        Raises:
            TypeError: If input type doesn't match.

        """
        if not isinstance(inputs, self.input_type):
            msg = (
                f"Input type mismatch: expected {self.input_type.__name__}, "
                f"got {type(inputs).__name__}"
            )
            raise TypeError(msg)

    def _get_node_dependencies(self, node_name: str) -> set[str]:
        """Get the set of nodes that this node depends on.

        Args:
            node_name: Name of the node to check dependencies for.

        Returns:
            Set of node names that are dependencies.

        """
        dependencies: set[str] = set()

        # Check static edges - find nodes that point to this node
        for source, targets in self.edges.items():
            if node_name in targets:
                dependencies.add(source)

        return dependencies

    def _are_dependencies_satisfied(
        self,
        node_name: str,
        completed: set[str],
    ) -> bool:
        """Check if all dependencies for a node have been satisfied.

        Args:
            node_name: Name of the node to check.
            completed: Set of node names that have completed execution.

        Returns:
            True if all dependencies are satisfied, False otherwise.

        """
        dependencies = self._get_node_dependencies(node_name)
        return dependencies.issubset(completed)

    def _get_ready_nodes(
        self,
        completed: set[str],
        running: set[str],
        scheduled: set[str],
    ) -> list[str]:
        """Get nodes that are ready to execute (dependencies satisfied).

        Args:
            completed: Set of nodes that have completed execution.
            running: Set of nodes currently executing.
            scheduled: Set of nodes already scheduled for execution.

        Returns:
            List of node names ready to execute.

        """
        ready: list[str] = []

        for node_name in self.nodes_by_name:
            # Skip if already done or in progress
            if node_name in completed or node_name in running or node_name in scheduled:
                continue

            # Check if dependencies are satisfied
            if self._are_dependencies_satisfied(node_name, completed):
                ready.append(node_name)

        return ready

    async def _execute_node(
        self,
        node_name: str,
        node_input: Any,
        results: dict[str, Any],
        config: RunConfig,
    ) -> AsyncIterator[ProgressItem]:
        """Execute a single node and stream its progress items.

        Args:
            node_name: Name of the node to execute.
            node_input: Input data for the node.
            results: Dictionary of results from completed nodes.
            config: Execution configuration.

        Yields:
            Progress items from the node execution.

        Raises:
            FlowError: If the node execution fails.

        """
        node = self.nodes_by_name[node_name]

        # Execute node and wrap exceptions
        try:
            final_result = None
            async for item in node.astream(node_input):
                yield item

                # Capture final result from ToolResult or StreamEnd
                if isinstance(item, ToolResult):
                    final_result = item.result
                elif (
                    isinstance(item, StreamEnd) and item.result and final_result is None
                ):
                    # Use result from StreamEnd
                    final_result = item.result

            # Store the result
            if final_result is not None:
                results[node_name] = final_result
        except FlowError:
            # Re-raise FlowError as-is
            raise
        except Exception as e:
            # Wrap other exceptions in FlowError
            msg = f"Flow execution failed in node '{node_name}': {e}"
            raise FlowError(msg) from e

    def _get_node_input(
        self,
        node: BaseNode[Any, Any],
        results: dict[str, Any],
    ) -> Any:
        """Get input data for a node from completed results.

        Args:
            node: The node to get input for.
            results: Dictionary of completed node results.

        Returns:
            Input data for the node.

        Raises:
            FlowError: If required input is missing.

        """
        # Determine how to gather input based on dependencies
        deps = node.dependencies

        if len(deps) == 0:
            # No dependencies - this shouldn't be called
            return None
        elif len(deps) == 1:
            # Single input dependency - check if node is a MergeNode
            # MergeNode always expects tuple input, even with single dependency
            input_node_name = deps[0].name
            if input_node_name not in results:
                msg = f"Missing input for node {node.name}: {input_node_name}"
                raise FlowError(msg)

            result_value = results[input_node_name]

            # Single input dependency - return directly
            return result_value
        else:
            # Multiple input dependencies - return as tuple
            return tuple(results[dep.name] for dep in deps)

    async def astream(  # noqa: PLR0915
        self,
        inputs: InputT,
        config: RunConfig | None = None,
    ) -> AsyncIterator[ProgressItem]:
        """Execute the flow and stream progress items eagerly.

        Nodes execute as soon as their dependencies are satisfied, maximizing
        parallelism and minimizing total execution time. Progress items are
        streamed in real-time as they occur.

        Args:
            inputs: Input data matching InputT.
            config: Optional execution configuration.

        Yields:
            ProgressItem objects from node execution in completion order.

        Raises:
            RecursionLimitError: If max_steps is exceeded.
            FlowTimeoutError: If timeout_seconds is exceeded.
            FlowError: For other execution errors.

        """
        if config is None:
            config = RunConfig()

        self._validate_input_type(inputs)

        start_time = time.time()
        results: dict[str, Any] = {}
        completed: set[str] = set()
        running: set[str] = set()
        tasks: dict[str, asyncio.Task[None]] = {}
        step_count = 0
        flow_ended = False  # Track if Route.END was returned
        run_id = config.run_id or "default_run"

        # Initialize checkpoint manager
        checkpoint_manager = await self._initialize_checkpoint_manager(config, run_id)

        # Track execution progress for checkpointing (if enabled)
        execution_progress: dict[str, str] = {}
        if checkpoint_manager is not None:
            for node_name in self.nodes_by_name:
                execution_progress[node_name] = "pending"

        # Queue for streaming progress items from concurrent nodes
        item_queue: asyncio.Queue[ProgressItem | None] = asyncio.Queue()

        # Queue for nodes that need to be scheduled due to routing
        routing_queue: asyncio.Queue[str] = asyncio.Queue()

        # Track routing sources: maps target node to the node that routed to it
        routing_sources: dict[str, str] = {}

        # Store entry node input
        entry_input = inputs

        async def node_executor(node_name: str) -> None:
            """Execute a node and queue its progress items."""
            nonlocal step_count, flow_ended

            try:
                running.add(node_name)
                node = self.nodes_by_name[node_name]

                # Track execution progress for checkpointing (if enabled)
                if checkpoint_manager is not None:
                    execution_progress[node_name] = "running"

                # Determine input for this node
                node_input = self._determine_node_input(
                    node, node_name, results, routing_sources, entry_input
                )

                # Execute and queue progress items
                async for item in self._execute_node(
                    node_name, node_input, results, config
                ):
                    await item_queue.put(item)

                # Mark as completed
                completed.add(node_name)
                running.remove(node_name)
                step_count += 1

                # Track completion for checkpointing (if enabled)
                if checkpoint_manager is not None:
                    execution_progress[node_name] = "completed"

                # Handle checkpoint and routing
                routed_nodes, ended = await self._handle_checkpoint_after_node(
                    checkpoint_manager, node_name, results, completed, running
                )

                # For loops: if node has downstream edges, clear completion
                # so they can re-execute. Only if flow isn't ending.
                if not ended and node_name in self.edges:
                    for downstream_node in self.edges[node_name]:
                        completed.discard(downstream_node)

                if ended:
                    flow_ended = True
                else:
                    # Queue routed nodes for execution and track routing source
                    for routed_node in routed_nodes:
                        await routing_queue.put(routed_node)
                        routing_sources[routed_node] = (
                            node_name  # Track which node routed here
                        )

                # Check execution limits
                self._check_execution_limits(step_count, start_time, config)

            except Exception:
                # Queue error for main loop to handle
                await item_queue.put(None)  # Signal completion
                raise

        # Start entry nodes
        for node_name in self.entry_nodes:
            task = asyncio.create_task(node_executor(node_name))
            tasks[node_name] = task

        # Main execution loop - continue while there's work to do
        while True:
            # Process routing queue
            await self._process_routing_queue(
                routing_queue, completed, running, tasks, node_executor
            )

            # Check for newly ready nodes (based on static edges)
            ready = self._get_ready_nodes(completed, running, set(tasks.keys()))

            # Schedule newly ready nodes
            for node_name in ready:
                task = asyncio.create_task(node_executor(node_name))
                tasks[node_name] = task

            # Check if flow ended via Route.END
            if flow_ended and not tasks and not running:
                break

            # If no tasks are running and none are ready, we're done
            if not tasks and not running:
                break

            # Wait for any task to make progress or complete
            if tasks:
                await self._handle_completed_tasks(tasks)
            else:
                # No tasks running, small sleep to prevent busy-waiting
                await asyncio.sleep(0.001)

            # Yield any queued progress items
            async for item in self._yield_queued_items(item_queue):
                yield item

        # Drain remaining items from queue
        async for item in self._yield_queued_items(item_queue):
            yield item

        # Finalize and emit result
        result = await self._finalize_flow(results, checkpoint_manager)
        yield result

    async def _apply_routing(
        self, node_name: str, results: dict[str, Any]
    ) -> tuple[set[str], bool]:
        """Apply conditional routing after a node completes.

        Args:
            node_name: Name of the node that just completed.
            results: Current execution results.

        Returns:
            Tuple of (routed_node_names, flow_ended).

        """
        routed_nodes: set[str] = set()
        flow_ended = False

        # Check for conditional edges from this node
        for source_name, router_fn, mapping in self.conditional_edges:
            if source_name == node_name:
                # Build state from results
                state = self._build_state(results)

                # Apply router
                raw_outcome = router_fn(state)
                outcomes = (
                    raw_outcome if isinstance(raw_outcome, list) else [raw_outcome]
                )

                # Resolve each outcome
                for outcome in outcomes:
                    target, ended = self._resolve_routing_outcome(outcome, mapping)
                    if target is not None:
                        routed_nodes.add(target)
                    flow_ended = flow_ended or ended

        return routed_nodes, flow_ended

    def _resolve_routing_outcome(
        self, outcome: T_Route, mapping: dict[Any, str] | None
    ) -> tuple[str | None, bool]:
        """Resolve a routing outcome to a target node name.

        Args:
            outcome: The routing outcome from the router function.
            mapping: Optional mapping from outcomes to node names.

        Returns:
            Tuple of (target_node_name, flow_ended).

        Raises:
            RoutingError: If outcome is invalid.

        """
        # Apply mapping if provided
        if mapping is not None:
            if outcome not in mapping:
                msg = (
                    f"Router outcome {outcome!r} not in mapping: {list(mapping.keys())}"
                )
                raise RoutingError(msg)
            target = mapping[outcome]
        else:
            target = outcome

        # Check for special Route.END
        if target == Route.END:
            return None, True

        # Validate target is a node name
        if isinstance(target, str):
            if target not in self.nodes_by_name:
                msg = f"Router target {target!r} is not a valid node name"
                raise RoutingError(msg)
            return target, False

        msg = f"Invalid routing target: {target!r}"
        raise RoutingError(msg)

    def _build_state(self, results: dict[str, Any]) -> BaseModel:
        """Build a state model from execution results.

        Args:
            results: Dictionary of node execution results.

        Returns:
            A BaseModel instance with results as fields.

        """
        state_dict = dict(results)
        annotations = {k: type(v) for k, v in state_dict.items()}
        state_model = type("State", (BaseModel,), {"__annotations__": annotations})
        return state_model(**state_dict)

    def _construct_output(self, results: dict[str, Any]) -> OutputT:
        """Construct the final output from node results.

        Args:
            results: Dictionary of all node results.

        Returns:
            Output data matching OutputT.

        Raises:
            FlowError: If output cannot be constructed.

        """
        # Find terminal nodes (nodes with no dependents)
        terminal_nodes = [name for name, deps in self.dependents.items() if not deps]

        if len(terminal_nodes) == 1:
            # Single terminal node - use its result
            terminal_result = results.get(terminal_nodes[0])
            if isinstance(terminal_result, self.output_type):
                return terminal_result

        # Multiple terminal nodes or need to construct from all results
        # Try to construct output from results dict
        try:
            # Convert results for construction, keeping BaseModels as-is
            results_for_construction = {}
            for key, value in results.items():
                if isinstance(value, GenericResult):
                    # Unwrap GenericResult to get the actual value
                    results_for_construction[key] = value.value
                else:
                    # Pass value as-is (BaseModel or primitive)
                    # Pydantic will handle BaseModel validation automatically
                    results_for_construction[key] = value
            return self.output_type(**results_for_construction)
        except Exception as e:
            msg = f"Failed to construct output type {self.output_type.__name__}: {e}"
            raise FlowError(msg) from e
