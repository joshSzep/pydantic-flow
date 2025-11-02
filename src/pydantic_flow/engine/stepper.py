"""Stepper-based execution engine for loop-capable flows.

This module provides a frontier-based execution engine that supports cycles,
conditional routing, and dynamic control flow.
"""

from collections.abc import AsyncIterator
from collections.abc import Callable
import time
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator

from pydantic_flow.cache.middleware import maybe_cached_execute
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
from pydantic_flow.streaming import ProgressItem
from pydantic_flow.streaming import ToolResult
from pydantic_flow.streaming.core_events import StreamEnd

if TYPE_CHECKING:
    pass


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
        cache_backend: Optional cache backend for node execution.
        default_cache_policy: Optional default cache policy for nodes.
        flow_id: Flow identifier for checkpoint tracking (from Flow.flow_id).

    """

    nodes: list[BaseNode[Any, Any]]
    edges: dict[str, list[str]] = Field(default_factory=dict)
    conditional_edges: list[ConditionalEdge[Any]] = Field(default_factory=list)
    entry_nodes: list[str]
    input_type: type[InputT]
    output_type: type[OutputT]
    cache_backend: Any = None
    default_cache_policy: Any = None
    flow_id: str

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
        self.cache_backend = config.cache_backend
        self.default_cache_policy = config.default_cache_policy
        self.flow_id = config.flow_id
        self._background_tasks: set[Any] = set()

    @staticmethod
    async def _extract_result_from_stream(
        stream: AsyncIterator[ProgressItem],
    ) -> Any:
        """Extract final result from a node's astream.

        Args:
            stream: Async iterator of progress items.

        Returns:
            The extracted result object.

        Raises:
            RuntimeError: If no result found in stream.

        """
        final_result: Any = None
        tool_result: Any = None

        async for item in stream:
            # Extract result from ToolResult (preferred - has actual object)
            if isinstance(item, ToolResult) and item.result is not None:
                tool_result = item.result
            # StreamEnd carries result preview as fallback
            elif isinstance(item, StreamEnd) and item.result_preview:
                final_result = item.result_preview

        # Prefer actual result from ToolResult if available
        if tool_result is not None:
            return tool_result

        if final_result is None:
            msg = "Node did not produce a result in stream"
            raise RuntimeError(msg)

        return final_result

    async def _maybe_checkpoint_after_frontier(
        self,
        config: RunConfig,
        current_frontier: list[str],
        results: dict[str, Any],
        run_id: str,
        execution_progress: dict[str, str] | None = None,
    ) -> None:
        """Create checkpoint after frontier execution if durability mode requires it.

        Note: V1 checkpoint logic removed. V2 CheckpointManager handles all
        checkpointing automatically during flow execution.

        Args:
            config: Run configuration with checkpoint store and durability mode.
            current_frontier: Nodes that were just executed.
            results: Current node execution results.
            run_id: Current run identifier.
            execution_progress: Optional execution progress tracking.

        """
        # V1 checkpoint code removed - V2 CheckpointManager handles this
        pass

    async def _checkpoint_on_exit(
        self,
        config: RunConfig,
        current_frontier: list[str],
        results: dict[str, Any],
        run_id: str,
        execution_progress: dict[str, str] | None = None,
        checkpoint_reason: str = "flow_end",
    ) -> None:
        """Create checkpoint on flow exit if EXIT durability mode is enabled.

        Note: V1 checkpoint logic removed. V2 CheckpointManager handles all
        checkpointing automatically during flow execution.

        Args:
            config: Run configuration with checkpoint store and durability mode.
            current_frontier: Last executed frontier.
            results: Current node execution results.
            run_id: Current run identifier.
            execution_progress: Optional execution progress tracking.
            checkpoint_reason: Reason for checkpoint (flow_end or error).

        """
        # V1 checkpoint code removed - V2 CheckpointManager handles this
        pass

    async def astream(  # noqa: PLR0912, PLR0915
        self,
        inputs: InputT,
        config: RunConfig | None = None,
    ) -> AsyncIterator[ProgressItem]:
        """Execute the flow and stream progress items.

        Args:
            inputs: Input data matching InputT.
            config: Optional execution configuration.

        Yields:
            ProgressItem objects from node execution.

        Returns:
            Final output data matching OutputT (via FinalResult progress item).

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
        run_id = config.run_id or "default_run"
        current_frontier: list[str] = []

        # Track execution progress for checkpointing
        execution_progress: dict[str, str] = {}
        for node_name in self.nodes_by_name:
            execution_progress[node_name] = "pending"

        # Initialize checkpoint manager (creates default in-memory backend if needed)
        backend = config.checkpoint_backend
        backend_created_by_us = False

        # Create default in-memory SQLite backend if none provided
        if backend is None:
            backend_created_by_us = True
            # isort: off
            from pathlib import Path  # noqa: PLC0415

            from pydantic_flow.checkpoints.backends.sqlite import (  # noqa: PLC0415
                SQLiteCheckpointBackend,
            )
            from pydantic_flow.checkpoints.backends.sqlite import SQLiteCheckpointConfig  # noqa: PLC0415
            # isort: on

            backend = SQLiteCheckpointBackend(
                config=SQLiteCheckpointConfig(db_path=Path(":memory:"))
            )
            await backend.initialize()

        # Always create checkpoint manager for HITL support
        # isort: off
        from pydantic_flow.checkpoints import CheckpointConfig  # noqa: PLC0415
        from pydantic_flow.checkpoints import CheckpointManager  # noqa: PLC0415
        from pydantic_flow.checkpoints.types import RunId as CheckpointRunId  # noqa: PLC0415
        # isort: on

        checkpoint_cfg = config.checkpoint_config or CheckpointConfig()
        checkpoint_manager = CheckpointManager(
            config=checkpoint_cfg,
            storage=backend,
            flow_id=self.flow_id,
            run_id=CheckpointRunId(run_id),
        )
        await checkpoint_manager.initialize_run()

        try:
            while frontier:
                self._check_limits(iteration, config, start_time, events)

                current_frontier = list(frontier)
                frontier = set()

                async for item in self._execute_frontier(
                    current_frontier,
                    inputs,
                    results,
                    execution_progress,
                    checkpoint_manager,
                ):
                    yield item

                next_frontier, ended = await self._route_next(current_frontier, results)

                await self._maybe_checkpoint_after_frontier(
                    config, current_frontier, results, run_id, execution_progress
                )

                # Save checkpoint v2 after wave execution
                if checkpoint_manager is not None:
                    await checkpoint_manager.save_wave_checkpoint(
                        current_state=results,
                        next_frontier=list(next_frontier),
                        routing_ended=ended,
                    )

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

            output = self.output_type(**results)
            await self._checkpoint_on_exit(
                config,
                current_frontier,
                results,
                run_id,
                execution_progress,
                checkpoint_reason="flow_end",
            )

            # Finalize checkpoint v2 run
            if checkpoint_manager is not None:
                from pydantic_flow.checkpoints.types import RunMetadata  # noqa: PLC0415

                await checkpoint_manager.finalize_run(
                    status=RunMetadata.Status.COMPLETED
                )

            # Yield final result instead of returning
            from pydantic_flow.streaming.core_events import FlowResult  # noqa: PLC0415

            yield FlowResult(
                run_id=run_id,
                node_id="flow",
                result=output,
            )

        except FlowError:
            # Mark failed nodes
            for node_name, status in execution_progress.items():
                if status == "running":
                    execution_progress[node_name] = "failed"
            await self._checkpoint_on_exit(
                config,
                current_frontier,
                results,
                run_id,
                execution_progress,
                checkpoint_reason="error",
            )

            # Finalize checkpoint v2 run as failed
            if checkpoint_manager is not None:
                from pydantic_flow.checkpoints.types import RunMetadata  # noqa: PLC0415

                await checkpoint_manager.finalize_run(status=RunMetadata.Status.FAILED)

            raise
        except Exception as e:
            # Mark failed nodes
            for node_name, status in execution_progress.items():
                if status == "running":
                    execution_progress[node_name] = "failed"
            await self._checkpoint_on_exit(
                config,
                current_frontier,
                results,
                run_id,
                execution_progress,
                checkpoint_reason="error",
            )

            # Finalize checkpoint v2 run as failed
            if checkpoint_manager is not None:
                from pydantic_flow.checkpoints.types import RunMetadata  # noqa: PLC0415

                await checkpoint_manager.finalize_run(status=RunMetadata.Status.FAILED)

            msg = f"Flow execution failed: {e}"
            raise FlowError(msg) from e
        finally:
            # Clean up auto-created backend to prevent hanging
            if backend_created_by_us and backend is not None:
                await backend.close()

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
        execution_progress: dict[str, str] | None = None,
        checkpoint_manager: Any = None,
    ) -> AsyncIterator[ProgressItem]:
        """Execute all nodes in the current frontier and stream progress.

        Note: Nodes are currently executed sequentially to enable real-time
        streaming. Parallel execution with streaming will be added in a future update.

        Args:
            frontier: List of node names to execute.
            inputs: Flow inputs.
            results: Current execution results.
            execution_progress: Optional execution progress tracking.
            checkpoint_manager: Optional checkpoint v2 manager for event logging.

        Yields:
            ProgressItem objects from node execution.

        """
        for node_name in frontier:
            if execution_progress is not None:
                execution_progress[node_name] = "running"

            node = self.nodes_by_name[node_name]
            input_data = self._get_node_input(node, inputs, results)

            # Create event log for this node if trace sampling enabled
            event_log = None
            if checkpoint_manager is not None:
                event_log = checkpoint_manager.create_event_log(node_id=node_name)

            # Check if node supports caching
            node_cache_policy = getattr(node, "cache_policy", None)
            effective_policy = node_cache_policy or self.default_cache_policy

            if self.cache_backend is not None and effective_policy is not None:
                # Use cached execution - need to wrap in stream extraction
                async def cached_exec(
                    _node: BaseNode[Any, Any] = node,
                    _input_data: Any = input_data,
                ) -> Any:
                    return await self._extract_result_from_stream(
                        _node.astream(_input_data)
                    )

                result, _cache_events = await maybe_cached_execute(
                    node_name=node_name,
                    inputs={"input": input_data},
                    exec_fn=cached_exec,
                    backend=self.cache_backend,
                    policy=effective_policy,
                    context=None,
                )
            else:
                # Stream through node execution and extract result
                result = None
                async for item in node.astream(input_data):
                    # Yield progress item to user
                    yield item

                    # Also log to checkpoint if enabled
                    if event_log is not None:
                        await event_log.append(item)

                    # Extract result from ToolResult if available
                    if isinstance(item, ToolResult) and item.result is not None:
                        result = item.result
                    # Or from StreamEnd result_preview
                    elif (
                        isinstance(item, StreamEnd)
                        and item.result_preview
                        and result is None
                    ):
                        # Only use preview if no ToolResult
                        result = item.result_preview

                if result is None:
                    msg = f"Node {node_name} did not produce a result"
                    raise FlowError(msg)

            if execution_progress is not None:
                execution_progress[node_name] = "completed"

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

        # Filter to only include nodes whose dependencies are satisfied
        ready_frontier = {
            node_name
            for node_name in next_frontier
            if self._dependencies_ready(node_name, results)
        }

        return ready_frontier, ended

    def _dependencies_ready(self, node_name: str, results: dict[str, Any]) -> bool:
        """Check if all dependencies for a node are satisfied."""
        node = self.nodes_by_name.get(node_name)
        if node is None:
            return False

        # Check multi-input dependencies
        if has_multiple_inputs(node):
            node_with_inputs = cast(NodeWithInputs, node)
            return all(dep.node.name in results for dep in node_with_inputs.inputs)

        # Check single-input dependency
        if has_input_dependency(node):
            node_with_input = cast(NodeWithInput, node)
            return node_with_input.input.node.name in results

        # Node has no explicit input dependencies
        return True

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
