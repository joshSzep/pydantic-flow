"""Flow orchestration for the pydantic-flow framework.

This module provides the Flow class that manages workflow execution,
dependency resolution, and DAG validation.

BREAKING CHANGE: Added HITL (Human-in-the-Loop) support with flow_id,
interrupt handlers, checkpoints, and resumption. Added checkpoint persistence
via pluggable checkpoint stores.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from enum import StrEnum
from typing import TYPE_CHECKING
from typing import Any
from typing import TypeVar
import uuid

from pydantic import BaseModel

from pydantic_flow.cache.base import CacheBackend
from pydantic_flow.cache.base import CachePolicy
from pydantic_flow.core.errors import FlowCheckpoint
from pydantic_flow.core.errors import FlowError
from pydantic_flow.core.errors import HandlerPriority
from pydantic_flow.core.errors import InterruptHandlerRegistration
from pydantic_flow.core.errors import InterruptionRequested
from pydantic_flow.core.routing import T_Route
from pydantic_flow.core.run_config import RunConfig
from pydantic_flow.engine.stepper import ConditionalEdge
from pydantic_flow.engine.stepper import EngineConfig
from pydantic_flow.engine.stepper import StepperEngine
from pydantic_flow.flow.exceptions import CyclicDependencyError
from pydantic_flow.memory import ConversationMemory
from pydantic_flow.memory import MemoryConfig
from pydantic_flow.memory import _active_flow_memory
from pydantic_flow.nodes import BaseNode
from pydantic_flow.nodes.protocols import has_input_dependency
from pydantic_flow.nodes.protocols import has_multiple_inputs
from pydantic_flow.streaming.events import InterruptCallback
from pydantic_flow.streaming.events import InterruptDecision
from pydantic_flow.streaming.events import ProgressItem
from pydantic_flow.streaming.events import ToolResult

if TYPE_CHECKING:
    from pydantic_flow.checkpoints.interface import CheckpointEnvelope
    from pydantic_flow.checkpoints.interface import CheckpointStore

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

    def __init__(
        self,
        *,
        input_type: type[InputT],
        output_type: type[OutputT],
        memory_config: MemoryConfig | None = None,
        cache_backend: CacheBackend | None = None,
        default_cache_policy: CachePolicy | None = None,
    ) -> None:
        """Initialize a flow with the required input and output types.

        Args:
            input_type: The BaseModel class that this flow accepts as input.
            output_type: The BaseModel class to construct from flow results.
            memory_config: Optional configuration for conversation memory.
                         If None, uses default MemoryConfig().
            cache_backend: Optional cache backend for node result caching.
            default_cache_policy: Optional default cache policy for nodes.

        """
        self.nodes: list[BaseNode[Any, Any]] = []
        self._execution_order: list[BaseNode[Any, Any]] = []
        self._results: dict[str, Any] = {}
        self._input_type = input_type
        self._output_type = output_type
        self._edges: dict[str, list[str]] = {}
        self._conditional_edges: list[ConditionalEdge[Any]] = []
        self._entry_nodes: list[str] | None = None
        self.flow_id: str = str(uuid.uuid4())
        self._interrupt_handlers: list[InterruptHandlerRegistration] = []
        self._edge_history: list[tuple[str, str]] = []
        self.memory_config = memory_config or MemoryConfig()
        self._cache_backend = cache_backend
        self._default_cache_policy = default_cache_policy
        self._conversation_memory: ConversationMemory | None = None
        if self.memory_config.enable_conversation_memory:
            self._conversation_memory = ConversationMemory(
                compressor=self.memory_config.compressor,
            )

    def register_interrupt_handler(
        self,
        callback: InterruptCallback,
        priority: int = HandlerPriority.NORMAL,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register a flow-level interrupt callback handler.

        Flow-level handlers execute for all progress items across all nodes.
        They run after node-level handlers.

        Args:
            callback: Async function that receives ProgressItem and returns
                InterruptDecision.
            priority: Priority level (0-100, lower executes first).
            metadata: Optional metadata about the handler.

        """
        registration = InterruptHandlerRegistration(
            callback=callback,
            priority=priority,
            metadata=metadata or {},
        )
        self._interrupt_handlers.append(registration)
        # Keep handlers sorted by priority
        self._interrupt_handlers.sort(key=lambda h: h.priority)

    def clear_interrupt_handlers(self) -> None:
        """Remove all registered flow-level interrupt handlers."""
        self._interrupt_handlers.clear()

    async def _check_interrupt_handlers(self, item: ProgressItem) -> InterruptDecision:
        """Check all registered flow-level interrupt handlers.

        Executes handlers in priority order. If any handler requests
        interruption, returns immediately with that decision.

        Args:
            item: The progress item to check.

        Returns:
            InterruptDecision indicating whether to interrupt.

        """
        for handler in self._interrupt_handlers:
            decision = await handler.callback(item)
            if decision.should_interrupt:
                return decision
        return InterruptDecision.proceed()

    def _create_checkpoint(
        self, interrupted_node_id: str, run_id: str
    ) -> FlowCheckpoint:
        """Create a checkpoint for flow resumption.

        Args:
            interrupted_node_id: ID of the node where interruption occurred.
            run_id: Current run identifier.

        Returns:
            FlowCheckpoint with captured state.

        """
        conversation_memory = None
        if self._conversation_memory is not None:
            conversation_memory = self._conversation_memory.get()

        return FlowCheckpoint(
            flow_id=self.flow_id,
            run_id=run_id,
            interrupted_node_id=interrupted_node_id,
            node_states=self._results.copy(),
            edge_history=self._edge_history.copy(),
            conversation_memory=conversation_memory,
        )

    async def _persist_checkpoint(
        self,
        checkpoint: FlowCheckpoint,
        store: CheckpointStore,
    ) -> CheckpointEnvelope:
        """Persist a checkpoint to the configured store.

        Args:
            checkpoint: The checkpoint to persist.
            store: The checkpoint store to use.

        Returns:
            The persisted checkpoint envelope with ID.

        Raises:
            CheckpointBackendError: If persistence fails.

        """
        # Local imports to avoid circular dependency
        from pydantic_flow.checkpoints.interface import CheckpointEnvelope
        from pydantic_flow.checkpoints.interface import CheckpointId
        from pydantic_flow.checkpoints.interface import RunId
        from pydantic_flow.checkpoints.interface import generate_checkpoint_id

        envelope = CheckpointEnvelope(
            id=CheckpointId(generate_checkpoint_id()),
            run_id=RunId(checkpoint.run_id),
            node_id=checkpoint.interrupted_node_id,
            checkpoint=checkpoint,
        )

        return await store.save(envelope, overwrite=False)

    async def resume(self, checkpoint: FlowCheckpoint, inputs: InputT) -> OutputT:
        """Resume flow execution from a checkpoint.

        Args:
            checkpoint: The checkpoint to resume from.
            inputs: Original input data.

        Returns:
            Flow output.

        Raises:
            FlowError: If checkpoint is invalid or execution fails.

        """
        # Validate checkpoint belongs to this flow
        if checkpoint.flow_id != self.flow_id:
            msg = (
                f"Checkpoint flow_id mismatch: expected {self.flow_id}, "
                f"got {checkpoint.flow_id}"
            )
            raise FlowError(msg)

        # Restore state from checkpoint
        self._results = checkpoint.node_states.copy()
        self._edge_history = checkpoint.edge_history.copy()

        # Restore conversation memory from checkpoint
        if checkpoint.conversation_memory is not None:
            if self._conversation_memory is not None:
                self._conversation_memory.clear()
                self._conversation_memory.extend(checkpoint.conversation_memory)
            else:
                # If memory wasn't enabled but checkpoint has it, create it
                self._conversation_memory = ConversationMemory(
                    initial_messages=checkpoint.conversation_memory,
                    compressor=self.memory_config.compressor,
                )

        # Find the interrupted node
        interrupted_node = None
        for node in self.nodes:
            if node.name == checkpoint.interrupted_node_id:
                interrupted_node = node
                break

        if interrupted_node is None:
            msg = f"Interrupted node {checkpoint.interrupted_node_id} not found in flow"
            raise FlowError(msg)

        # Resume from the node after the interrupted one
        resume_from_index = self._execution_order.index(interrupted_node) + 1

        # Set conversation memory in context if enabled
        token = None
        if self._conversation_memory is not None:
            token = _active_flow_memory.set(self._conversation_memory)

        try:
            for node in self._execution_order[resume_from_index:]:
                # Determine input data for the node
                if has_multiple_inputs(node):
                    input_data: Any = tuple(
                        self._results[dep.node.name] for dep in node.inputs
                    )
                elif has_input_dependency(node):
                    input_node = node.input.node
                    if input_node.name not in self._results:
                        msg = f"Input node {input_node.name} has not been executed"
                        raise FlowError(msg)
                    input_data = self._results[input_node.name]
                else:
                    input_data = inputs

                # Execute the node
                result = await node.run(input_data)  # type: ignore[union-attr]
                self._results[node.name] = result

            # Construct the output BaseModel from the results
            return self._output_type(**self._results)

        except Exception as e:
            if isinstance(e, FlowError):
                raise
            msg = f"Flow resumption failed: {e}"
            raise FlowError(msg) from e

        finally:
            # Reset context variable if it was set
            if token is not None:
                _active_flow_memory.reset(token)

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

    async def run(self, inputs: InputT, config: RunConfig | None = None) -> OutputT:
        """Execute the flow with the given inputs.

        Args:
            inputs: The input data for the flow (must match the flow's InputT type)
            config: Optional run configuration including checkpoint store

        Returns:
            The flow results with the specified OutputT type

        Raises:
            FlowError: If the flow execution fails
            TypeError: If the input type doesn't match the expected input_type
            InterruptionRequested: If a HITL interrupt handler requests stopping

        """
        if not isinstance(inputs, self._input_type):
            expected_name = self._input_type.__name__
            actual_name = type(inputs).__name__
            msg = f"Input type mismatch: expected {expected_name}, got {actual_name}"
            raise TypeError(msg)

        config = config or RunConfig()
        run_id = config.run_id or str(uuid.uuid4())

        self._results = {}
        self._edge_history = []

        # Set conversation memory in context if enabled
        token = None
        if self._conversation_memory is not None:
            token = _active_flow_memory.set(self._conversation_memory)

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
                    # Track edge
                    self._edge_history.append((input_node.name, node.name))
                else:
                    # No-input node: takes input from flow inputs
                    input_data = inputs

                # Execute node with interrupt checking
                # Consume stream to check interrupts, get result via node.run()
                result: Any = None
                tool_result: Any = None
                stream_items: list[ProgressItem] = []

                async for item in node.astream(input_data):  # type: ignore[union-attr]
                    stream_items.append(item)
                    # Check interrupt handlers on each progress item
                    decision = await self._check_interrupt_handlers(item)
                    if decision.should_interrupt:
                        checkpoint = self._create_checkpoint(node.name, run_id)
                        raise InterruptionRequested(
                            checkpoint=checkpoint,
                            decision=decision,
                        )

                    # Extract result using same logic as BaseNode.run()
                    if isinstance(item, ToolResult) and item.result is not None:
                        tool_result = item.result

                # Prefer ToolResult if available, otherwise run full reconstruction
                if tool_result is not None:
                    result = tool_result
                else:
                    # Need to reconstruct from stream - just call node.run()
                    result = await node.run(input_data)  # type: ignore[union-attr]

                self._results[node.name] = result

            # Smart detection: If we have a single result that's already
            # the output type, return it directly instead of trying to reconstruct
            if len(self._results) == 1:
                single_result = next(iter(self._results.values()))
                if isinstance(single_result, self._output_type):
                    return single_result  # type: ignore

            # Construct the output BaseModel from the results
            return self._output_type(**self._results)

        except InterruptionRequested as e:
            # Enhance checkpoint with flow-level information
            e.checkpoint.flow_id = self.flow_id
            e.checkpoint.node_states = self._results.copy()
            e.checkpoint.edge_history = self._edge_history.copy()
            # Capture conversation memory
            if self._conversation_memory is not None:
                e.checkpoint.conversation_memory = self._conversation_memory.get()

            # Persist checkpoint if store configured
            if config.checkpoint_store is not None:
                envelope = await self._persist_checkpoint(
                    e.checkpoint, config.checkpoint_store
                )

                # Attach checkpoint ID to exception metadata
                e.checkpoint.metadata = e.checkpoint.metadata or {}
                e.checkpoint.metadata["checkpoint_id"] = envelope.id
                e.checkpoint.metadata["run_id"] = run_id

            raise

        except Exception as e:
            # Wrap any other exception in a FlowError for consistency
            if isinstance(e, FlowError):
                raise
            msg = f"Flow execution failed: {e}"
            raise FlowError(msg) from e

        finally:
            # Reset context variable if it was set
            if token is not None:
                _active_flow_memory.reset(token)

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
                cache_backend=self._cache_backend,
                default_cache_policy=self._default_cache_policy,
            )
            engine = StepperEngine(engine_config)
            return CompiledFlow(flow=self, engine=engine, use_stepper=True)

        self._calculate_execution_order()
        return CompiledFlow(flow=self, use_stepper=False)

    async def cache_delete(self, key: str) -> None:
        """Delete a specific cache entry.

        Args:
            key: The cache key to delete.

        Raises:
            FlowError: If no cache backend is configured.

        """
        if self._cache_backend is None:
            msg = "No cache backend configured for this flow"
            raise FlowError(msg)
        await self._cache_backend.delete(key)

    async def cache_invalidate(self, namespace: str) -> int:
        """Invalidate all cache entries in a namespace.

        Args:
            namespace: The cache namespace to invalidate.

        Returns:
            Number of entries invalidated.

        Raises:
            FlowError: If no cache backend is configured.

        """
        if self._cache_backend is None:
            msg = "No cache backend configured for this flow"
            raise FlowError(msg)
        return await self._cache_backend.invalidate_namespace(namespace)

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

        Raises:
            InterruptionRequested: If a HITL interrupt occurs.

        """
        if self.use_stepper and self.engine is not None:
            # Set memory context for stepper engine execution
            token = None
            if self.flow is not None and self.flow._conversation_memory is not None:
                token = _active_flow_memory.set(self.flow._conversation_memory)
            try:
                return await self.engine.invoke(inputs, config)
            finally:
                if token is not None:
                    _active_flow_memory.reset(token)
        if self.flow is not None:
            return await self.flow.run(inputs)
        msg = "CompiledFlow has neither flow nor engine configured"
        raise FlowError(msg)

    async def resume(self, checkpoint: FlowCheckpoint, inputs: InputT) -> OutputT:
        """Resume flow execution from a checkpoint.

        Args:
            checkpoint: The checkpoint to resume from.
            inputs: Original input data.

        Returns:
            Flow output.

        Raises:
            FlowError: If checkpoint is invalid or engine doesn't support resumption.

        """
        if self.flow is not None:
            return await self.flow.resume(checkpoint, inputs)
        # TODO: Implement resumption for stepper engine
        msg = "Resumption not yet supported for stepper engine"
        raise FlowError(msg)
