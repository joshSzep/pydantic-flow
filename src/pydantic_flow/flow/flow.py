"""Flow orchestration for the pydantic-flow framework.

This module provides the Flow class that manages workflow execution,
dependency resolution, and DAG validation.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING
from typing import Any
from typing import TypeVar
import uuid

from pydantic import BaseModel

from pydantic_flow.cache.base import CacheBackend
from pydantic_flow.cache.base import CachePolicy
from pydantic_flow.core.errors import FlowError
from pydantic_flow.core.routing import Route
from pydantic_flow.core.run_config import RunConfig
from pydantic_flow.engine.stepper import ConditionalEdge
from pydantic_flow.engine.stepper import EngineConfig
from pydantic_flow.engine.stepper import StepperEngine
from pydantic_flow.flow.exceptions import CyclicDependencyError
from pydantic_flow.hitl.decisions import InterruptCallback
from pydantic_flow.hitl.decisions import InterruptDecision
from pydantic_flow.hitl.interrupts import FlowCheckpoint
from pydantic_flow.hitl.interrupts import HandlerPriority
from pydantic_flow.hitl.interrupts import InterruptHandlerRegistration
from pydantic_flow.hitl.interrupts import InterruptionRequested
from pydantic_flow.memory import ConversationMemory
from pydantic_flow.memory import MemoryConfig
from pydantic_flow.memory import _active_flow_memory
from pydantic_flow.nodes import BaseNode
from pydantic_flow.nodes.protocols import has_input_dependency
from pydantic_flow.nodes.protocols import has_multiple_inputs
from pydantic_flow.streaming.base import ProgressItem
from pydantic_flow.streaming.tool_events import ToolResult

if TYPE_CHECKING:
    from pydantic_flow.hitl.checkpoints.interface import CheckpointEnvelope
    from pydantic_flow.hitl.checkpoints.interface import CheckpointStore

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


@dataclass
class Edge:
    """Represents a directed edge between two nodes.

    Attributes:
        source: The source node.
        target: The target node.

    """

    source: BaseNode[Any, Any]
    target: BaseNode[Any, Any]


@dataclass
class ConditionalEdgeConfig:
    """Configuration for a conditional routing edge.

    Attributes:
        source: The source node where routing decision is made.
        router: Function that determines the next node(s) to execute.

    """

    source: BaseNode[Any, Any]
    router: Callable[[BaseModel], Any]


@dataclass
class GraphAnalysis:
    """Results of flow graph structure analysis.

    Attributes:
        has_cycles: Whether the graph contains cycles.
        has_conditional_edges: Whether there are conditional routing edges.
        has_explicit_edges: Whether there are explicit edges defined.
        entry_nodes: List of nodes with no incoming dependencies.
        execution_order: Topologically sorted node order (None if has cycles).
        mode: Recommended execution mode based on analysis.
        reasons: Human-readable reasons for mode selection.

    """

    has_cycles: bool
    has_conditional_edges: bool
    has_explicit_edges: bool
    entry_nodes: list[BaseNode[Any, Any]]
    execution_order: list[BaseNode[Any, Any]] | None
    mode: ExecutionMode
    reasons: list[str]


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

        # Node-based edge storage
        self._explicit_edges: list[Edge] = []
        self._conditional_edges: list[ConditionalEdgeConfig] = []
        self._conditional_mappings: dict[
            BaseNode[Any, Any], dict[str, BaseNode[Any, Any] | Route]
        ] = {}
        self._entry_nodes: list[BaseNode[Any, Any]] | None = None

        # Node lookup dictionaries
        self._nodes_by_name: dict[str, BaseNode[Any, Any]] = {}
        self._nodes_by_id: dict[int, BaseNode[Any, Any]] = {}

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
        is_interrupted: bool = False,
        interrupt_reason: str | None = None,
        interrupt_metadata: dict[str, Any] | None = None,
    ) -> CheckpointEnvelope:
        """Persist a checkpoint to the configured store.

        Args:
            checkpoint: The checkpoint to persist.
            store: The checkpoint store to use.
            is_interrupted: Whether this is an interrupt checkpoint.
            interrupt_reason: Reason for the interruption if applicable.
            interrupt_metadata: Additional interrupt metadata.

        Returns:
            The persisted checkpoint envelope with ID.

        Raises:
            CheckpointBackendError: If persistence fails.

        """
        # Local imports to avoid circular dependency
        from pydantic_flow.hitl.checkpoints.interface import CheckpointEnvelope
        from pydantic_flow.hitl.checkpoints.interface import CheckpointId
        from pydantic_flow.hitl.checkpoints.interface import RunId
        from pydantic_flow.hitl.checkpoints.interface import generate_checkpoint_id

        envelope = CheckpointEnvelope(
            id=CheckpointId(generate_checkpoint_id()),
            run_id=RunId(checkpoint.run_id),
            node_id=checkpoint.interrupted_node_id,
            checkpoint=checkpoint,
            is_interrupted=is_interrupted,
            interrupt_reason=interrupt_reason,
            interrupt_metadata=interrupt_metadata,
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

            # Smart detection: If we have a single result that's already
            # the output type, return it directly instead of trying to reconstruct
            if len(self._results) == 1:
                single_result = next(iter(self._results.values()))
                if isinstance(single_result, self._output_type):
                    return single_result  # type: ignore

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

    async def resume_from_envelope(
        self, envelope: CheckpointEnvelope, inputs: InputT
    ) -> OutputT:
        """Resume flow execution from a checkpoint envelope.

        Args:
            envelope: The checkpoint envelope to resume from.
            inputs: Original input data.

        Returns:
            Flow output.

        Raises:
            FlowError: If checkpoint is invalid or execution fails.

        """
        return await self.resume(envelope.checkpoint, inputs)

    async def resume_from_store(
        self,
        store: CheckpointStore,
        checkpoint_id: str,
        run_id: str,
        inputs: InputT,
    ) -> OutputT:
        """Resume flow execution from a checkpoint in a store.

        Args:
            store: The checkpoint store to load from.
            checkpoint_id: The checkpoint ID to load.
            run_id: The run ID for the checkpoint.
            inputs: Original input data.

        Returns:
            Flow output.

        Raises:
            FlowError: If checkpoint not found, invalid, or execution fails.

        """
        # Local imports to avoid circular dependency
        from pydantic_flow.hitl.checkpoints.interface import CheckpointId
        from pydantic_flow.hitl.checkpoints.interface import RunId

        envelope = await store.get(RunId(run_id), CheckpointId(checkpoint_id))
        if envelope is None:
            msg = f"Checkpoint {checkpoint_id} not found for run {run_id}"
            raise FlowError(msg)

        return await self.resume_from_envelope(envelope, inputs)

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
                # Update lookup dictionaries
                self._nodes_by_name[node.name] = node
                self._nodes_by_id[id(node)] = node

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

        # Telemetry: check if enabled before importing/instrumenting
        from contextlib import nullcontext

        from pydantic_flow.telemetry.setup import is_enabled

        if is_enabled():
            from pydantic_flow.telemetry.attributes import AttributeKey
            from pydantic_flow.telemetry.attributes import MetricName
            from pydantic_flow.telemetry.attributes import SpanKind
            from pydantic_flow.telemetry.helpers import create_span_async
            from pydantic_flow.telemetry.helpers import measure_duration_async
            from pydantic_flow.telemetry.helpers import record_counter

            flow_attrs: dict[str, Any] = {
                str(AttributeKey.FLOW_ID): self.flow_id,
                str(AttributeKey.RUN_ID): run_id,
                str(AttributeKey.EXECUTION_MODE): "dag",
            }
            record_counter(MetricName.FLOW_RUNS, attributes=flow_attrs)

            span_ctx = create_span_async(SpanKind.FLOW_RUN, attributes=flow_attrs)
            duration_ctx = measure_duration_async(
                MetricName.FLOW_DURATION, attributes=flow_attrs
            )
        else:
            span_ctx = nullcontext()
            duration_ctx = nullcontext()

        async with span_ctx, duration_ctx:
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

                        # Extract result using same logic as BaseNode.run()
                        if isinstance(item, ToolResult) and item.result is not None:
                            tool_result = item.result

                        # Check interrupt handlers on each progress item
                        decision = await self._check_interrupt_handlers(item)
                        if decision.should_interrupt:
                            # Store result before interrupting if we have it
                            if tool_result is not None:
                                self._results[node.name] = tool_result

                            checkpoint = self._create_checkpoint(node.name, run_id)
                            raise InterruptionRequested(
                                checkpoint=checkpoint,
                                decision=decision,
                            )

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
                    # Extract interrupt information from decision
                    interrupt_reason = getattr(e.decision, "reason", None)
                    interrupt_metadata = getattr(e.decision, "metadata", None)
                    # Convert empty dict to None
                    if interrupt_metadata is not None and not interrupt_metadata:
                        interrupt_metadata = None

                    envelope = await self._persist_checkpoint(
                        checkpoint=e.checkpoint,
                        store=config.checkpoint_store,
                        is_interrupted=True,
                        interrupt_reason=interrupt_reason,
                        interrupt_metadata=interrupt_metadata,
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

    def _resolve_node_reference(
        self, node_ref: BaseNode[Any, Any] | str, param_name: str
    ) -> BaseNode[Any, Any]:
        """Resolve a node reference to a node object.

        Args:
            node_ref: Node object or string name to resolve.
            param_name: Parameter name for error messages.

        Returns:
            Resolved node object.

        Raises:
            ValueError: If node not found in flow.

        """
        if isinstance(node_ref, BaseNode):
            # Verify node is in the flow
            if node_ref not in self.nodes:
                msg = f"{param_name}: Node '{node_ref.name}' not in flow"
                raise ValueError(msg)
            return node_ref

        # String reference - look up by name
        if node_ref not in self._nodes_by_name:
            available = sorted(self._nodes_by_name.keys())
            msg = (
                f"{param_name}: Unknown node name '{node_ref}'. "
                f"Available nodes: {available}"
            )
            raise ValueError(msg)
        return self._nodes_by_name[node_ref]

    def add_edge(
        self,
        from_node: BaseNode[Any, Any] | str,
        to_node: BaseNode[Any, Any] | str,
    ) -> None:
        """Add a static edge between two nodes.

        Args:
            from_node: Source node object.
            to_node: Target node object.

        Raises:
            ValueError: If node not found in flow.

        """
        # Resolve to node objects
        source = self._resolve_node_reference(from_node, "from_node")
        target = self._resolve_node_reference(to_node, "to_node")

        # Store edge
        self._explicit_edges.append(Edge(source=source, target=target))

    def add_conditional_edges(
        self,
        from_node: BaseNode[Any, Any] | str,
        router: Callable[[BaseModel], Any],
        mapping: dict[Any, BaseNode[Any, Any] | str | Route] | None = None,
    ) -> None:
        """Add conditional routing edges from a node.

        Args:
            from_node: Source node object.
            router: Function that receives state and returns routing target(s).
                   Can return Route.END, node object, or list of any.
            mapping: Optional dict to map router string outcomes to nodes.
                   Keys are router return values, values are target nodes or Route.END.

        """
        # Resolve source node
        source = self._resolve_node_reference(from_node, "from_node")

        # Store conditional edge
        self._conditional_edges.append(
            ConditionalEdgeConfig(source=source, router=router)
        )

        # Process mapping if provided
        if mapping:
            resolved_mapping: dict[str, BaseNode[Any, Any] | Route] = {}
            for key, value in mapping.items():
                # Handle Route.END specially - it's a terminal value, not a node
                if isinstance(value, Route):
                    resolved_mapping[str(key)] = value
                else:
                    target_node = self._resolve_node_reference(value, f"mapping[{key}]")
                    resolved_mapping[str(key)] = target_node
            self._conditional_mappings[source] = resolved_mapping

    def set_entry_nodes(self, *nodes: BaseNode[Any, Any] | str) -> None:
        """Set the entry nodes for loop-capable execution.

        Args:
            *nodes: Node objects to execute first.

        Raises:
            ValueError: If no nodes specified or if any node name doesn't exist.

        """
        if not nodes:
            msg = "Must specify at least one entry node"
            raise ValueError(msg)

        # Resolve all node references
        resolved_nodes: list[BaseNode[Any, Any]] = []
        for i, node_ref in enumerate(nodes):
            resolved = self._resolve_node_reference(node_ref, f"node[{i}]")
            resolved_nodes.append(resolved)

        # Store entry nodes
        self._entry_nodes = resolved_nodes

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
        # Analyze graph structure
        analysis: GraphAnalysis | None = None

        # Determine which engine to use
        if mode == ExecutionMode.AUTO:
            analysis = self._analyze_graph()
            use_stepper = analysis.mode == ExecutionMode.STEPPER
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
            # Build edge dictionary from node-based edges for stepper
            edges_dict: dict[str, list[str]] = {}
            for edge in self._explicit_edges:
                if edge.source.name not in edges_dict:
                    edges_dict[edge.source.name] = []
                edges_dict[edge.source.name].append(edge.target.name)

            # Build ConditionalEdge list from node-based conditional edges
            conditional_edges_list: list[ConditionalEdge[Any]] = []
            for cond_edge in self._conditional_edges:
                # Get mapping if exists
                mapping = None
                if cond_edge.source in self._conditional_mappings:
                    node_mapping = self._conditional_mappings[cond_edge.source]
                    # Convert node mapping to string mapping
                    mapping = {}
                    for key, value in node_mapping.items():
                        if isinstance(value, Route):
                            mapping[key] = value.value
                        else:
                            mapping[key] = value.name

                conditional_edges_list.append(
                    ConditionalEdge(cond_edge.source.name, cond_edge.router, mapping)
                )

            entry_node_names = [
                n.name
                for n in (
                    self._entry_nodes
                    if self._entry_nodes
                    else self._infer_entry_nodes()
                )
            ]
            engine_config = EngineConfig(
                nodes=self.nodes,
                edges=edges_dict,
                conditional_edges=conditional_edges_list,
                entry_nodes=entry_node_names,
                input_type=self._input_type,
                output_type=self._output_type,
                cache_backend=self._cache_backend,
                default_cache_policy=self._default_cache_policy,
            )
            engine = StepperEngine(engine_config)
            return CompiledFlow(
                flow=self,
                engine=engine,
                use_stepper=True,
                analysis=analysis,
            )

        self._calculate_execution_order()
        return CompiledFlow(
            flow=self,
            use_stepper=False,
            analysis=analysis,
        )

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
            adj[node.name] = []

        for edge in self._explicit_edges:
            adj[edge.source.name].append(edge.target.name)

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

    def _infer_entry_nodes(self) -> list[BaseNode[Any, Any]]:
        """Infer entry nodes from nodes with no dependencies."""
        entry = []
        for node in self.nodes:
            deps = getattr(node, "dependencies", [])
            if not deps:
                entry.append(node)
        return entry if entry else [self.nodes[0]] if self.nodes else []

    def _analyze_graph(self) -> GraphAnalysis:
        """Analyze flow graph structure to determine optimal execution engine.

        Returns:
            GraphAnalysis with details about graph structure and recommended mode.

        """
        reasons: list[str] = []

        # Check for conditional edges
        has_conditional = len(self._conditional_edges) > 0
        if has_conditional:
            reasons.append("Flow contains conditional routing edges")

        # Check for explicit edges (indicates potential cycles)
        has_explicit = len(self._explicit_edges) > 0
        if has_explicit:
            reasons.append("Flow contains explicit edges")

        # Detect cycles
        has_cycles = self._detect_cycles_efficiently()
        if has_cycles:
            reasons.append("Flow contains cycles")

        # Find entry nodes
        if self._entry_nodes:
            entry_nodes = self._entry_nodes
        else:
            # Infer from nodes with no dependencies
            entry_nodes = []
            nodes_with_deps = set()

            # Check implicit dependencies
            for node in self.nodes:
                if has_input_dependency(node):
                    nodes_with_deps.add(node.input.node)
                if has_multiple_inputs(node):
                    for inp in node.inputs:
                        nodes_with_deps.add(inp.node)

            # Check explicit edges
            for edge in self._explicit_edges:
                nodes_with_deps.add(edge.target)

            # Entry nodes are those with no incoming edges
            entry_nodes = [n for n in self.nodes if n not in nodes_with_deps]

            if not entry_nodes and self.nodes:
                entry_nodes = [self.nodes[0]]

        # Try topological sort (only possible if no cycles)
        execution_order = None
        if not has_cycles and not has_conditional:
            try:
                self._calculate_execution_order()
                execution_order = self._execution_order.copy()
            except CyclicDependencyError:
                pass

        # Determine recommended mode
        if has_conditional or has_cycles:
            mode = ExecutionMode.STEPPER
            if not reasons:
                reasons.append("Complex control flow detected")
        else:
            mode = ExecutionMode.DAG
            if not reasons:
                reasons.append("Simple acyclic workflow")

        return GraphAnalysis(
            has_cycles=has_cycles,
            has_conditional_edges=has_conditional,
            has_explicit_edges=has_explicit,
            entry_nodes=entry_nodes,
            execution_order=execution_order,
            mode=mode,
            reasons=reasons,
        )

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
        analysis: GraphAnalysis | None = None,
    ) -> None:
        """Initialize compiled flow.

        Args:
            flow: Source flow (for DAG mode).
            engine: Stepper engine (for STEPPER mode).
            use_stepper: Whether using stepper engine.
            analysis: Graph analysis results (if available).

        """
        self._flow = flow
        self._engine = engine
        self._use_stepper = use_stepper
        self._analysis = analysis

        # Legacy attributes for backward compatibility
        self.flow = flow
        self.engine = engine
        self.use_stepper = use_stepper

    def explain(self) -> str:
        """Explain the execution engine selection.

        Returns:
            Human-readable explanation of why this engine was selected.

        """
        if self._analysis is None:
            if self._use_stepper:
                return "Execution Engine: STEPPER\nReason: Explicitly requested"
            return "Execution Engine: DAG\nReason: Explicitly requested"

        lines = [
            f"Execution Engine: {self._analysis.mode.value.upper()}",
            "Reasons:",
        ]
        for reason in self._analysis.reasons:
            lines.append(f"  - {reason}")

        lines.append("\nGraph Structure:")
        lines.append(f"  - Nodes: {len(self._flow.nodes) if self._flow else 'N/A'}")
        lines.append(f"  - Has cycles: {self._analysis.has_cycles}")
        lines.append(
            f"  - Has conditional edges: {self._analysis.has_conditional_edges}"
        )
        lines.append(f"  - Has explicit edges: {self._analysis.has_explicit_edges}")
        lines.append(f"  - Entry nodes: {[n.name for n in self._analysis.entry_nodes]}")

        return "\n".join(lines)

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
            from contextlib import nullcontext

            from pydantic_flow.telemetry.setup import is_enabled

            # Prepare config
            config = config or RunConfig()
            run_id = config.run_id or str(uuid.uuid4())

            # Telemetry: check if enabled before importing/instrumenting
            if is_enabled():
                from pydantic_flow.telemetry.attributes import AttributeKey
                from pydantic_flow.telemetry.attributes import MetricName
                from pydantic_flow.telemetry.attributes import SpanKind
                from pydantic_flow.telemetry.helpers import create_span_async
                from pydantic_flow.telemetry.helpers import measure_duration_async
                from pydantic_flow.telemetry.helpers import record_counter

                flow_attrs: dict[str, Any] = {
                    str(AttributeKey.RUN_ID): run_id,
                    str(AttributeKey.EXECUTION_MODE): "stepper",
                }
                if self.flow is not None:
                    flow_attrs[str(AttributeKey.FLOW_ID)] = self.flow.flow_id

                record_counter(MetricName.FLOW_RUNS, attributes=flow_attrs)

                span_ctx = create_span_async(SpanKind.FLOW_RUN, attributes=flow_attrs)
                duration_ctx = measure_duration_async(
                    MetricName.FLOW_DURATION, attributes=flow_attrs
                )
            else:
                span_ctx = nullcontext()
                duration_ctx = nullcontext()

            async with span_ctx, duration_ctx:
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
