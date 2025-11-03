"""Flow orchestration for the pydantic-flow framework.

This module provides the Flow class that manages workflow execution
and dependency resolution using the dataflow engine.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from typing import TypeVar
import uuid

from pydantic import BaseModel

from pydantic_flow.cache.base import CacheBackend
from pydantic_flow.cache.base import CachePolicy
from pydantic_flow.checkpoints.interface import CheckpointStorageBackend
from pydantic_flow.checkpoints.types import StateSnapshot
from pydantic_flow.core.errors import FlowError
from pydantic_flow.core.routing import Route
from pydantic_flow.core.run_config import RunConfig
from pydantic_flow.engine.dataflow import DataflowEngine
from pydantic_flow.hitl.decisions import InterruptCallback
from pydantic_flow.hitl.decisions import InterruptDecision
from pydantic_flow.hitl.interrupts import HandlerPriority
from pydantic_flow.hitl.interrupts import InterruptHandlerRegistration
from pydantic_flow.memory import ConversationMemory
from pydantic_flow.memory import MemoryConfig
from pydantic_flow.nodes import BaseNode
from pydantic_flow.nodes.protocols import has_input_dependency
from pydantic_flow.nodes.protocols import has_multiple_inputs
from pydantic_flow.streaming.base import ProgressItem

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
        self._background_tasks: set[Any] = set()  # Track async checkpoint tasks
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

    async def astream(
        self, inputs: InputT, config: RunConfig | None = None
    ) -> AsyncIterator[ProgressItem]:
        """Execute the flow and stream progress items with eager dataflow execution.

        Nodes execute as soon as their dependencies are satisfied, maximizing
        parallelism and minimizing total execution time. This method uses the
        dataflow engine for optimal execution without compilation steps.

        Args:
            inputs: The input data for the flow (must match the flow's InputT type)
            config: Optional run configuration including checkpoint store

        Yields:
            ProgressItem objects from flow execution, ending with FlowResult.

        Raises:
            FlowError: If the flow execution fails
            TypeError: If the input type doesn't match the expected input_type
            InterruptionRequested: If a HITL interrupt handler requests stopping

        """
        from pydantic_flow.engine.dataflow import DataflowEngine

        # Build edge dictionary from node-based edges
        edges_dict: dict[str, list[str]] = {}

        # Add explicit edges
        for edge in self._explicit_edges:
            if edge.source.name not in edges_dict:
                edges_dict[edge.source.name] = []
            edges_dict[edge.source.name].append(edge.target.name)

        # Add implicit edges from node dependencies
        for node in self.nodes:
            deps = getattr(node, "dependencies", [])
            for dep in deps:
                if dep.name not in edges_dict:
                    edges_dict[dep.name] = []
                if node.name not in edges_dict[dep.name]:
                    edges_dict[dep.name].append(node.name)

        # Build conditional edges list
        conditional_edges_list: list[
            tuple[str, Callable[[BaseModel], Any], dict[Any, str] | None]
        ] = []
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

            conditional_edges_list.append((
                cond_edge.source.name,
                cond_edge.router,
                mapping,
            ))

        entry_node_names = [
            n.name
            for n in (
                self._entry_nodes if self._entry_nodes else self._infer_entry_nodes()
            )
        ]

        # Create dataflow engine
        from pydantic_flow.engine.dataflow import DataflowConfig

        engine = DataflowEngine(
            config=DataflowConfig(
                nodes=self.nodes,
                edges=edges_dict,
                conditional_edges=conditional_edges_list,
                entry_nodes=entry_node_names,
                input_type=self._input_type,
                output_type=self._output_type,
                cache_backend=self._cache_backend,
                default_cache_policy=self._default_cache_policy,
            )
        )

        # Execute with dataflow engine
        async for item in engine.astream(inputs, config):
            yield item

    async def astream_from_snapshot(
        self,
        snapshot: StateSnapshot,
        storage: CheckpointStorageBackend,
        config: RunConfig | None = None,
    ) -> AsyncIterator[ProgressItem]:
        """Resume from V2 StateSnapshot and stream progress.

        Works for all snapshot types: HITL interrupts, manual pauses,
        error recovery, fork points, and debugging time-travel.

        This method reconstructs the full state (if needed), restores
        the execution context, and continues from the next_frontier.

        Args:
            snapshot: StateSnapshot to resume from.
            storage: V2 checkpoint storage backend.
            config: Optional run configuration.

        Yields:
            ProgressItem objects from resumed execution, ending with FlowResult.

        Raises:
            FlowError: If resumption fails.
            TypeError: If snapshot run_id doesn't match flow configuration.

        """
        from pydantic_flow.checkpoints.reconstructor import StateReconstructor

        config = config or RunConfig()

        # Reconstruct full state if snapshot only has deltas
        if snapshot.full_state is None:
            reconstructor = StateReconstructor(storage)
            full_state = await reconstructor.reconstruct_state_at(
                run_id=snapshot.run_id, wave_number=snapshot.wave_number
            )
        else:
            full_state = snapshot.full_state

        # Restore node states
        self._results = dict(full_state)

        # Configure for resume
        resume_config = RunConfig(
            **config.model_dump(exclude={"run_id"}),
            run_id=snapshot.run_id,
        )

        # Continue execution from next_frontier - stream the results
        async for item in self.astream(
            input=self._results,  # type: ignore
            config=resume_config,
        ):
            yield item

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

    def compile(self) -> CompiledFlow[InputT, OutputT]:
        """Compile the flow into an executable form using the stepper engine.

        Returns:
            CompiledFlow instance ready for execution.

        Raises:
            FlowError: If flow structure is invalid.

        """
        # Build edge dictionary from node-based edges for stepper
        edges_dict: dict[str, list[str]] = {}

        # Add explicit edges
        for edge in self._explicit_edges:
            if edge.source.name not in edges_dict:
                edges_dict[edge.source.name] = []
            edges_dict[edge.source.name].append(edge.target.name)

        # Add implicit edges from node dependencies
        for node in self.nodes:
            deps = getattr(node, "dependencies", [])
            for dep in deps:
                if dep.name not in edges_dict:
                    edges_dict[dep.name] = []
                if node.name not in edges_dict[dep.name]:
                    edges_dict[dep.name].append(node.name)

        # Build conditional edges list from node-based conditional edges
        # DataflowEngine expects tuples of (source_name, router_fn, mapping)
        conditional_edges_tuples: list[
            tuple[str, Callable[[BaseModel], Any], dict[Any, str] | None]
        ] = []
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

            conditional_edges_tuples.append((
                cond_edge.source.name,
                cond_edge.router,
                mapping,
            ))

        entry_node_names = [
            n.name
            for n in (
                self._entry_nodes if self._entry_nodes else self._infer_entry_nodes()
            )
        ]
        from pydantic_flow.engine.dataflow import DataflowConfig

        engine = DataflowEngine(
            config=DataflowConfig(
                nodes=self.nodes,
                edges=edges_dict,
                conditional_edges=conditional_edges_tuples,
                entry_nodes=entry_node_names,
                input_type=self._input_type,
                output_type=self._output_type,
                cache_backend=self._cache_backend,
                default_cache_policy=self._default_cache_policy,
                flow_id=self.flow_id,
            )
        )
        return CompiledFlow(engine=engine)

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
    """Compiled flow ready for execution using the dataflow engine."""

    def __init__(self, engine: DataflowEngine[InputT, OutputT]) -> None:
        """Initialize compiled flow.

        Args:
            engine: Dataflow engine for execution.

        """
        self._engine = engine
        self.engine = engine  # Backward compatibility

    async def astream(
        self, inputs: InputT, config: RunConfig | None = None
    ) -> AsyncIterator[ProgressItem]:
        """Execute the compiled flow and stream progress items.

        Args:
            inputs: Input data.
            config: Optional execution configuration.

        Yields:
            ProgressItem objects from flow execution, ending with FlowResult.

        Raises:
            InterruptionRequested: If a HITL interrupt occurs.

        """
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
                str(AttributeKey.EXECUTION_MODE): "dataflow",
                str(AttributeKey.FLOW_ID): self.engine.flow_id,
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
            async for item in self.engine.astream(inputs, config):
                yield item
