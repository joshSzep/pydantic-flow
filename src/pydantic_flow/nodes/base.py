"""Core node abstractions for the pydantic-flow framework.

This module provides the foundational building blocks for creating type-safe,
composable AI workflows using Pydantic models with streaming-native execution.
"""

from abc import ABC
from abc import abstractmethod
from collections.abc import AsyncIterator
from typing import Any
from typing import cast

from pydantic import BaseModel

from pydantic_flow.cache.base import CachePolicy
from pydantic_flow.hitl.decisions import InterruptCallback
from pydantic_flow.hitl.decisions import InterruptDecision
from pydantic_flow.hitl.interrupts import HandlerPriority
from pydantic_flow.hitl.interrupts import InterruptHandlerRegistration
from pydantic_flow.streaming.base import ProgressItem
from pydantic_flow.streaming.core_events import StreamEnd
from pydantic_flow.streaming.core_events import StreamStart
from pydantic_flow.streaming.tool_events import ToolResult


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

    All nodes have these capabilities built-in:
    - Streaming via astream() and convenience run() method
    - Caching via cache_policy attribute
    - Interrupts via interrupt handler registration
    - Dependency tracking via dependencies property

    This eliminates the need for mixins and provides a consistent interface.
    """

    def __init__(
        self,
        name: str | None = None,
        run_id: str | None = None,
        cache_policy: CachePolicy | None = None,
    ) -> None:
        """Initialize the base node.

        Args:
            name: Optional unique identifier for this node. If not provided,
                  will be auto-generated based on the class name.
            run_id: Optional run identifier for tracking execution.
            cache_policy: Optional cache policy for this node.

        """
        self.name = name or f"{self.__class__.__name__}_{id(self):x}"
        self.run_id = run_id
        self.cache_policy = cache_policy
        self._output: NodeOutput[OutputT] = NodeOutput(node=self)
        self._interrupt_handlers: list[InterruptHandlerRegistration] = []

        # Store type information for runtime inspection
        # Find the base with generic type parameters (handles multiple inheritance)
        type_base = None
        for base in self.__class__.__orig_bases__:  # type: ignore
            if hasattr(base, "__args__") and len(base.__args__) >= 2:  # type: ignore  # noqa: PLR2004
                type_base = base
                break
        if type_base is not None:
            self._input_type: type[InputT] = type_base.__args__[0]  # type: ignore
            self._output_type: type[OutputT] = type_base.__args__[1]  # type: ignore
        else:
            # Fallback for edge cases
            self._input_type = Any  # type: ignore
            self._output_type = Any  # type: ignore

    @property
    def output(self) -> NodeOutput[OutputT]:
        """Get the typed output reference for this node."""
        return self._output

    @property
    def dependencies(self) -> list[BaseNode[Any, Any]]:
        """Get the list of nodes this node depends on.

        This is the canonical dependency interface used by Flow for
        dependency resolution and execution ordering.

        Returns:
            List of upstream nodes this node depends on. Empty list means
            this is an entry node (or has no explicit dependencies).

        """
        return []

    def register_interrupt_handler(
        self,
        callback: InterruptCallback,
        priority: int = HandlerPriority.NORMAL,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register an interrupt callback handler for this node.

        Handlers are invoked in priority order (lowest first) when
        checking for interrupts. Critical handlers (0-25) always execute.

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
        """Remove all registered interrupt handlers from this node."""
        self._interrupt_handlers.clear()

    async def _check_interrupt_handlers(self, item: ProgressItem) -> InterruptDecision:
        """Check all registered interrupt handlers for this progress item.

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

    def is_cacheable(self) -> bool:
        """Check if this node has caching enabled.

        Returns:
            True if cache_policy is set and enabled.

        """
        return self.cache_policy is not None and self.cache_policy.enabled

    @abstractmethod
    async def astream(self, input_data: InputT) -> AsyncIterator[ProgressItem]:
        """Stream progress items while executing the node's logic.

        This is the primary interface for node execution. For convenience,
        use run() to get the final result directly.

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
        """Consume the stream and return final result.

        This method provides a simpler interface for users who don't need
        streaming progress. It internally calls astream() and extracts the
        final result.

        Args:
            input_data: The input data for this node

        Returns:
            The final output value of type OutputT

        Raises:
            ValueError: If the node execution completes without producing a result

        """
        result = None
        async for item in self.astream(input_data):
            if hasattr(item, "result") and item.result is not None:
                result = item.result
        if result is None:
            msg = f"Node {self.name} did not produce a result"
            raise ValueError(msg)
        return result  # type: ignore

    def _record_stream_event(self, item: ProgressItem) -> None:
        """Record a streaming progress item as a span event.

        Args:
            item: The progress item to record.

        """
        from pydantic_flow.cache.events import CacheHit
        from pydantic_flow.cache.events import CacheMiss
        from pydantic_flow.cache.events import CacheWrite
        from pydantic_flow.streaming.core_events import StreamStart
        from pydantic_flow.streaming.core_events import TokenChunk
        from pydantic_flow.streaming.tool_events import ToolCall
        from pydantic_flow.telemetry.attributes import EventName
        from pydantic_flow.telemetry.helpers import record_span_event

        # Map stream events to span events
        event_name: str | None = None
        event_attrs: dict[str, Any] = {}

        if isinstance(item, StreamStart):
            event_name = EventName.STREAM_START
        elif isinstance(item, StreamEnd):
            event_name = EventName.STREAM_END
        elif isinstance(item, TokenChunk):
            event_name = EventName.STREAM_CHUNK
            event_attrs["token"] = item.text[:50]  # Truncate
        elif isinstance(item, CacheHit):
            event_name = EventName.CACHE_HIT
            event_attrs["key"] = item.key[:32]  # Truncate
        elif isinstance(item, CacheMiss):
            event_name = EventName.CACHE_MISS
            event_attrs["key"] = item.key[:32]
        elif isinstance(item, CacheWrite):
            event_name = EventName.CACHE_WRITE
            event_attrs["key"] = item.key[:32] if item.key else ""
        elif isinstance(item, ToolCall):
            event_name = EventName.TOOL_CALL
            event_attrs["tool"] = item.tool_name
        elif isinstance(item, ToolResult):
            event_name = EventName.TOOL_RESULT
            event_attrs["tool"] = item.tool_name

        if event_name:
            record_span_event(event_name, event_attrs)

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


class Node[InputT, OutputT](BaseNode[InputT, OutputT]):
    """Concrete base class for nodes with optional single input dependency.

    This is the primary base class for creating custom nodes. It provides
    all the built-in capabilities (caching, interrupts, streaming) with
    support for a single input dependency from another node.

    For specialized behavior, use:
    - AgentNode for LLM operations
    - FlowNode for sub-flows
    - ToolNode for function calls
    - MergeNode for fan-in patterns with multiple inputs

    Example:
        class MyCustomNode(Node[Query, Answer]):
            async def astream(self, input_data: Query):
                yield StreamStart(...)
                # Your logic here
                yield StreamEnd(result=answer)

    """

    def __init__(
        self,
        input: NodeOutput[InputT] | None = None,
        name: str | None = None,
        run_id: str | None = None,
        cache_policy: CachePolicy | None = None,
    ) -> None:
        """Initialize a node with optional input dependency.

        Args:
            input: Optional input from another node's output
            name: Optional unique identifier for this node
            run_id: Optional run identifier for tracking execution
            cache_policy: Optional cache policy for this node

        """
        super().__init__(name, run_id, cache_policy)
        self._input = input

    @property
    def input(self) -> NodeOutput[InputT] | None:
        """Get the input node output reference."""
        return self._input

    @property
    def dependencies(self) -> list[BaseNode[Any, Any]]:
        """Get the list of nodes this node depends on."""
        if self._input is None:
            return []
        return [self._input.node]


# Keep NodeWithInput as an alias for backward compatibility during migration
NodeWithInput = Node


class MergeNode[*InputTs, OutputT](BaseNode[tuple[*InputTs], OutputT]):
    """Base class for nodes that merge multiple inputs (fan-in pattern).

    This class enables fan-in patterns where a node needs to combine
    outputs from multiple upstream nodes.

    Uses PEP 646 TypeVarTuple for arbitrary input types, allowing
    full type safety across multiple inputs.

    Example:
        class CombineResults(MergeNode[DataA, DataB, Result]):
            async def astream(self, inputs: tuple[DataA, DataB]):
                data_a, data_b = inputs
                yield StreamStart(...)
                # Combine data_a and data_b
                yield StreamEnd(result=combined)

    """

    def __init__(
        self,
        inputs: tuple[NodeOutput[Any], ...],
        name: str | None = None,
        run_id: str | None = None,
        cache_policy: CachePolicy | None = None,
    ) -> None:
        """Initialize a merge node with multiple input dependencies.

        Args:
            inputs: Tuple of NodeOutput references from upstream nodes
            name: Optional unique identifier for this node
            run_id: Optional run identifier for tracking execution
            cache_policy: Optional cache policy for this node

        """
        super().__init__(name, run_id, cache_policy)
        self._inputs = inputs

    @property
    def inputs(self) -> tuple[NodeOutput[Any], ...]:
        """Get the tuple of input node output references."""
        return self._inputs

    @property
    def dependencies(self) -> list[BaseNode[Any, Any]]:
        """Get all dependency nodes from multiple inputs."""
        return [node_output.node for node_output in self._inputs]


# Remove redundant protocol classes - BaseNode as ABC is sufficient
# These added no value and violated the "clear APIs" principle
