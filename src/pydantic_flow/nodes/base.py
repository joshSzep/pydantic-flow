"""Core node abstractions for the pydantic-flow framework.

This module provides the foundational building blocks for creating type-safe,
composable AI workflows using Pydantic models with streaming-native execution.

BREAKING CHANGE: Added HITL (Human-in-the-Loop) interrupt handler support to BaseNode.
"""

from abc import ABC
from abc import abstractmethod
from collections.abc import AsyncIterator
from typing import Any
from typing import Protocol
from typing import cast

from pydantic import BaseModel

from pydantic_flow.core.errors import HandlerPriority
from pydantic_flow.core.errors import InterruptHandlerRegistration
from pydantic_flow.streaming.events import InterruptCallback
from pydantic_flow.streaming.events import InterruptDecision
from pydantic_flow.streaming.events import ProgressItem
from pydantic_flow.streaming.events import StreamEnd
from pydantic_flow.streaming.events import StreamStart
from pydantic_flow.streaming.events import ToolResult


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

    Nodes are streaming-native: the primary interface is astream() which
    yields progress items, with run() as a convenience wrapper that assembles
    the final result.
    """

    def __init__(self, name: str | None = None, run_id: str | None = None) -> None:
        """Initialize the base node.

        Args:
            name: Optional unique identifier for this node. If not provided,
                  will be auto-generated based on the class name.
            run_id: Optional run identifier for tracking execution.

        """
        self.name = name or f"{self.__class__.__name__}_{id(self):x}"
        self.run_id = run_id
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

    @abstractmethod
    async def astream(self, input_data: InputT) -> AsyncIterator[ProgressItem]:
        """Stream progress items while executing the node's logic.

        This is the primary interface for node execution. It yields a
        coherent sequence: start, useful progress, clean end.

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
        """Execute the node and return the final validated result.

        This is a convenience method that consumes the astream() and
        assembles the final output.

        Args:
            input_data: The input data for this node

        Returns:
            The final validated output data

        """
        from contextlib import nullcontext

        from pydantic_flow.telemetry.setup import is_enabled

        # Telemetry: check if enabled before importing/instrumenting
        if is_enabled():
            from pydantic_flow.telemetry.attributes import AttributeKey
            from pydantic_flow.telemetry.attributes import MetricName
            from pydantic_flow.telemetry.attributes import SpanKind
            from pydantic_flow.telemetry.helpers import create_span_async
            from pydantic_flow.telemetry.helpers import measure_duration_async
            from pydantic_flow.telemetry.helpers import record_counter

            node_attrs: dict[str, Any] = {
                str(AttributeKey.NODE_ID): self.name,
                str(AttributeKey.NODE_TYPE): self.__class__.__name__,
            }
            if self.run_id:
                node_attrs[str(AttributeKey.RUN_ID)] = self.run_id

            record_counter(MetricName.NODE_EXECUTIONS, attributes=node_attrs)

            span_ctx = create_span_async(SpanKind.NODE_RUN, attributes=node_attrs)
            duration_ctx = measure_duration_async(
                MetricName.NODE_DURATION, attributes=node_attrs
            )
        else:
            span_ctx = nullcontext()
            duration_ctx = nullcontext()

        async with span_ctx, duration_ctx:
            final_result: OutputT | None = None
            tool_result: Any = None

            async for item in self.astream(input_data):
                # Record stream events as span events
                self._record_stream_event(item)

                # Try to extract result from ToolResult first (has the actual object)
                if isinstance(item, ToolResult) and item.result is not None:
                    tool_result = item.result
                # StreamEnd carries the final result preview as fallback
                elif isinstance(item, StreamEnd) and item.result_preview:
                    # Reconstruct the output from the preview
                    # Try Pydantic validation first
                    try:
                        if hasattr(self._output_type, "model_validate"):
                            final_result = self._output_type.model_validate(  # type: ignore
                                item.result_preview
                            )
                        else:
                            final_result = item.result_preview  # type: ignore
                    except Exception:
                        # Fall back to direct assignment
                        final_result = item.result_preview  # type: ignore

            # Prefer the actual result from ToolResult if available
            if tool_result is not None:
                final_result = tool_result  # type: ignore

            if final_result is None:
                msg = f"Node {self.name} did not produce a result"
                raise RuntimeError(msg)

            return final_result

    def _record_stream_event(self, item: ProgressItem) -> None:
        """Record a streaming progress item as a span event.

        Args:
            item: The progress item to record.

        """
        from pydantic_flow.streaming.events import CacheHit
        from pydantic_flow.streaming.events import CacheMiss
        from pydantic_flow.streaming.events import CacheWrite
        from pydantic_flow.streaming.events import StreamStart
        from pydantic_flow.streaming.events import TokenChunk
        from pydantic_flow.streaming.events import ToolCall
        from pydantic_flow.streaming.events import ToolResult
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


class NodeWithInput[InputT, OutputT](BaseNode[InputT, OutputT]):
    """Base class for nodes that take input from other nodes.

    This class handles the common pattern of nodes that depend on
    the output of other nodes in the workflow.
    """

    def __init__(
        self,
        input: NodeOutput[InputT] | None = None,
        name: str | None = None,
        run_id: str | None = None,
    ) -> None:
        """Initialize a node with optional input dependency.

        Args:
            input: Optional input from another node's output
            name: Optional unique identifier for this node
            run_id: Optional run identifier for tracking execution

        """
        super().__init__(name, run_id)
        self.input = input

    @property
    def dependencies(self) -> list[BaseNode[Any, Any]]:
        """Get the list of nodes this node depends on."""
        if self.input is None:
            return []
        return [self.input.node]


class MergeNode[*InputTs, OutputT](BaseNode[tuple[*InputTs], OutputT]):
    """Base class for nodes that merge multiple inputs.

    This class enables fan-in patterns where a node needs to combine
    outputs from multiple upstream nodes.

    Uses PEP 646 TypeVarTuple for arbitrary input types, allowing
    full type safety across multiple inputs.

    Example:
        MergeNode[DataA, DataB, DataC, Result] represents a node that
        takes three inputs (DataA, DataB, DataC) and produces Result.

    """

    def __init__(
        self,
        inputs: tuple[NodeOutput[Any], ...],
        name: str | None = None,
        run_id: str | None = None,
    ) -> None:
        """Initialize a merge node with multiple input dependencies.

        Args:
            inputs: Tuple of NodeOutput references from upstream nodes
            name: Optional unique identifier for this node
            run_id: Optional run identifier for tracking execution

        """
        super().__init__(name, run_id)
        self.inputs = inputs

    @property
    def dependencies(self) -> list[BaseNode[Any, Any]]:
        """Get all dependency nodes from multiple inputs."""
        return [node_output.node for node_output in self.inputs]


# Protocol classes for type safety
class NodeProtocol[InputT, OutputT](Protocol):
    """Protocol defining the interface all nodes must implement."""

    name: str
    output: NodeOutput[OutputT]

    async def astream(self, input_data: InputT) -> AsyncIterator[ProgressItem]:
        """Stream progress items during execution."""
        ...

    async def run(self, input_data: InputT) -> OutputT:
        """Execute the node's logic and return final result."""
        ...


class RunnableNode[InputT, OutputT](Protocol):
    """Protocol for nodes that can be executed."""

    name: str

    async def astream(self, input_data: InputT) -> AsyncIterator[ProgressItem]:
        """Stream progress items during execution."""
        ...

    async def run(self, input_data: InputT) -> OutputT:
        """Execute the node's logic and return final result."""
        ...
