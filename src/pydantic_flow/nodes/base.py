"""Core node abstractions for the pydantic-flow framework.

This module provides the foundational building blocks for creating type-safe,
composable AI workflows using Pydantic models with streaming-native execution.
"""

from abc import ABC
from abc import abstractmethod
from collections.abc import AsyncIterator
from typing import Any
from typing import Protocol
from typing import cast

from pydantic import BaseModel

from pydantic_flow.nodes.mixins import InterruptibleNodeMixin
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


class BaseNode[InputT, OutputT](InterruptibleNodeMixin, ABC):
    """Abstract base class for all workflow nodes.

    Nodes are streaming-native: the primary interface is astream() which
    yields progress items, with run() as a convenience wrapper that assembles
    the final result.

    Inherits from InterruptibleNodeMixin to provide HITL interrupt support.
    """

    def __init__(self, name: str | None = None, run_id: str | None = None) -> None:
        """Initialize the base node.

        Args:
            name: Optional unique identifier for this node. If not provided,
                  will be auto-generated based on the class name.
            run_id: Optional run identifier for tracking execution.

        """
        super().__init__()
        self.name = name or f"{self.__class__.__name__}_{id(self):x}"
        self.run_id = run_id
        self._output: NodeOutput[OutputT] = NodeOutput(node=self)
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

    @abstractmethod
    async def astream(self, input_data: InputT) -> AsyncIterator[ProgressItem]:
        """Stream progress items while executing the node's logic.

        This is the primary and only interface for node execution.
        Events are the fundamental unit, not nodes.

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
        self._inputs = inputs

    @property
    def inputs(self) -> tuple[NodeOutput[Any], ...]:
        """Get the tuple of input node output references."""
        return self._inputs

    @property
    def dependencies(self) -> list[BaseNode[Any, Any]]:
        """Get all dependency nodes from multiple inputs."""
        return [node_output.node for node_output in self._inputs]


# Protocol classes for type safety
class NodeProtocol[InputT, OutputT](Protocol):
    """Protocol defining the interface all nodes must implement."""

    name: str
    output: NodeOutput[OutputT]

    async def astream(self, input_data: InputT) -> AsyncIterator[ProgressItem]:
        """Stream progress items during execution."""
        ...


class RunnableNode[InputT, OutputT](Protocol):
    """Protocol for nodes that can be executed."""

    name: str

    async def astream(self, input_data: InputT) -> AsyncIterator[ProgressItem]:
        """Stream progress items during execution."""
        ...
