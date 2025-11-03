"""FlowNode implementation for composable sub-flows."""

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING
from typing import Any

from pydantic import BaseModel

from pydantic_flow.memory import ConversationMemory
from pydantic_flow.memory import MemoryMode
from pydantic_flow.memory import MemoryProtocol
from pydantic_flow.memory import ReadOnlyConversationMemory
from pydantic_flow.memory import _active_flow_memory
from pydantic_flow.nodes.base import BaseNode
from pydantic_flow.nodes.base import NodeOutput
from pydantic_flow.nodes.base import NodeWithInput
from pydantic_flow.streaming.base import ProgressItem
from pydantic_flow.streaming.core_events import FlowResult
from pydantic_flow.streaming.core_events import GenericResult
from pydantic_flow.streaming.core_events import StreamEnd
from pydantic_flow.streaming.core_events import StreamStart
from pydantic_flow.streaming.tool_events import ToolResult

if TYPE_CHECKING:
    from pydantic_flow.flow.flow import Flow


class FlowNode[InputModel: BaseModel, OutputModel: BaseModel](
    NodeWithInput[InputModel, OutputModel]
):
    """A node that wraps a Flow, enabling sub-flows within larger workflows.

    This node allows for hierarchical composition of flows, where a complete
    Flow can be used as a single node within another Flow. This enables
    building complex workflows from simpler, reusable sub-flows.
    """

    def __init__(
        self,
        flow: Flow[InputModel, OutputModel],
        *,
        input: NodeOutput[InputModel] | None = None,
        name: str | None = None,
        memory_mode: MemoryMode = MemoryMode.SHARED,
        seed_isolated_memory: bool = False,
    ) -> None:
        """Initialize a FlowNode with a wrapped Flow.

        Args:
            flow: The Flow to wrap as a node. The flow's input and output types
                 must match the FlowNode's type parameters.
            input: Optional input from another node's output
            name: Optional unique identifier for this node. If not provided,
                 will use the format "FlowNode_{flow_repr}"
            memory_mode: How to handle conversation memory for the sub-flow.
                - SHARED (default): Sub-flow uses parent's memory directly
                - ISOLATED: Sub-flow gets separate memory
                - READONLY: Sub-flow has read-only access to parent memory
            seed_isolated_memory: If True and mode is ISOLATED, seed the
                sub-flow's memory with parent's message history for context.
                Default False.

        """
        # Generate a meaningful default name that includes the wrapped flow info
        if name is None:
            flow_repr = repr(flow)
            name = f"FlowNode_{flow_repr}"

        super().__init__(input, name)
        self.flow = flow
        self.memory_mode = memory_mode
        self.seed_isolated_memory = seed_isolated_memory

    async def astream(self, input_data: InputModel) -> AsyncIterator[ProgressItem]:
        """Stream progress items while executing the wrapped flow.

        Handles memory propagation based on memory_mode:
        - SHARED: Sub-flow uses parent's memory directly
        - ISOLATED: Sub-flow gets new memory (optionally seeded)
        - READONLY: Sub-flow gets read-only wrapper of parent memory

        Yields:
            StreamStart, progress from the wrapped flow, and StreamEnd.

        """
        run_id = self.run_id or ""
        node_id = self.name

        yield StreamStart(run_id=run_id, node_id=node_id)

        # Get parent memory from context
        parent_memory = _active_flow_memory.get()

        # Setup memory for sub-flow based on mode
        sub_flow_memory: MemoryProtocol | None = None
        memory_token = None

        if parent_memory is not None:
            if self.memory_mode == MemoryMode.SHARED:
                # SHARED: Use parent memory directly
                sub_flow_memory = parent_memory
            elif self.memory_mode == MemoryMode.ISOLATED:
                # ISOLATED: Create new memory, optionally seeded
                if self.seed_isolated_memory:
                    # Seed with parent's message history
                    sub_flow_memory = parent_memory.copy()
                else:
                    # Start with empty memory
                    sub_flow_memory = ConversationMemory()
            elif self.memory_mode == MemoryMode.READONLY:
                # READONLY: Wrap parent memory for read-only access
                if isinstance(parent_memory, ConversationMemory):
                    sub_flow_memory = ReadOnlyConversationMemory(parent_memory)
                else:
                    # If parent is already ReadOnly, use its underlying memory
                    sub_flow_memory = ReadOnlyConversationMemory(
                        parent_memory._memory  # type: ignore[attr-defined]
                    )

            # Set sub-flow memory in context
            if sub_flow_memory is not None:
                memory_token = _active_flow_memory.set(sub_flow_memory)

        try:
            # Stream from the flow
            result = None
            result_model = None
            async for item in self.flow.astream(input_data):  # type: ignore
                # Handle FlowResult from wrapped flow
                if isinstance(item, FlowResult):
                    result = item.result
                    if isinstance(result, BaseModel):
                        result_model = result
                    else:
                        result_model = GenericResult(value=result)
                # Don't forward StreamStart/StreamEnd from wrapped flow
                elif isinstance(item, (StreamStart, StreamEnd)):
                    continue
                elif isinstance(item, ToolResult) and item.result:
                    # Capture actual result if available
                    result = item.result
                    if isinstance(result, BaseModel):
                        result_model = result
                    else:
                        result_model = GenericResult(value=result)
                else:
                    # Forward other progress items
                    yield item

            # Emit ToolResult with actual result if we have it
            if result is not None:
                yield ToolResult(
                    run_id=run_id,
                    node_id=node_id,
                    tool_name="flow",
                    call_id="",
                    result=result,
                    error=None,
                )

            # Emit our own StreamEnd with the result
            yield StreamEnd(run_id=run_id, node_id=node_id, result=result_model)
        finally:
            # Restore parent memory context
            if memory_token is not None:
                _active_flow_memory.reset(memory_token)

    @property
    def dependencies(self) -> list[BaseNode[Any, Any]]:
        """Get the list of nodes this FlowNode depends on.

        Returns:
            List containing the input node if this FlowNode takes input from
            another node, otherwise an empty list.

        Note:
            The wrapped flow's internal dependencies are not exposed here
            since they are encapsulated within the flow execution.

        """
        return super().dependencies

    def __repr__(self) -> str:
        """Return a string representation of the FlowNode."""
        return f"FlowNode(name='{self.name}', flow={self.flow!r})"
