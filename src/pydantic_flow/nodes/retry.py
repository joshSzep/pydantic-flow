"""RetryNode implementation for automatic retry logic."""

from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel

from pydantic_flow.cache.base import CachePolicy
from pydantic_flow.nodes.base import BaseNode
from pydantic_flow.streaming.base import ProgressItem
from pydantic_flow.streaming.core_events import StreamEnd
from pydantic_flow.streaming.core_events import StreamStart
from pydantic_flow.streaming.system_events import NonFatalError


class RetryNode[OutputModel: BaseModel](BaseNode[Any, OutputModel]):
    """A node that wraps another node and retries on failure.

    This node provides resilience by retrying operations that may fail intermittently.
    """

    def __init__(
        self,
        wrapped_node: BaseNode[Any, OutputModel],
        *,
        max_retries: int = 3,
        name: str | None = None,
        cache_policy: CachePolicy | None = None,
    ) -> None:
        """Initialize a RetryNode.

        Args:
            wrapped_node: The node to wrap with retry logic
            max_retries: Maximum number of retry attempts
            name: Optional unique identifier for this node
            cache_policy: Optional cache policy for this node

        """
        super().__init__(wrapped_node._inputs, name, cache_policy=cache_policy)
        self.wrapped_node = wrapped_node
        self.max_retries = max_retries

    @property
    def dependencies(self) -> list[Any]:
        """Get the list of nodes this node depends on."""
        return self.wrapped_node.dependencies

    async def astream(self, input_data: Any) -> AsyncIterator[ProgressItem]:
        """Stream progress items while executing with retry logic.

        Yields:
            StreamStart, progress from wrapped node (with retries on failure),
            and StreamEnd.

        """
        run_id = self.run_id or ""
        node_id = self.name

        yield StreamStart(run_id=run_id, node_id=node_id)

        from pydantic_flow.streaming.tool_events import ToolResult  # noqa: PLC0415

        result = None

        for attempt in range(self.max_retries + 1):
            try:
                # Stream from the wrapped node
                async for item in self.wrapped_node.astream(input_data):
                    # Don't forward StreamStart/StreamEnd from wrapped node
                    if isinstance(item, StreamStart):
                        continue
                    elif isinstance(item, StreamEnd):
                        # Capture result on success
                        result = item.result
                    elif isinstance(item, ToolResult) and item.result is not None:
                        # Capture actual result
                        result = item.result
                        # Forward ToolResult
                        yield item
                    else:
                        # Forward other progress items
                        yield item

                # If we get here, wrapped node succeeded
                break

            except Exception as e:
                if attempt < self.max_retries:
                    # Emit retry warning
                    yield NonFatalError(
                        run_id=run_id,
                        node_id=node_id,
                        message=f"Attempt {attempt + 1} failed, retrying: {e}",
                        recoverable=True,
                    )
                    continue
                else:
                    # Final attempt failed
                    yield NonFatalError(
                        run_id=run_id,
                        node_id=node_id,
                        message=f"All {self.max_retries + 1} attempts failed: {e}",
                        recoverable=False,
                    )
                    raise

        # Yield ToolResult if we have it (and didn't already forward it)
        if result is not None:
            # Check if we already yielded it by checking if wrapped node forwarded it
            # For simplicity, just yield it - duplicate ToolResults are ok
            pass  # Already forwarded above

        # Emit our own StreamEnd with the result
        yield StreamEnd(run_id=run_id, node_id=node_id, result=result)
