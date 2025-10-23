"""RetryNode implementation for automatic retry logic."""

from typing import Any

from pydantic import BaseModel

from pydantic_flow.nodes.base import NodeWithInput


class RetryNode[OutputModel: BaseModel](NodeWithInput[Any, OutputModel]):
    """A node that wraps another node and retries on failure.

    This node provides resilience by retrying operations that may fail intermittently.
    """

    def __init__(
        self,
        wrapped_node: NodeWithInput[Any, OutputModel],
        *,
        max_retries: int = 3,
        name: str | None = None,
    ) -> None:
        """Initialize a RetryNode.

        Args:
            wrapped_node: The node to wrap with retry logic
            max_retries: Maximum number of retry attempts
            name: Optional unique identifier for this node

        """
        super().__init__(wrapped_node.input, name)
        self.wrapped_node = wrapped_node
        self.max_retries = max_retries

    @property
    def dependencies(self) -> list[Any]:
        """Get the list of nodes this node depends on."""
        return self.wrapped_node.dependencies

    async def run(self, input_data: Any) -> OutputModel:
        """Execute the wrapped node with retry logic.

        Args:
            input_data: The input data for this node

        Returns:
            The output from the wrapped node

        Raises:
            Exception: If all retry attempts fail

        """
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return await self.wrapped_node.run(input_data)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    continue
                break

        # If we get here, all retries failed
        msg = "Retry node failed with unknown error"
        raise last_exception or Exception(msg)
