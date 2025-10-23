"""PromptNode implementation for LLM-based processing."""

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from pydantic_flow.nodes.base import NodeOutput
from pydantic_flow.nodes.base import NodeWithInput


@dataclass
class PromptConfig:
    """Configuration for PromptNode."""

    model: str = "openai:gpt-4"
    system_prompt: str | None = None
    result_type: type[Any] | None = None


class PromptNode[InputModel: BaseModel, OutputT](NodeWithInput[InputModel, OutputT]):
    """A node that calls an LLM using a templated prompt.

    This node integrates with pydantic-ai to make LLM calls with structured
    input and output handling.
    """

    def __init__(
        self,
        prompt: str,
        *,
        config: PromptConfig | None = None,
        input: NodeOutput[InputModel] | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize a PromptNode.

        Args:
            prompt: The prompt template to use
            config: Configuration for the LLM (model, system prompt, etc.)
            input: Optional input from another node's output
            name: Optional unique identifier for this node

        """
        super().__init__(input, name)
        self.prompt = prompt
        self.config = config or PromptConfig()

    async def run(self, input_data: InputModel) -> OutputT:
        """Execute the LLM call with the given input.

        Args:
            input_data: The input data for this node

        Returns:
            The LLM response, either as a string or structured data

        """
        # For now, return a placeholder - actual pydantic-ai integration
        # will be added when the package is available
        formatted_prompt = self.prompt.format(**input_data.model_dump())
        # TODO: Implement actual LLM call with pydantic-ai using self.config
        return formatted_prompt  # type: ignore[return-value]
