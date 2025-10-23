"""MergePromptNode implementation for multi-input prompt generation."""

from typing import Any

from pydantic_flow.nodes.base import MergeNode
from pydantic_flow.nodes.base import NodeOutput
from pydantic_flow.nodes.prompt import PromptConfig


class MergePromptNode[*InputTs, OutputT](MergeNode[*InputTs, OutputT]):
    r"""A prompt node that merges multiple inputs before calling an LLM.

    This node enables patterns where multiple upstream outputs need to be
    combined into a single prompt for LLM processing.

    Example:
        research_node = ToolNode[Input, Research](...)
        analysis_node = ToolNode[Input, Analysis](...)

        # Combine both into a single prompt
        merge_prompt = MergePromptNode[Research, Analysis, str](
            inputs=(research_node.output, analysis_node.output),
            prompt="Summarize research: {research}\nWith analysis: {analysis}",
            name="summary_prompt"
        )

    """

    def __init__(
        self,
        prompt: str,
        *,
        inputs: tuple[NodeOutput[Any], ...],
        model: str | None = None,
        config: PromptConfig | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize a MergePromptNode.

        Args:
            prompt: Prompt template string. Can reference inputs by index
                   (e.g., {0}, {1}) or by providing a custom format function.
            inputs: Tuple of NodeOutput references from upstream nodes
            model: Optional model identifier (e.g., "openai:gpt-4")
            config: Optional prompt configuration
            name: Optional unique identifier for this node

        """
        super().__init__(inputs, name)
        self.prompt = prompt
        self.model = model
        self.config = config or PromptConfig()

    async def run(self, input_data: tuple[Any, ...]) -> OutputT:
        """Execute the prompt with merged inputs.

        Args:
            input_data: Tuple of data from all dependency nodes

        Returns:
            The LLM's response

        Raises:
            NotImplementedError: LLM integration not yet implemented

        """
        msg = (
            "MergePromptNode LLM integration not yet implemented. "
            "Use MergeToolNode with a custom LLM wrapper function instead."
        )
        raise NotImplementedError(msg)
