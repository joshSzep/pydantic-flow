"""MergePromptNode implementation for multi-input prompt generation."""

from collections.abc import AsyncIterator
from typing import Any

from pydantic_flow.nodes.base import MergeNode
from pydantic_flow.nodes.base import NodeOutput
from pydantic_flow.nodes.prompt import PromptConfig
from pydantic_flow.streaming.events import ProgressItem
from pydantic_flow.streaming.events import StreamStart


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

    async def astream(self, input_data: tuple[Any, ...]) -> AsyncIterator[ProgressItem]:
        """Stream progress items while executing the merge prompt.

        Yields:
            StreamStart and raises NotImplementedError.

        """
        run_id = self.run_id or ""
        node_id = self.name

        yield StreamStart(run_id=run_id, node_id=node_id)

        msg = (
            "MergePromptNode LLM integration not yet implemented. "
            "Use MergeToolNode with a custom LLM wrapper function instead."
        )
        raise NotImplementedError(msg)
