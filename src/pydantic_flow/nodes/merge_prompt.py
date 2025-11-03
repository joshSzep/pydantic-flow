"""MergePromptNode implementation for multi-input prompt generation."""

from collections.abc import AsyncIterator
from typing import Any
import uuid

from pydantic_ai import Agent

from pydantic_flow.cache.base import CachePolicy
from pydantic_flow.checkpoints.types import RunId
from pydantic_flow.checkpoints.types import SnapshotReason
from pydantic_flow.checkpoints.types import StateSnapshot
from pydantic_flow.checkpoints.types import generate_run_id
from pydantic_flow.checkpoints.types import generate_snapshot_id
from pydantic_flow.hitl.interrupts import InterruptionRequested
from pydantic_flow.nodes.base import MergeNode
from pydantic_flow.nodes.base import NodeOutput
from pydantic_flow.nodes.prompt import PromptConfig
from pydantic_flow.streaming.base import ProgressItem
from pydantic_flow.streaming.core_events import StreamStart
from pydantic_flow.streaming.observers import observe_agent_stream
from pydantic_flow.streaming.system_events import NonFatalError


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
            prompt="Summarize research: {0}\nWith analysis: {1}",
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
        cache_policy: CachePolicy | None = None,
    ) -> None:
        """Initialize a MergePromptNode.

        Args:
            prompt: Prompt template string. Can reference inputs by index
                   (e.g., {0}, {1}) or by custom field names in the template.
            inputs: Tuple of NodeOutput references from upstream nodes
            model: Optional model identifier (e.g., "openai:gpt-4")
            config: Optional prompt configuration
            name: Optional unique identifier for this node
            cache_policy: Optional cache policy for this node

        """
        super().__init__(inputs, name, cache_policy=cache_policy)
        self.prompt = prompt
        self.model = model or (config.model if config else "test")
        self.config = config or PromptConfig()

        # Create internal pydantic-ai agent
        instructions = self.config.system_prompt or "Be helpful and concise."
        if self.config.result_type:
            self._agent = Agent(
                self.model,
                instructions=instructions,
                output_type=self.config.result_type,
            )
        else:
            self._agent = Agent(
                self.model,
                instructions=instructions,
            )  # type: ignore[assignment]

    def _format_prompt(self, input_data: tuple[Any, ...]) -> str:
        """Format the prompt template with the merged input data.

        Args:
            input_data: Tuple of inputs from upstream nodes

        Returns:
            Formatted prompt string

        """
        # Try formatting with positional arguments first
        try:
            return self.prompt.format(*input_data)
        except IndexError, KeyError:
            pass

        # Try formatting with indices as keyword arguments
        try:
            kwargs = {str(i): val for i, val in enumerate(input_data)}
            return self.prompt.format(**kwargs)
        except IndexError, KeyError:
            pass

        # Try extracting model_dump() if available and formatting with that
        try:
            merged_dict = {}
            for i, val in enumerate(input_data):
                if hasattr(val, "model_dump"):
                    merged_dict.update(val.model_dump())
                else:
                    merged_dict[str(i)] = val
            return self.prompt.format(**merged_dict)
        except Exception:
            pass

        # Fallback: concatenate inputs as strings, return raw
        return "\n\n".join(str(val) for val in input_data)

    def _create_checkpoint(self, item: ProgressItem) -> StateSnapshot:
        """Create a checkpoint for resumption.

        Args:
            item: Progress item at interruption point.

        Returns:
            StateSnapshot for resuming execution.

        """
        # Extract run_id from item if available
        run_id_str = getattr(item, "run_id", None) or ""

        return StateSnapshot(
            snapshot_id=generate_snapshot_id(),
            run_id=RunId(run_id_str) if run_id_str else generate_run_id(),
            wave_number=0,
            state_hash="",
            next_frontier=[],
            routing_ended=False,
            reason=SnapshotReason.HITL_INTERRUPT,
            interrupted_node_id=self.name,
        )

    async def astream(self, input_data: tuple[Any, ...]) -> AsyncIterator[ProgressItem]:
        """Stream progress items while executing the merge prompt.

        Yields:
            StreamStart, TokenChunk items during generation, and StreamEnd
            with the final result.

        """
        formatted_prompt = self._format_prompt(input_data)
        actual_run_id = self.run_id or str(uuid.uuid4())

        start_item = StreamStart(
            run_id=actual_run_id,
            node_id=self.name,
            input_preview={
                "prompt": formatted_prompt[:100],
                "num_inputs": len(input_data),
            },
        )
        decision = await self._check_interrupt_handlers(start_item)
        if decision.should_interrupt:
            raise InterruptionRequested(
                snapshot=self._create_checkpoint(start_item),
                decision=decision,
            )
        yield start_item

        try:
            # Use observer to translate agent stream to our progress items
            async for item in observe_agent_stream(
                self._agent,
                formatted_prompt,
                message_history=None,  # MergePromptNode doesn't use conversation memory
                run_id=actual_run_id,
                node_id=self.name,
            ):
                # Check interrupt handlers on each progress item
                decision = await self._check_interrupt_handlers(item)
                if decision.should_interrupt:
                    raise InterruptionRequested(
                        snapshot=self._create_checkpoint(item),
                        decision=decision,
                    )
                yield item

        except InterruptionRequested:
            raise
        except Exception as e:
            # Emit error and re-raise
            yield NonFatalError(
                message=f"MergePromptNode failed: {e}",
                run_id=actual_run_id,
                node_id=self.name,
                recoverable=False,
            )
            raise
