"""Streaming-native PromptNode for LLM operations."""

from collections.abc import AsyncIterator
from typing import Any
import uuid

from pydantic import BaseModel
from pydantic_ai import Agent

from pydantic_flow.nodes.base import NodeWithInput
from pydantic_flow.streaming.events import NonFatalError
from pydantic_flow.streaming.events import ProgressItem
from pydantic_flow.streaming.events import StreamEnd
from pydantic_flow.streaming.events import StreamStart
from pydantic_flow.streaming.events import TokenChunk
from pydantic_flow.streaming.observers import observe_agent_stream


class AgentNode[InputModel: BaseModel, OutputT](NodeWithInput[InputModel, OutputT]):
    """A streaming-native node that uses a pydantic-ai Agent.

    This node integrates user-supplied pydantic-ai agents with our streaming
    infrastructure, yielding tokens and progress items while the agent runs.
    """

    def __init__(
        self,
        agent: Agent[Any, OutputT],
        prompt_template: str | None = None,
        *,
        input: Any = None,
        name: str | None = None,
        run_id: str | None = None,
    ) -> None:
        """Initialize an AgentNode.

        Args:
            agent: The pydantic-ai Agent instance to use.
            prompt_template: Optional prompt template string. Uses {field}
                           syntax for variable interpolation from input.
            input: Optional input from another node's output.
            name: Optional unique identifier for this node.
            run_id: Optional run identifier for tracking execution.

        """
        super().__init__(input, name, run_id)
        self.agent = agent
        self.prompt_template = prompt_template or ""

    async def astream(self, input_data: InputModel) -> AsyncIterator[ProgressItem]:
        """Stream progress items while executing the LLM call.

        Yields:
            StreamStart, TokenChunk items during generation, and StreamEnd
            with the final result.

        """
        # Format prompt from input data
        prompt = self._format_prompt(input_data)

        # Use observer to translate agent stream to our progress items
        async for item in observe_agent_stream(
            self.agent,
            prompt,
            run_id=self.run_id or str(uuid.uuid4()),
            node_id=self.name,
        ):
            yield item

    def _format_prompt(self, input_data: InputModel) -> str:
        """Format the prompt template with input data.

        Args:
            input_data: The input model instance.

        Returns:
            Formatted prompt string.

        """
        if not self.prompt_template:
            # No template, use input directly
            if hasattr(input_data, "model_dump_json"):
                return input_data.model_dump_json()
            return str(input_data)

        # Format template with input fields
        return self.prompt_template.format(**input_data.model_dump())


class LLMNode[InputModel: BaseModel, OutputModel: BaseModel](
    NodeWithInput[InputModel, OutputModel]
):
    """A streaming-native LLM node with structured output.

    This node wraps a pydantic-ai agent and provides streaming of both
    tokens and partial structured fields during generation.
    """

    def __init__(
        self,
        agent: Agent[Any, OutputModel],
        prompt_template: str,
        *,
        input: Any = None,
        name: str | None = None,
        run_id: str | None = None,
    ) -> None:
        """Initialize an LLMNode.

        Args:
            agent: The pydantic-ai Agent instance configured for structured output.
            prompt_template: Prompt template string with {field} placeholders.
            input: Optional input from another node's output.
            name: Optional unique identifier for this node.
            run_id: Optional run identifier for tracking execution.

        """
        super().__init__(input, name, run_id)
        self.agent = agent
        self.prompt_template = prompt_template

    async def astream(self, input_data: InputModel) -> AsyncIterator[ProgressItem]:
        """Stream progress items including tokens and partial fields.

        Yields:
            StreamStart, TokenChunk during generation, and StreamEnd with
            validated structured output.

        """
        prompt = self.prompt_template.format(**input_data.model_dump())
        actual_run_id = self.run_id or str(uuid.uuid4())

        yield StreamStart(
            run_id=actual_run_id,
            node_id=self.name,
            input_preview={"prompt": prompt[:100]},
        )

        try:
            # Stream from agent
            async with self.agent.run_stream(prompt) as stream:
                token_index = 0
                async for chunk in stream.stream_text():
                    yield TokenChunk(
                        text=chunk,
                        token_index=token_index,
                        run_id=actual_run_id,
                        node_id=self.name,
                    )
                    token_index += 1

                # Get final structured result
                result = await stream.get_output()

            # Emit end with result preview
            result_preview = None
            if hasattr(result, "model_dump"):
                result_preview = result.model_dump()

            yield StreamEnd(
                run_id=actual_run_id,
                node_id=self.name,
                result_preview=result_preview,
            )

        except Exception as e:
            yield NonFatalError(
                message=f"LLM execution failed: {e}",
                recoverable=False,
                run_id=actual_run_id,
                node_id=self.name,
            )
            raise
