"""PromptNode implementation for LLM-based processing."""

from collections.abc import AsyncIterator
from typing import Any
import uuid

from pydantic import BaseModel
from pydantic_ai import Agent

from pydantic_flow.nodes.base import NodeOutput
from pydantic_flow.nodes.base import NodeWithInput
from pydantic_flow.streaming.events import NonFatalError
from pydantic_flow.streaming.events import ProgressItem
from pydantic_flow.streaming.events import StreamEnd
from pydantic_flow.streaming.events import StreamStart
from pydantic_flow.streaming.events import TokenChunk
from pydantic_flow.streaming.events import ToolResult


class PromptConfig(BaseModel):
    """Configuration for PromptNode."""

    model_config = {"frozen": True}

    model: str = "test"
    system_prompt: str | None = None
    result_type: type[Any] | None = None


class PromptNode[InputModel: BaseModel, OutputT](NodeWithInput[InputModel, OutputT]):
    """A streaming-native node that calls an LLM using a templated prompt.

    This node creates a pydantic-ai agent internally and provides streaming
    execution with token visibility.
    """

    def __init__(
        self,
        prompt: str,
        *,
        config: PromptConfig | None = None,
        input: NodeOutput[InputModel] | None = None,
        name: str | None = None,
        run_id: str | None = None,
    ) -> None:
        """Initialize a PromptNode.

        Args:
            prompt: The prompt template to use
            config: Configuration for the LLM (model, system prompt, etc.)
            input: Optional input from another node's output
            name: Optional unique identifier for this node
            run_id: Optional run identifier for tracking execution

        """
        super().__init__(input, name, run_id)
        self.prompt = prompt
        self.config = config or PromptConfig()

        # Create internal pydantic-ai agent
        instructions = self.config.system_prompt or "Be helpful and concise."
        if self.config.result_type:
            self._agent = Agent(
                self.config.model,
                instructions=instructions,
                output_type=self.config.result_type,
            )
        else:
            self._agent = Agent(
                self.config.model,
                instructions=instructions,
            )  # type: ignore[assignment]

    async def astream(self, input_data: InputModel) -> AsyncIterator[ProgressItem]:
        """Stream progress items while executing the LLM call.

        Yields:
            StreamStart, TokenChunk items during generation, and StreamEnd
            with the final result.

        """
        # Format prompt from input data
        formatted_prompt = self.prompt.format(**input_data.model_dump())
        actual_run_id = self.run_id or str(uuid.uuid4())

        yield StreamStart(
            run_id=actual_run_id,
            node_id=self.name,
            input_preview={"prompt": formatted_prompt[:100]},
        )

        try:
            # Stream from agent
            async with self._agent.run_stream(formatted_prompt) as stream:
                token_index = 0
                async for chunk in stream.stream_text():
                    yield TokenChunk(
                        text=chunk,
                        token_index=token_index,
                        run_id=actual_run_id,
                        node_id=self.name,
                    )
                    token_index += 1

                # Get final result
                result = await stream.get_output()

            # Emit ToolResult with the actual result
            yield ToolResult(
                run_id=actual_run_id,
                node_id=self.name,
                tool_name="llm",
                call_id="",
                result=result,
                error=None,
            )

            # Emit end with result preview
            result_preview = None
            if hasattr(result, "model_dump"):
                result_preview = result.model_dump()  # type: ignore
            elif result is not None:
                result_preview = {"value": str(result)}

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
