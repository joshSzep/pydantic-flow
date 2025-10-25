"""PromptNode implementation for LLM-based processing."""

from collections.abc import AsyncIterator
from typing import Any
import uuid

from pydantic import BaseModel
from pydantic_ai import Agent

from pydantic_flow.nodes.base import NodeOutput
from pydantic_flow.nodes.base import NodeWithInput
from pydantic_flow.prompt.engines import get_renderer
from pydantic_flow.prompt.enums import JoinStrategy
from pydantic_flow.prompt.enums import TemplateFormat
from pydantic_flow.prompt.templates import ChatPromptTemplate
from pydantic_flow.prompt.templates import PromptTemplate
from pydantic_flow.prompt.types import OutputParser
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
    template_format: TemplateFormat = TemplateFormat.F_STRING
    chat_join_strategy: JoinStrategy = JoinStrategy.SIMPLE


class PromptNode[InputModel: BaseModel, OutputT](NodeWithInput[InputModel, OutputT]):
    """A streaming-native node that calls an LLM using a templated prompt.

    This node creates a pydantic-ai agent internally and provides streaming
    execution with token visibility. It supports both simple string templates
    (for backward compatibility) and full PromptTemplate/ChatPromptTemplate
    objects for advanced templating with type safety.
    """

    def __init__(  # noqa: PLR0913
        self,
        prompt: (
            str | PromptTemplate[InputModel, str] | ChatPromptTemplate[InputModel, str]
        ),
        *,
        config: PromptConfig | None = None,
        output_parser: OutputParser[OutputT] | None = None,
        input: NodeOutput[InputModel] | None = None,
        name: str | None = None,
        run_id: str | None = None,
    ) -> None:
        """Initialize a PromptNode.

        Args:
            prompt: The prompt template - can be a simple string (uses
                config.template_format for rendering) or a PromptTemplate/
                ChatPromptTemplate object with embedded format
            config: Configuration for the LLM (model, system prompt, etc.)
            output_parser: Optional parser for structured output extraction
            input: Optional input from another node's output
            name: Optional unique identifier for this node
            run_id: Optional run identifier for tracking execution

        """
        super().__init__(input, name, run_id)
        self.config = config or PromptConfig()
        self.output_parser = output_parser

        # Handle different prompt types
        if isinstance(prompt, (PromptTemplate, ChatPromptTemplate)):
            self._template = prompt
            self._raw_prompt = None
        else:
            # Backward compatibility: string prompt
            self._raw_prompt = prompt
            self._template = None

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
        # Format prompt from input data using appropriate template system
        if self._template is not None:
            # Use the full PromptTemplate system
            if isinstance(self._template, ChatPromptTemplate):
                # Render chat messages and join them
                messages = self._template.render_messages(input_data)
                formatted_prompt = self._template.join(
                    self.config.chat_join_strategy, messages
                )
            else:
                # Render simple prompt template
                formatted_prompt = self._template.render(input_data)
        else:
            # Backward compatibility: use simple string format with configured format
            renderer = get_renderer(self.config.template_format)
            formatted_prompt = renderer.render(
                self._raw_prompt, input_data.model_dump()
            )

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

            # Apply output parser if configured
            if self.output_parser is not None:
                # Convert result to string if it's not already
                result_str = str(result)
                result = await self.output_parser.parse(result_str)

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
