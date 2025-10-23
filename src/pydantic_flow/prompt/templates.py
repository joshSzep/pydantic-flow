"""Template classes for prompt rendering and structured output."""

from collections.abc import Mapping
from typing import TypeVar

from pydantic import BaseModel

from pydantic_flow.prompt.engines import get_renderer
from pydantic_flow.prompt.enums import JoinStrategy
from pydantic_flow.prompt.enums import TemplateFormat
from pydantic_flow.prompt.types import ChatMessage
from pydantic_flow.prompt.types import OutputParser
from pydantic_flow.prompt.types import PromptRenderer

TIn = TypeVar("TIn", bound=BaseModel)
TOut = TypeVar("TOut")


class PromptTemplate[TIn: BaseModel, TOut]:
    """Type-safe prompt template with structured input and optional parser."""

    def __init__(
        self,
        *,
        template: object,
        format: TemplateFormat,
        input_model: type[TIn],
        output_parser: OutputParser[TOut] | None = None,
    ) -> None:
        """Initialize a prompt template.

        Args:
            template: Template object (typically str)
            format: Template format to use for rendering
            input_model: Pydantic model defining input variables
            output_parser: Optional parser for structured output

        """
        self.template = template
        self.format = format
        self.input_model = input_model
        self.output_parser = output_parser
        self._renderer: PromptRenderer = get_renderer(format)

    def render(self, input: TIn, extra: Mapping[str, object] | None = None) -> str:
        """Render template with input model instance.

        Args:
            input: Input model instance providing variables
            extra: Optional extra variables not in model

        Returns:
            Rendered prompt string

        Raises:
            TypeError: When template format is invalid
            KeyError: When required variables are missing

        """
        data = input.model_dump()
        if extra:
            data = {**data, **extra}
        return self._renderer.render(self.template, data)

    async def render_and_parse(
        self, input: TIn, extra: Mapping[str, object] | None = None
    ) -> TOut | str:
        """Render template and parse output if parser is configured.

        Args:
            input: Input model instance providing variables
            extra: Optional extra variables not in model

        Returns:
            Parsed output if parser configured, else rendered string

        """
        rendered = self.render(input, extra)
        if self.output_parser:
            return await self.output_parser.parse(rendered)
        return rendered


class ChatPromptTemplate[TIn: BaseModel, TOut]:
    """Type-safe chat prompt template with role-based messages."""

    def __init__(
        self,
        *,
        messages: list[ChatMessage],
        format: TemplateFormat,
        input_model: type[TIn],
        output_parser: OutputParser[TOut] | None = None,
    ) -> None:
        """Initialize a chat prompt template.

        Args:
            messages: List of message templates with roles
            format: Template format to use for rendering
            input_model: Pydantic model defining input variables
            output_parser: Optional parser for structured output

        """
        self.messages = messages
        self.format = format
        self.input_model = input_model
        self.output_parser = output_parser
        self._renderer: PromptRenderer = get_renderer(format)

    def render_messages(
        self, input: TIn, extra: Mapping[str, object] | None = None
    ) -> list[ChatMessage]:
        """Render all message templates with input model instance.

        Args:
            input: Input model instance providing variables
            extra: Optional extra variables not in model

        Returns:
            List of rendered chat messages with roles

        Raises:
            TypeError: When template format is invalid
            KeyError: When required variables are missing

        """
        data = input.model_dump()
        if extra:
            data = {**data, **extra}

        return [
            ChatMessage(role=m.role, content=self._renderer.render(m.content, data))
            for m in self.messages
        ]

    def join(
        self,
        strategy: JoinStrategy,
        rendered: list[ChatMessage] | None = None,
    ) -> str:
        """Join chat messages into a single string.

        Args:
            strategy: Strategy for joining messages
            rendered: Optional pre-rendered messages (uses templates if None)

        Returns:
            Joined message string

        """
        msgs = rendered or self.messages

        match strategy:
            case JoinStrategy.OPENAI:
                return "\n".join(f"{m.role.value}: {m.content}" for m in msgs)
            case JoinStrategy.ANTHROPIC:
                return "\n\n".join(f"{m.role.value}: {m.content}" for m in msgs)
            case JoinStrategy.SIMPLE:
                return "\n".join(m.content for m in msgs)

        msg = f"Unsupported join strategy: {strategy!s}"
        raise ValueError(msg)

    async def render_and_parse(
        self, input: TIn, extra: Mapping[str, object] | None = None
    ) -> TOut | list[ChatMessage]:
        """Render messages and parse output if parser is configured.

        Args:
            input: Input model instance providing variables
            extra: Optional extra variables not in model

        Returns:
            Parsed output if parser configured, else rendered messages

        """
        rendered = self.render_messages(input, extra)
        if self.output_parser:
            joined = self.join(JoinStrategy.SIMPLE, rendered)
            return await self.output_parser.parse(joined)
        return rendered


def from_template[TIn: BaseModel, TOut](
    template: object,
    *,
    format: TemplateFormat,
    input_model: type[TIn],
    output_parser: OutputParser[TOut] | None = None,
) -> PromptTemplate[TIn, TOut]:
    """Create a prompt template from a template string.

    Args:
        template: Template object (typically str)
        format: Template format to use for rendering
        input_model: Pydantic model defining input variables
        output_parser: Optional parser for structured output

    Returns:
        Configured prompt template instance

    """
    return PromptTemplate(
        template=template,
        format=format,
        input_model=input_model,
        output_parser=output_parser,
    )
