"""Prompt node integration for pydantic-flow workflows."""

from typing import Any

from pydantic import BaseModel

from pydantic_flow.prompt.templates import ChatPromptTemplate
from pydantic_flow.prompt.templates import PromptTemplate
from pydantic_flow.prompt.types import OutputParser


class TypedPromptNode[TIn: BaseModel, TOut]:
    """Typed prompt node for workflow integration with LLM calls."""

    def __init__(
        self,
        template: PromptTemplate[TIn, str] | ChatPromptTemplate[TIn, str],
        parser: OutputParser[TOut] | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize a typed prompt node.

        Args:
            template: Prompt or chat template to render
            parser: Optional parser for structured output
            name: Optional node name (default "prompt")

        """
        self.template = template
        self.parser = parser or template.output_parser
        self.name = name or "prompt"

    async def run(
        self, input: TIn, model_adapter: Any | None = None
    ) -> TOut | str | list[Any]:
        """Run the prompt node with input data.

        Args:
            input: Input model instance
            model_adapter: Optional LLM adapter to call with rendered prompt

        Returns:
            Parsed output if parser configured, else raw result

        Raises:
            NotImplementedError: When model_adapter integration is not configured

        """
        if isinstance(self.template, ChatPromptTemplate):
            rendered = self.template.render_messages(input)
        else:
            rendered = self.template.render(input)

        if model_adapter is None:
            msg = "Model adapter integration not yet implemented"
            raise NotImplementedError(msg)

        raw = await self._call_model(rendered, model_adapter)

        if self.parser:
            return await self.parser.parse(raw)

        return raw

    async def _call_model(self, prompt: Any, adapter: Any) -> str:
        """Call the LLM model adapter with rendered prompt.

        Args:
            prompt: Rendered prompt or messages
            adapter: LLM adapter instance

        Returns:
            Raw model output text

        Raises:
            NotImplementedError: Model integration not yet implemented

        """
        msg = "Model adapter integration not yet implemented"
        raise NotImplementedError(msg)
