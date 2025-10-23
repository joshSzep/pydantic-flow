"""Core types and protocols for the prompt system."""

from collections.abc import Mapping
from typing import Any
from typing import Protocol
from typing import TypeVar

from pydantic import BaseModel

from pydantic_flow.prompt.enums import ChatRole

TIn = TypeVar("TIn", bound=BaseModel)
TOut_co = TypeVar("TOut_co", covariant=True)


class PromptRenderer(Protocol):
    """Protocol for template rendering engines."""

    def render(self, template: object, variables: Mapping[str, object]) -> str:
        """Render a template with the provided variables."""
        ...


class OutputParser[TOut_co](Protocol):
    """Protocol for parsing LLM outputs into structured types."""

    async def parse(self, text: str) -> TOut_co:
        """Parse raw text output into structured data."""
        ...


class ChatMessage(BaseModel):
    """A single chat message with role and content."""

    role: ChatRole
    content: str

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Override to ensure role is serialized as string value."""
        data = super().model_dump(**kwargs)
        data["role"] = self.role.value
        return data
