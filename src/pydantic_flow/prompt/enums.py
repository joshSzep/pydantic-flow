"""Type-safe enumerations for prompt system."""

from enum import StrEnum


class TemplateFormat(StrEnum):
    """Supported template rendering formats."""

    F_STRING = "f-string"
    JINJA2 = "jinja2"
    MUSTACHE = "mustache"


class ChatRole(StrEnum):
    """Chat message roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class JoinStrategy(StrEnum):
    """Strategies for joining chat messages into a single string."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    SIMPLE = "simple"


class EscapePolicy(StrEnum):
    """Escape policies for template rendering."""

    NONE = "none"
    HTML = "html"
