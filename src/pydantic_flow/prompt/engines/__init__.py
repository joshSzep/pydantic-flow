"""Engine registry for template renderers."""

from pydantic_flow.prompt.engines.f_string import FStringRenderer
from pydantic_flow.prompt.engines.jinja2 import Jinja2Renderer
from pydantic_flow.prompt.engines.mustache import MustacheRenderer
from pydantic_flow.prompt.enums import TemplateFormat
from pydantic_flow.prompt.types import PromptRenderer

__all__ = [
    "FStringRenderer",
    "Jinja2Renderer",
    "MustacheRenderer",
    "get_renderer",
]


def get_renderer(fmt: TemplateFormat) -> PromptRenderer:
    """Get a template renderer for the specified format.

    Args:
        fmt: Template format to get renderer for

    Returns:
        Renderer instance for the specified format

    Raises:
        ValueError: When format is not supported

    """
    match fmt:
        case TemplateFormat.F_STRING:
            return FStringRenderer()
        case TemplateFormat.JINJA2:
            return Jinja2Renderer()
        case TemplateFormat.MUSTACHE:
            return MustacheRenderer()
    msg = f"Unsupported template format: {fmt!s}"
    raise ValueError(msg)
