"""Serialization and observability for prompt templates."""

from collections.abc import Mapping
import hashlib
import time
from typing import Any
from typing import cast

from opentelemetry import trace

from pydantic_flow.prompt.enums import TemplateFormat
from pydantic_flow.prompt.templates import ChatPromptTemplate
from pydantic_flow.prompt.templates import PromptTemplate
from pydantic_flow.prompt.validation import collect_variables

tracer = trace.get_tracer(__name__)


def to_dict(
    template: PromptTemplate[Any, Any] | ChatPromptTemplate[Any, Any],
) -> dict[str, Any]:
    """Serialize a prompt template to dictionary.

    Args:
        template: Template to serialize

    Returns:
        Dictionary representation of template

    """
    base_dict: dict[str, Any] = {
        "format": template.format.value,
        "input_model": template.input_model.__name__,
        "has_parser": template.output_parser is not None,
    }

    if isinstance(template, ChatPromptTemplate):
        chat_template = cast(ChatPromptTemplate[Any, Any], template)
        base_dict["type"] = "chat"
        base_dict["messages"] = [
            {"role": m.role.value, "content": m.content} for m in chat_template.messages
        ]
    else:
        base_dict["type"] = "prompt"
        base_dict["template"] = str(template.template)

    return base_dict


def from_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Deserialize a prompt template from dictionary.

    Args:
        data: Dictionary representation

    Returns:
        Dictionary with template configuration

    Raises:
        NotImplementedError: Full deserialization not yet implemented

    """
    msg = "Template deserialization not yet fully implemented"
    raise NotImplementedError(msg)


def render_with_observability(
    template: object,
    variables: Mapping[str, object],
    format: TemplateFormat,
    renderer: Any,
) -> str:
    """Render template with OpenTelemetry tracing.

    Args:
        template: Template to render
        variables: Variable mapping
        format: Template format
        renderer: Renderer instance

    Returns:
        Rendered string with telemetry

    """
    with tracer.start_as_current_span("prompt.render") as span:
        start_time = time.perf_counter()

        span.set_attribute("prompt.format", format.value)
        span.set_attribute("prompt.template_hash", _hash_template(template))
        span.set_attribute("prompt.variable_count", len(variables) if variables else 0)

        try:
            vars_required = collect_variables(template, format)
            span.set_attribute("prompt.required_vars", ",".join(sorted(vars_required)))

            missing = vars_required - set(variables.keys())
            if missing:
                span.set_attribute("prompt.missing_vars", ",".join(sorted(missing)))
        except TypeError, ValueError:
            pass

        result = renderer.render(template, variables)

        render_ms = (time.perf_counter() - start_time) * 1000
        span.set_attribute("prompt.render_ms", render_ms)
        span.set_attribute("prompt.result_length", len(result))

        return result


def _hash_template(template: object) -> str:
    """Generate hash of template for telemetry."""
    template_str = str(template)[:500]
    return hashlib.sha256(template_str.encode()).hexdigest()[:16]
