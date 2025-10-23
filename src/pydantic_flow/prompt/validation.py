"""Template validation utilities."""

from collections.abc import Mapping
import re
import string

from pydantic_flow.prompt.enums import TemplateFormat


class VariableValidationError(ValueError):
    """Raised when template variable validation fails."""

    pass


def collect_variables(template: object, fmt: TemplateFormat) -> set[str]:
    """Extract variable names from a template.

    Args:
        template: Template object (str for most formats)
        fmt: Template format

    Returns:
        Set of variable names found in template

    Raises:
        TypeError: When template is not the expected type for format

    """
    if not isinstance(template, str):
        msg = f"Cannot collect variables from {type(template).__name__}"
        raise TypeError(msg)

    match fmt:
        case TemplateFormat.F_STRING:
            return _collect_f_string_vars(template)
        case TemplateFormat.JINJA2:
            return _collect_jinja2_vars(template)
        case TemplateFormat.MUSTACHE:
            return _collect_mustache_vars(template)

    msg = f"Unsupported template format: {fmt!s}"
    raise ValueError(msg)


def validate_missing_or_extra(
    vars_required: set[str], provided: Mapping[str, object]
) -> None:
    """Validate that all required variables are provided without extras.

    Args:
        vars_required: Set of variable names required by template
        provided: Mapping of provided variable names to values

    Raises:
        VariableValidationError: When variables are missing or extra

    """
    provided_set = set(provided.keys())
    missing = vars_required - provided_set
    extra = provided_set - vars_required

    if missing or extra:
        msg_parts = []
        if missing:
            msg_parts.append(f"Missing variables: {', '.join(sorted(missing))}")
        if extra:
            msg_parts.append(f"Extra variables: {', '.join(sorted(extra))}")
        msg = "; ".join(msg_parts)
        raise VariableValidationError(msg)


def _collect_f_string_vars(template: str) -> set[str]:
    """Extract variables from f-string template."""
    formatter = string.Formatter()
    vars_found = set()

    for _, field_name, _, _ in formatter.parse(template):
        if field_name is not None:
            var_name = field_name.split(".")[0].split("[")[0]
            if var_name:
                vars_found.add(var_name)

    return vars_found


def _collect_jinja2_vars(template: str) -> set[str]:
    """Extract variables from Jinja2 template."""
    pattern = r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\|[^}]*)?\}\}"
    matches = re.findall(pattern, template)
    return set(matches)


def _collect_mustache_vars(template: str) -> set[str]:
    """Extract variables from Mustache template."""
    pattern = r"\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}"
    matches = re.findall(pattern, template)
    return set(matches)
