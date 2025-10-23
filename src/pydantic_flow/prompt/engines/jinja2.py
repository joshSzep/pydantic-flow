"""Jinja2 template engine with sandboxed environment."""

from collections.abc import Mapping

from jinja2 import Environment
from jinja2 import StrictUndefined
from jinja2.sandbox import SandboxedEnvironment

from pydantic_flow.prompt.enums import EscapePolicy
from pydantic_flow.prompt.escape_policies import escape_html
from pydantic_flow.prompt.escape_policies import no_escape


class Jinja2Renderer:
    """Render Jinja2 templates with sandboxed environment."""

    def __init__(
        self,
        *,
        escape_policy: EscapePolicy = EscapePolicy.NONE,
        autoescape: bool = False,
    ) -> None:
        """Initialize the Jinja2 renderer.

        Args:
            escape_policy: Policy for escaping interpolated values
            autoescape: Enable Jinja2's built-in autoescaping

        """
        self.escape_policy = escape_policy
        self._env: Environment = SandboxedEnvironment(
            undefined=StrictUndefined, autoescape=autoescape
        )
        if escape_policy is EscapePolicy.HTML:
            self._env.filters["escape_html"] = escape_html
        self._env.filters["no_escape"] = no_escape

    def render(self, template: object, variables: Mapping[str, object]) -> str:
        """Render Jinja2 template with variables.

        Args:
            template: Template string with Jinja2 syntax
            variables: Mapping of variable names to values

        Returns:
            Rendered string

        Raises:
            TypeError: When template is not a string
            jinja2.TemplateError: When template syntax is invalid or variables are
                missing

        """
        if not isinstance(template, str):
            msg = f"Jinja2 template must be str, got {type(template).__name__}"
            raise TypeError(msg)

        tmpl = self._env.from_string(template)
        result = tmpl.render(variables)

        if self.escape_policy is EscapePolicy.HTML:
            return escape_html(result)
        return result
