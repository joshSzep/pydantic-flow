"""Mustache template engine using chevron."""

from collections.abc import Mapping

import chevron


class MustacheRenderer:
    """Render Mustache (logic-less) templates using chevron."""

    def __init__(self, *, strict: bool = True) -> None:
        """Initialize the Mustache renderer.

        Args:
            strict: Whether to error on missing variables (default True)

        """
        self.strict = strict

    def render(self, template: object, variables: Mapping[str, object]) -> str:
        """Render Mustache template with variables.

        Args:
            template: Template string with {{variable}} syntax
            variables: Mapping of variable names to values

        Returns:
            Rendered string

        Raises:
            TypeError: When template is not a string
            chevron.ChevronError: When template syntax is invalid

        """
        if not isinstance(template, str):
            msg = f"Mustache template must be str, got {type(template).__name__}"
            raise TypeError(msg)

        return chevron.render(template, dict(variables), warn=self.strict)
