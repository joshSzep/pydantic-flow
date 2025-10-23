"""F-string template engine using string.Formatter."""

from collections.abc import Mapping
import string


class MissingVariableError(KeyError):
    """Raised when a required template variable is missing."""

    def __init__(self, variable: str, template: str) -> None:
        """Initialize with variable name and template."""
        self.variable = variable
        self.template = template
        super().__init__(
            f"Missing required variable '{variable}' for template: {template[:100]}"
        )


class FStringRenderer:
    """Render f-string style templates using string.Formatter."""

    def __init__(self) -> None:
        """Initialize the f-string renderer."""
        self._formatter = string.Formatter()

    def render(self, template: object, variables: Mapping[str, object]) -> str:
        """Render f-string template with variables.

        Args:
            template: Template string with {variable} or {variable:format_spec}
            variables: Mapping of variable names to values

        Returns:
            Rendered string

        Raises:
            MissingVariableError: When a required variable is missing
            ValueError: When template format is invalid

        """
        if not isinstance(template, str):
            msg = f"F-string template must be str, got {type(template).__name__}"
            raise TypeError(msg)

        try:
            return self._formatter.vformat(template, (), variables)
        except KeyError as e:
            var_name = str(e).strip("'\"")
            raise MissingVariableError(var_name, template) from e
