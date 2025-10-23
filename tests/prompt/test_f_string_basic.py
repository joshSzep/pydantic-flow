"""Tests for f-string template engine."""

import pytest

from pydantic_flow.prompt.engines.f_string import FStringRenderer
from pydantic_flow.prompt.engines.f_string import MissingVariableError


class TestFStringRenderer:
    """Test f-string template rendering."""

    def test_simple_substitution(self) -> None:
        """Test basic variable substitution."""
        renderer = FStringRenderer()
        result = renderer.render("Hello {name}!", {"name": "World"})
        assert result == "Hello World!"

    def test_multiple_variables(self) -> None:
        """Test multiple variable substitutions."""
        renderer = FStringRenderer()
        result = renderer.render(
            "{greeting} {name}, you are {age} years old!",
            {"greeting": "Hello", "name": "Alice", "age": 30},
        )
        assert result == "Hello Alice, you are 30 years old!"

    def test_format_spec(self) -> None:
        """Test format specifications."""
        renderer = FStringRenderer()
        result = renderer.render("Temperature: {temp:.1f}°C", {"temp": 23.456})
        assert result == "Temperature: 23.5°C"

    def test_format_spec_integer(self) -> None:
        """Test integer format specification."""
        renderer = FStringRenderer()
        result = renderer.render("Count: {n:03d}", {"n": 5})
        assert result == "Count: 005"

    def test_missing_variable_error(self) -> None:
        """Test error on missing variable."""
        renderer = FStringRenderer()
        with pytest.raises(MissingVariableError) as exc_info:
            renderer.render("Hello {name}!", {})

        error: MissingVariableError = exc_info.value  # type: ignore[assignment]
        assert "name" in str(error)
        assert error.variable == "name"

    def test_invalid_template_type(self) -> None:
        """Test error on non-string template."""
        renderer = FStringRenderer()
        with pytest.raises(TypeError) as exc_info:
            renderer.render(123, {})

        assert "must be str" in str(exc_info.value)

    def test_empty_template(self) -> None:
        """Test empty template."""
        renderer = FStringRenderer()
        result = renderer.render("", {})
        assert result == ""

    def test_no_variables(self) -> None:
        """Test template with no variables."""
        renderer = FStringRenderer()
        result = renderer.render("Just plain text", {})
        assert result == "Just plain text"

    def test_extra_variables_ignored(self) -> None:
        """Test that extra variables are ignored."""
        renderer = FStringRenderer()
        result = renderer.render("Hello {name}!", {"name": "World", "extra": "ignored"})
        assert result == "Hello World!"

    def test_repeated_variable(self) -> None:
        """Test using same variable multiple times."""
        renderer = FStringRenderer()
        result = renderer.render("{name} {name} {name}!", {"name": "Echo"})
        assert result == "Echo Echo Echo!"
