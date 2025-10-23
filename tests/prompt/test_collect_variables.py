"""Tests for variable collection from templates."""

import pytest

from pydantic_flow.prompt.enums import TemplateFormat
from pydantic_flow.prompt.validation import collect_variables


class TestCollectVariables:
    """Test variable extraction from templates."""

    def test_collect_f_string_simple(self) -> None:
        """Test collecting variables from f-string template."""
        vars_found = collect_variables("Hello {name}!", TemplateFormat.F_STRING)
        assert vars_found == {"name"}

    def test_collect_f_string_multiple(self) -> None:
        """Test collecting multiple variables from f-string."""
        vars_found = collect_variables(
            "{greeting} {name}, you are {age}!",
            TemplateFormat.F_STRING,
        )
        assert vars_found == {"greeting", "name", "age"}

    def test_collect_f_string_with_format(self) -> None:
        """Test collecting variables with format specs."""
        vars_found = collect_variables(
            "Temperature: {temp:.1f}°C", TemplateFormat.F_STRING
        )
        assert vars_found == {"temp"}

    def test_collect_jinja2_simple(self) -> None:
        """Test collecting variables from Jinja2 template."""
        vars_found = collect_variables("Hello {{ name }}!", TemplateFormat.JINJA2)
        assert vars_found == {"name"}

    def test_collect_jinja2_multiple(self) -> None:
        """Test collecting multiple variables from Jinja2."""
        vars_found = collect_variables(
            "{{ greeting }} {{ name }}, you are {{ age }}!",
            TemplateFormat.JINJA2,
        )
        assert vars_found == {"greeting", "name", "age"}

    def test_collect_jinja2_with_filter(self) -> None:
        """Test collecting variables with Jinja2 filters."""
        vars_found = collect_variables("{{ name|upper }}", TemplateFormat.JINJA2)
        assert vars_found == {"name"}

    def test_collect_mustache_simple(self) -> None:
        """Test collecting variables from Mustache template."""
        vars_found = collect_variables("Hello {{name}}!", TemplateFormat.MUSTACHE)
        assert vars_found == {"name"}

    def test_collect_mustache_multiple(self) -> None:
        """Test collecting multiple variables from Mustache."""
        vars_found = collect_variables(
            "{{greeting}} {{name}}, you are {{age}}!",
            TemplateFormat.MUSTACHE,
        )
        assert vars_found == {"greeting", "name", "age"}

    def test_collect_no_variables(self) -> None:
        """Test template with no variables."""
        vars_found = collect_variables("Just plain text", TemplateFormat.F_STRING)
        assert vars_found == set()

    def test_collect_repeated_variable(self) -> None:
        """Test collecting repeated variables (deduplicated)."""
        vars_found = collect_variables("{name} {name} {name}", TemplateFormat.F_STRING)
        assert vars_found == {"name"}

    def test_invalid_template_type(self) -> None:
        """Test error on non-string template."""
        with pytest.raises(TypeError):
            collect_variables(123, TemplateFormat.F_STRING)

    def test_unsupported_format(self) -> None:
        """Test error on unsupported format."""
        with pytest.raises(ValueError) as exc_info:
            collect_variables("test", "unsupported")  # type: ignore[arg-type]

        assert "Unsupported" in str(exc_info.value)
