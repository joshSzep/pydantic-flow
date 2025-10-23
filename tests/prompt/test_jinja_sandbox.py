"""Tests for Jinja2 template engine."""

from jinja2 import TemplateError
import pytest

from pydantic_flow.prompt.engines.jinja2 import Jinja2Renderer
from pydantic_flow.prompt.enums import EscapePolicy


class TestJinja2Renderer:
    """Test Jinja2 template rendering."""

    def test_simple_substitution(self) -> None:
        """Test basic variable substitution."""
        renderer = Jinja2Renderer()
        result = renderer.render("Hello {{ name }}!", {"name": "World"})
        assert result == "Hello World!"

    def test_multiple_variables(self) -> None:
        """Test multiple variable substitutions."""
        renderer = Jinja2Renderer()
        result = renderer.render(
            "{{ greeting }} {{ name }}, you are {{ age }} years old!",
            {"greeting": "Hello", "name": "Alice", "age": 30},
        )
        assert result == "Hello Alice, you are 30 years old!"

    def test_conditional(self) -> None:
        """Test conditional rendering."""
        renderer = Jinja2Renderer()
        template = "{% if admin %}Admin{% else %}User{% endif %}"
        result = renderer.render(template, {"admin": True})
        assert result == "Admin"

        result = renderer.render(template, {"admin": False})
        assert result == "User"

    def test_loop(self) -> None:
        """Test loop rendering."""
        renderer = Jinja2Renderer()
        template = "{% for item in items %}{{ item }} {% endfor %}"
        result = renderer.render(template, {"items": ["a", "b", "c"]})
        assert result == "a b c "

    def test_missing_variable_error(self) -> None:
        """Test error on missing variable."""
        renderer = Jinja2Renderer()
        with pytest.raises(TemplateError):
            renderer.render("Hello {{ name }}!", {})

    def test_invalid_template_type(self) -> None:
        """Test error on non-string template."""
        renderer = Jinja2Renderer()
        with pytest.raises(TypeError) as exc_info:
            renderer.render(123, {})

        assert "must be str" in str(exc_info.value)

    def test_empty_template(self) -> None:
        """Test empty template."""
        renderer = Jinja2Renderer()
        result = renderer.render("", {})
        assert result == ""

    def test_no_variables(self) -> None:
        """Test template with no variables."""
        renderer = Jinja2Renderer()
        result = renderer.render("Just plain text", {})
        assert result == "Just plain text"

    def test_html_escape_policy(self) -> None:
        """Test HTML escape policy."""
        renderer = Jinja2Renderer(escape_policy=EscapePolicy.HTML)
        result = renderer.render("Text: {{ text }}", {"text": "<script>"})
        assert "&lt;script&gt;" in result

    def test_autoescape_enabled(self) -> None:
        """Test with autoescape enabled."""
        renderer = Jinja2Renderer(autoescape=True)
        result = renderer.render("Text: {{ text }}", {"text": "<script>"})
        assert "&lt;script&gt;" in result

    def test_filter_usage(self) -> None:
        """Test Jinja2 filters."""
        renderer = Jinja2Renderer()
        result = renderer.render("{{ name|upper }}", {"name": "alice"})
        assert result == "ALICE"

    def test_whitespace_control(self) -> None:
        """Test whitespace control."""
        renderer = Jinja2Renderer()
        template = "{% for i in items -%}\n{{ i }}\n{%- endfor %}"
        result = renderer.render(template, {"items": [1, 2, 3]})
        assert result == "123"
