"""Tests for Mustache template engine."""

import pytest

from pydantic_flow.prompt.engines.mustache import MustacheRenderer


class TestMustacheRenderer:
    """Test Mustache template rendering."""

    def test_simple_substitution(self) -> None:
        """Test basic variable substitution."""
        renderer = MustacheRenderer()
        result = renderer.render("Hello {{name}}!", {"name": "World"})
        assert result == "Hello World!"

    def test_multiple_variables(self) -> None:
        """Test multiple variable substitutions."""
        renderer = MustacheRenderer()
        result = renderer.render(
            "{{greeting}} {{name}}, you are {{age}} years old!",
            {"greeting": "Hello", "name": "Alice", "age": 30},
        )
        assert result == "Hello Alice, you are 30 years old!"

    def test_section_with_list(self) -> None:
        """Test section rendering with list."""
        renderer = MustacheRenderer()
        template = "{{#items}}{{.}} {{/items}}"
        result = renderer.render(template, {"items": ["a", "b", "c"]})
        assert result == "a b c "

    def test_section_with_bool_true(self) -> None:
        """Test section rendering with boolean true."""
        renderer = MustacheRenderer()
        template = "{{#show}}visible{{/show}}"
        result = renderer.render(template, {"show": True})
        assert result == "visible"

    def test_section_with_bool_false(self) -> None:
        """Test section rendering with boolean false."""
        renderer = MustacheRenderer()
        template = "{{#show}}visible{{/show}}"
        result = renderer.render(template, {"show": False})
        assert result == ""

    def test_inverted_section(self) -> None:
        """Test inverted section rendering."""
        renderer = MustacheRenderer()
        template = "{{^show}}hidden{{/show}}"
        result = renderer.render(template, {"show": False})
        assert result == "hidden"

        result = renderer.render(template, {"show": True})
        assert result == ""

    def test_missing_variable_non_strict(self) -> None:
        """Test missing variable with non-strict mode."""
        renderer = MustacheRenderer(strict=False)
        result = renderer.render("Hello {{name}}!", {})
        assert result == "Hello !"

    def test_invalid_template_type(self) -> None:
        """Test error on non-string template."""
        renderer = MustacheRenderer()
        with pytest.raises(TypeError) as exc_info:
            renderer.render(123, {})

        assert "must be str" in str(exc_info.value)

    def test_empty_template(self) -> None:
        """Test empty template."""
        renderer = MustacheRenderer()
        result = renderer.render("", {})
        assert result == ""

    def test_no_variables(self) -> None:
        """Test template with no variables."""
        renderer = MustacheRenderer()
        result = renderer.render("Just plain text", {})
        assert result == "Just plain text"

    def test_nested_section(self) -> None:
        """Test nested section."""
        renderer = MustacheRenderer()
        template = "{{#outer}}{{#inner}}{{value}}{{/inner}}{{/outer}}"
        result = renderer.render(template, {"outer": {"inner": {"value": "nested"}}})
        assert result == "nested"

    def test_comment(self) -> None:
        """Test comment is ignored."""
        renderer = MustacheRenderer()
        result = renderer.render("Before{{! this is a comment }}After", {})
        assert result == "BeforeAfter"
