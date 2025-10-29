"""Tests for prompt serialization and observability."""

from pydantic import BaseModel
import pytest

from pydantic_flow.prompt.enums import ChatRole
from pydantic_flow.prompt.enums import TemplateFormat
from pydantic_flow.prompt.serde import _hash_template
from pydantic_flow.prompt.serde import from_dict
from pydantic_flow.prompt.serde import render_with_observability
from pydantic_flow.prompt.serde import to_dict
from pydantic_flow.prompt.templates import ChatPromptTemplate
from pydantic_flow.prompt.templates import PromptTemplate
from pydantic_flow.prompt.types import ChatMessage


class SimpleInput(BaseModel):
    """Test input model."""

    name: str
    age: int


class SimpleParser:
    """Simple output parser for testing."""

    async def parse(self, text: str) -> str:
        """Parse text to uppercase."""
        return text.upper()


def test_to_dict_simple_prompt() -> None:
    """Test serializing simple prompt template."""
    template = PromptTemplate[SimpleInput, None](
        template="Hello {name}, you are {age} years old",
        input_model=SimpleInput,
        format=TemplateFormat.F_STRING,
    )

    result = to_dict(template)

    assert result["type"] == "prompt"
    assert result["format"] == TemplateFormat.F_STRING.value
    assert result["input_model"] == "SimpleInput"
    assert result["has_parser"] is False
    assert result["template"] == "Hello {name}, you are {age} years old"


def test_to_dict_chat_prompt() -> None:
    """Test serializing chat prompt template."""
    messages = [
        ChatMessage(role=ChatRole.SYSTEM, content="You are a helpful assistant"),
        ChatMessage(role=ChatRole.USER, content="Hello {name}"),
    ]
    template = ChatPromptTemplate[SimpleInput, None](
        messages=messages,
        input_model=SimpleInput,
        format=TemplateFormat.F_STRING,
    )

    result = to_dict(template)

    assert result["type"] == "chat"
    assert result["format"] == TemplateFormat.F_STRING.value
    assert result["input_model"] == "SimpleInput"
    assert result["has_parser"] is False
    assert len(result["messages"]) == 2
    assert result["messages"][0]["role"] == ChatRole.SYSTEM.value
    assert result["messages"][0]["content"] == "You are a helpful assistant"
    assert result["messages"][1]["role"] == ChatRole.USER.value
    assert result["messages"][1]["content"] == "Hello {name}"


def test_to_dict_with_parser() -> None:
    """Test serializing template with output parser."""
    template = PromptTemplate[SimpleInput, str](
        template="Hello {name}",
        input_model=SimpleInput,
        format=TemplateFormat.F_STRING,
        output_parser=SimpleParser(),
    )

    result = to_dict(template)

    assert result["has_parser"] is True


def test_from_dict_raises_not_implemented() -> None:
    """Test that deserialization raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match="not yet fully implemented"):
        from_dict({"type": "prompt", "template": "test"})


def test_hash_template_basic() -> None:
    """Test template hashing."""
    template = "Hello {name}"
    hash1 = _hash_template(template)

    assert isinstance(hash1, str)
    assert len(hash1) == 16


def test_hash_template_consistent() -> None:
    """Test that identical templates produce identical hashes."""
    template = "Hello {name}, you are {age} years old"
    hash1 = _hash_template(template)
    hash2 = _hash_template(template)

    assert hash1 == hash2


def test_hash_template_different_for_different_content() -> None:
    """Test that different templates produce different hashes."""
    hash1 = _hash_template("Hello {name}")
    hash2 = _hash_template("Goodbye {name}")

    assert hash1 != hash2


def test_hash_template_truncates_long_strings() -> None:
    """Test that long templates are truncated before hashing."""
    long_template = "x" * 1000
    hash_result = _hash_template(long_template)

    assert isinstance(hash_result, str)
    assert len(hash_result) == 16


class MockRenderer:
    """Mock renderer for testing."""

    def render(self, template: object, variables: dict[str, object]) -> str:
        """Mock render method."""
        return f"rendered: {template} with {variables}"


def test_render_with_observability_basic() -> None:
    """Test rendering with observability tracking."""
    template = "Hello {name}"
    variables = {"name": "Alice"}
    renderer = MockRenderer()

    result = render_with_observability(
        template=template,
        variables=variables,
        format=TemplateFormat.F_STRING,
        renderer=renderer,
    )

    assert result == "rendered: Hello {name} with {'name': 'Alice'}"


def test_render_with_observability_no_variables() -> None:
    """Test rendering with empty variable dict."""
    template = "Hello world"
    variables: dict[str, object] = {}
    renderer = MockRenderer()

    result = render_with_observability(
        template=template,
        variables=variables,
        format=TemplateFormat.F_STRING,
        renderer=renderer,
    )

    assert result == "rendered: Hello world with {}"


def test_render_with_observability_multiple_variables() -> None:
    """Test rendering with multiple variables."""
    template = "Hello {name}, you are {age} years old"
    variables = {"name": "Bob", "age": 30}
    renderer = MockRenderer()

    result = render_with_observability(
        template=template,
        variables=variables,
        format=TemplateFormat.F_STRING,
        renderer=renderer,
    )

    assert "rendered:" in result
    assert "'name': 'Bob'" in result
    assert "'age': 30" in result
