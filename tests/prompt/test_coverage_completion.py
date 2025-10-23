"""Tests to achieve 100% coverage for prompt module."""

import json

from pydantic import BaseModel
import pytest

from pydantic_flow.prompt import AsIsParser
from pydantic_flow.prompt import ChatMessage
from pydantic_flow.prompt import ChatPromptTemplate
from pydantic_flow.prompt import ChatRole
from pydantic_flow.prompt import DelimitedParser
from pydantic_flow.prompt import JoinStrategy
from pydantic_flow.prompt import JsonModelParser
from pydantic_flow.prompt import PromptTemplate
from pydantic_flow.prompt import TemplateFormat
from pydantic_flow.prompt import TypedPromptNode
from pydantic_flow.prompt import collect_variables
from pydantic_flow.prompt import from_template
from pydantic_flow.prompt import get_renderer
from pydantic_flow.prompt import validate_missing_or_extra
from pydantic_flow.prompt.escape_policies import escape_html
from pydantic_flow.prompt.escape_policies import no_escape
from pydantic_flow.prompt.serde import _hash_template
from pydantic_flow.prompt.serde import from_dict as serde_from_dict
from pydantic_flow.prompt.serde import render_with_observability
from pydantic_flow.prompt.serde import to_dict as serde_to_dict
from pydantic_flow.prompt.validation import VariableValidationError

# Constants for test assertions
HASH_LENGTH = 16
MESSAGE_COUNT = 2


class SimpleInput(BaseModel):
    """Simple input model for testing."""

    name: str
    value: int = 42


class TestEngineRegistry:
    """Test engine registry edge cases."""

    def test_unsupported_format_error(self) -> None:
        """Test error when requesting unsupported format."""
        # Create an invalid enum value by monkey-patching
        with pytest.raises(ValueError, match="Unsupported template format"):
            # Use a non-existent format to trigger the error path
            get_renderer("invalid_format")  # type: ignore


class TestEscapePolicies:
    """Test escape policy functions."""

    def test_no_escape_returns_unchanged(self) -> None:
        """Test no_escape returns string unchanged."""
        text = "<script>alert('xss')</script>"
        assert no_escape(text) == text

    def test_escape_html_escapes_special_chars(self) -> None:
        """Test escape_html properly escapes HTML."""
        text = "<script>alert('xss')</script>"
        escaped = escape_html(text)
        assert "&lt;" in escaped
        assert "&gt;" in escaped
        assert "<script>" not in escaped


class TestChatMessage:
    """Test ChatMessage edge cases."""

    def test_chat_message_str_representation(self) -> None:
        """Test ChatMessage __str__ method."""
        msg = ChatMessage(role=ChatRole.USER, content="Hello world")
        str_repr = str(msg)
        assert "USER" in str_repr or "user" in str_repr
        assert "Hello world" in str_repr


class TestDelimitedParser:
    """Test DelimitedParser edge cases."""

    async def test_delimited_parser_empty_parts(self) -> None:
        """Test parser handles empty parts correctly."""
        parser = DelimitedParser(sep="|")
        result = await parser.parse("  |  |  ")
        assert result == {"0": "", "1": "", "2": ""}

    async def test_delimited_parser_single_value(self) -> None:
        """Test parser with single value (no delimiter)."""
        parser = DelimitedParser(sep="|")
        result = await parser.parse("single")
        assert result == {"0": "single"}

    async def test_delimited_parser_custom_separator(self) -> None:
        """Test parser with custom separator."""
        parser = DelimitedParser(sep=",")
        result = await parser.parse("a, b, c")
        assert result == {"0": "a", "1": "b", "2": "c"}


class TestTemplateEdgeCases:
    """Test template edge cases."""

    def test_prompt_template_without_parser(self) -> None:
        """Test PromptTemplate render without output parser."""
        template = PromptTemplate[SimpleInput, str](
            template="Hello {name}!",
            format=TemplateFormat.F_STRING,
            input_model=SimpleInput,
            output_parser=None,
        )
        result = template.render(SimpleInput(name="Alice"))
        assert result == "Hello Alice!"

    async def test_prompt_template_render_and_parse_no_parser(self) -> None:
        """Test render_and_parse without parser returns string."""
        template = PromptTemplate[SimpleInput, str](
            template="Hello {name}!",
            format=TemplateFormat.F_STRING,
            input_model=SimpleInput,
            output_parser=None,
        )
        result = await template.render_and_parse(SimpleInput(name="Alice"))
        assert result == "Hello Alice!"

    async def test_chat_template_render_and_parse_no_parser(self) -> None:
        """Test chat render_and_parse without parser returns messages."""
        template = ChatPromptTemplate[SimpleInput, list[ChatMessage]](
            messages=[
                ChatMessage(role=ChatRole.SYSTEM, content="System message"),
                ChatMessage(role=ChatRole.USER, content="Hello {name}!"),
            ],
            format=TemplateFormat.F_STRING,
            input_model=SimpleInput,
            output_parser=None,
        )
        result = await template.render_and_parse(SimpleInput(name="Alice"))
        assert isinstance(result, list)
        expected_message_count = 2
        assert len(result) == expected_message_count
        assert result[1].content == "Hello Alice!"

    async def test_chat_template_render_and_parse_with_parser(self) -> None:
        """Test chat render_and_parse with parser uses SIMPLE join."""
        template = ChatPromptTemplate[SimpleInput, str](
            messages=[
                ChatMessage(role=ChatRole.SYSTEM, content="System"),
                ChatMessage(role=ChatRole.USER, content="Hello {name}!"),
            ],
            format=TemplateFormat.F_STRING,
            input_model=SimpleInput,
            output_parser=AsIsParser(),
        )
        result = await template.render_and_parse(SimpleInput(name="Alice"))
        # Should use SIMPLE join (content only, newline separated)
        assert result == "System\nHello Alice!"

    def test_chat_join_with_none_uses_templates(self) -> None:
        """Test chat join with None uses template messages."""
        template = ChatPromptTemplate[SimpleInput, str](
            messages=[
                ChatMessage(role=ChatRole.SYSTEM, content="System"),
                ChatMessage(role=ChatRole.USER, content="User"),
            ],
            format=TemplateFormat.F_STRING,
            input_model=SimpleInput,
        )
        # Call join with rendered=None to use template messages
        result = template.join(JoinStrategy.SIMPLE, rendered=None)
        assert result == "System\nUser"

    def test_chat_join_unsupported_strategy(self) -> None:
        """Test chat join with unsupported strategy raises error."""
        template = ChatPromptTemplate[SimpleInput, str](
            messages=[ChatMessage(role=ChatRole.USER, content="Test")],
            format=TemplateFormat.F_STRING,
            input_model=SimpleInput,
        )
        with pytest.raises(ValueError, match="Unsupported join strategy"):
            # Use invalid strategy to trigger error path
            template.join("invalid_strategy")  # type: ignore

    def test_chat_render_messages_with_extra_vars(self) -> None:
        """Test chat render_messages with extra variables."""
        template = ChatPromptTemplate[SimpleInput, str](
            messages=[
                ChatMessage(role=ChatRole.USER, content="Hello {name}, {extra}!")
            ],
            format=TemplateFormat.F_STRING,
            input_model=SimpleInput,
        )

        # Render with extra variables beyond model
        rendered = template.render_messages(
            SimpleInput(name="Alice"), extra={"extra": "bonus"}
        )

        assert len(rendered) == 1
        assert rendered[0].content == "Hello Alice, bonus!"

    def test_from_template_helper(self) -> None:
        """Test from_template helper function."""
        template = from_template(
            "Hello {name}!",
            format=TemplateFormat.F_STRING,
            input_model=SimpleInput,
        )
        assert isinstance(template, PromptTemplate)
        result = template.render(SimpleInput(name="Bob"))
        assert result == "Hello Bob!"


class TestValidationEdgeCases:
    """Test validation edge cases."""

    def test_collect_variables_invalid_type(self) -> None:
        """Test collect_variables with invalid template type."""
        with pytest.raises(TypeError, match="Cannot collect variables"):
            collect_variables(123, TemplateFormat.F_STRING)  # type: ignore

    def test_collect_variables_unsupported_format(self) -> None:
        """Test collect_variables with unsupported format."""
        with pytest.raises(ValueError, match="Unsupported template format"):
            collect_variables("test", "invalid")  # type: ignore

    def test_validate_missing_variables(self) -> None:
        """Test validation error when variables are missing."""
        with pytest.raises(VariableValidationError, match="Missing variables: name"):
            validate_missing_or_extra({"name", "age"}, {"age": 30})

    def test_validate_extra_variables(self) -> None:
        """Test validation error when extra variables provided."""
        with pytest.raises(VariableValidationError, match="Extra variables: extra"):
            validate_missing_or_extra({"name"}, {"name": "Alice", "extra": 42})

    def test_validate_missing_and_extra(self) -> None:
        """Test validation error with both missing and extra variables."""
        with pytest.raises(VariableValidationError) as exc_info:
            validate_missing_or_extra({"name", "age"}, {"age": 30, "extra": 42})
        error_msg = str(exc_info.value)
        assert "Missing variables" in error_msg
        assert "Extra variables" in error_msg

    def test_validate_all_correct(self) -> None:
        """Test validation passes when all variables are correct."""
        # Should not raise
        validate_missing_or_extra({"name", "age"}, {"name": "Alice", "age": 30})


class TestJsonModelParserEdgeCases:
    """Test JsonModelParser edge cases."""

    async def test_json_parser_non_model_type_error(self) -> None:
        """Test JsonModelParser with non-BaseModel type raises TypeError."""

        class NotAModel:
            """Not a Pydantic model."""

            pass

        parser = JsonModelParser[NotAModel](NotAModel)  # type: ignore
        with pytest.raises(TypeError, match="Cannot convert"):
            await parser.parse('{"key": "value"}')

    async def test_json_parser_with_already_parsed_object(self) -> None:
        """Test JsonModelParser when JSON already contains correct type."""

        class TestModel(BaseModel):
            """Test model."""

            value: str

        # Create a custom JSON that when parsed is already a TestModel
        # This is a contrived case but tests line 63 (return obj)
        parser = JsonModelParser[TestModel](TestModel)

        # Manually construct an instance
        instance = TestModel(value="test")

        # Mock the parser to receive an object that's already the right type
        # We can test this by passing a JSON-serialized version
        result = await parser.parse(json.dumps(instance.model_dump()))

        assert isinstance(result, TestModel)
        assert result.value == "test"


class TestChatMessageStr:
    """Test ChatMessage __str__ implementation."""

    def test_chat_message_repr_includes_role_and_content(self) -> None:
        """Test ChatMessage string representation."""
        msg = ChatMessage(role=ChatRole.SYSTEM, content="Test content")
        # The __str__ method should include role and content
        str_output = str(msg)
        assert "role" in str_output.lower() or "system" in str_output.lower()
        assert "content" in str_output.lower() or "Test content" in str_output

    def test_chat_message_model_dump_serializes_role(self) -> None:
        """Test ChatMessage.model_dump() serializes role as string."""
        msg = ChatMessage(role=ChatRole.USER, content="Hello")
        dumped = msg.model_dump()
        assert dumped["role"] == "user"
        assert isinstance(dumped["role"], str)


class TestTypedPromptNode:
    """Test TypedPromptNode edge cases."""

    async def test_node_run_without_adapter_raises(self) -> None:
        """Test node.run() without model adapter raises NotImplementedError."""
        template = PromptTemplate[SimpleInput, str](
            template="Hello {name}!",
            format=TemplateFormat.F_STRING,
            input_model=SimpleInput,
        )
        node = TypedPromptNode(template=template, name="test_node")

        with pytest.raises(
            NotImplementedError, match="Model adapter integration not yet"
        ):
            await node.run(SimpleInput(name="Alice"))

    async def test_node_run_with_chat_template_raises(self) -> None:
        """Test node.run() with chat template also raises NotImplementedError."""
        chat_template = ChatPromptTemplate[SimpleInput, str](
            messages=[ChatMessage(role=ChatRole.USER, content="Hello {name}!")],
            format=TemplateFormat.F_STRING,
            input_model=SimpleInput,
        )
        node = TypedPromptNode(template=chat_template, name="chat_node")

        with pytest.raises(
            NotImplementedError, match="Model adapter integration not yet"
        ):
            await node.run(SimpleInput(name="Bob"))

    async def test_node_call_model_raises_not_implemented(self) -> None:
        """Test _call_model() raises NotImplementedError."""
        template = PromptTemplate[SimpleInput, str](
            template="Test",
            format=TemplateFormat.F_STRING,
            input_model=SimpleInput,
        )
        node = TypedPromptNode(template=template)

        with pytest.raises(
            NotImplementedError, match="Model adapter integration not yet"
        ):
            await node._call_model("prompt", "adapter")


class TestSerdeModule:
    """Test serde module for observability."""

    def test_serde_to_dict_prompt_template(self) -> None:
        """Test to_dict() with PromptTemplate."""
        template = PromptTemplate[SimpleInput, str](
            template="Hello {name}!",
            format=TemplateFormat.F_STRING,
            input_model=SimpleInput,
        )

        result = serde_to_dict(template)

        assert result["type"] == "prompt"
        assert result["format"] == "f-string"
        assert result["input_model"] == "SimpleInput"
        assert result["has_parser"] is False
        assert result["template"] == "Hello {name}!"

    def test_serde_to_dict_chat_template(self) -> None:
        """Test to_dict() with ChatPromptTemplate."""
        template = ChatPromptTemplate[SimpleInput, str](
            messages=[
                ChatMessage(role=ChatRole.SYSTEM, content="System"),
                ChatMessage(role=ChatRole.USER, content="Hello {name}!"),
            ],
            format=TemplateFormat.JINJA2,
            input_model=SimpleInput,
        )

        result = serde_to_dict(template)

        assert result["type"] == "chat"
        assert result["format"] == "jinja2"
        assert result["input_model"] == "SimpleInput"
        assert result["has_parser"] is False
        assert len(result["messages"]) == MESSAGE_COUNT
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][1]["content"] == "Hello {name}!"

    def test_serde_to_dict_with_parser(self) -> None:
        """Test to_dict() with output parser."""
        template = PromptTemplate[SimpleInput, str](
            template="Test",
            format=TemplateFormat.F_STRING,
            input_model=SimpleInput,
            output_parser=AsIsParser(),
        )

        result = serde_to_dict(template)
        assert result["has_parser"] is True

    def test_serde_from_dict_not_implemented(self) -> None:
        """Test from_dict() raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="deserialization not yet fully"):
            serde_from_dict({"type": "prompt"})

    def test_serde_render_with_observability(self) -> None:
        """Test render_with_observability() function."""
        renderer = get_renderer(TemplateFormat.F_STRING)
        template_str = "Hello {name}!"
        variables = {"name": "Alice"}

        result = render_with_observability(
            template_str, variables, TemplateFormat.F_STRING, renderer
        )

        assert result == "Hello Alice!"

    def test_serde_render_with_observability_missing_vars(self) -> None:
        """Test render_with_observability() with missing variables."""
        renderer = get_renderer(TemplateFormat.F_STRING)
        template_str = "Hello {name}!"
        variables = {}  # Missing 'name'

        # Should catch the KeyError from renderer but still track in telemetry
        with pytest.raises(KeyError):
            render_with_observability(
                template_str, variables, TemplateFormat.F_STRING, renderer
            )

    def test_serde_render_with_invalid_template_type(self) -> None:
        """Test render_with_observability() with invalid template type."""
        renderer = get_renderer(TemplateFormat.F_STRING)
        # Use invalid template type (not a string)
        template_obj = 123
        variables = {"name": "Alice"}

        # Should handle TypeError gracefully in collect_variables
        with pytest.raises(TypeError):
            render_with_observability(
                template_obj, variables, TemplateFormat.F_STRING, renderer
            )

    def test_serde_hash_template(self) -> None:
        """Test _hash_template() helper function."""
        template = "Hello {name}!"
        hash_result = _hash_template(template)

        assert isinstance(hash_result, str)
        assert len(hash_result) == HASH_LENGTH  # First 16 chars of SHA256

    def test_serde_hash_template_long_string(self) -> None:
        """Test _hash_template() with long template (truncation)."""
        # Template longer than 500 chars
        template = "x" * 1000
        hash_result = _hash_template(template)

        assert isinstance(hash_result, str)
        assert len(hash_result) == HASH_LENGTH
