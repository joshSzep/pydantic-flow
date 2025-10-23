"""Tests for chat message joining strategies."""

from pydantic import BaseModel
import pytest

from pydantic_flow.prompt.enums import ChatRole
from pydantic_flow.prompt.enums import JoinStrategy
from pydantic_flow.prompt.enums import TemplateFormat
from pydantic_flow.prompt.templates import ChatPromptTemplate
from pydantic_flow.prompt.types import ChatMessage


class SimpleInput(BaseModel):
    """Simple input model for testing."""

    topic: str


class TestChatJoiners:
    """Test chat message joining strategies."""

    def test_join_openai_style(self) -> None:
        """Test OpenAI-style message joining."""
        messages = [
            ChatMessage(role=ChatRole.SYSTEM, content="You are helpful."),
            ChatMessage(role=ChatRole.USER, content="Hello!"),
            ChatMessage(role=ChatRole.ASSISTANT, content="Hi there!"),
        ]

        template = ChatPromptTemplate[SimpleInput, str](
            messages=messages,
            format=TemplateFormat.F_STRING,
            input_model=SimpleInput,
        )

        result = template.join(JoinStrategy.OPENAI, messages)

        expected = "system: You are helpful.\nuser: Hello!\nassistant: Hi there!"
        assert result == expected

    def test_join_anthropic_style(self) -> None:
        """Test Anthropic-style message joining."""
        messages = [
            ChatMessage(role=ChatRole.SYSTEM, content="You are helpful."),
            ChatMessage(role=ChatRole.USER, content="Hello!"),
            ChatMessage(role=ChatRole.ASSISTANT, content="Hi there!"),
        ]

        template = ChatPromptTemplate[SimpleInput, str](
            messages=messages,
            format=TemplateFormat.F_STRING,
            input_model=SimpleInput,
        )

        result = template.join(JoinStrategy.ANTHROPIC, messages)

        expected = "system: You are helpful.\n\nuser: Hello!\n\nassistant: Hi there!"
        assert result == expected

    def test_join_simple_style(self) -> None:
        """Test simple message joining."""
        messages = [
            ChatMessage(role=ChatRole.SYSTEM, content="You are helpful."),
            ChatMessage(role=ChatRole.USER, content="Hello!"),
            ChatMessage(role=ChatRole.ASSISTANT, content="Hi there!"),
        ]

        template = ChatPromptTemplate[SimpleInput, str](
            messages=messages,
            format=TemplateFormat.F_STRING,
            input_model=SimpleInput,
        )

        result = template.join(JoinStrategy.SIMPLE, messages)

        expected = "You are helpful.\nHello!\nHi there!"
        assert result == expected

    def test_join_single_message(self) -> None:
        """Test joining single message."""
        messages = [ChatMessage(role=ChatRole.USER, content="Hello!")]

        template = ChatPromptTemplate[SimpleInput, str](
            messages=messages,
            format=TemplateFormat.F_STRING,
            input_model=SimpleInput,
        )

        result = template.join(JoinStrategy.OPENAI, messages)
        assert result == "user: Hello!"

    def test_join_empty_messages(self) -> None:
        """Test joining empty message list."""
        template = ChatPromptTemplate[SimpleInput, str](
            messages=[],
            format=TemplateFormat.F_STRING,
            input_model=SimpleInput,
        )

        result = template.join(JoinStrategy.OPENAI, [])
        assert result == ""

    def test_join_with_tool_role(self) -> None:
        """Test joining messages with tool role."""
        messages = [
            ChatMessage(role=ChatRole.USER, content="What's the weather?"),
            ChatMessage(role=ChatRole.TOOL, content='{"temp": 72}'),
            ChatMessage(role=ChatRole.ASSISTANT, content="It's 72°F."),
        ]

        template = ChatPromptTemplate[SimpleInput, str](
            messages=messages,
            format=TemplateFormat.F_STRING,
            input_model=SimpleInput,
        )

        result = template.join(JoinStrategy.OPENAI, messages)
        assert "tool:" in result
        assert '{"temp": 72}' in result

    def test_join_uses_template_messages_by_default(self) -> None:
        """Test that join uses template messages when rendered is None."""
        messages = [
            ChatMessage(role=ChatRole.USER, content="Test message"),
        ]

        template = ChatPromptTemplate[SimpleInput, str](
            messages=messages,
            format=TemplateFormat.F_STRING,
            input_model=SimpleInput,
        )

        result = template.join(JoinStrategy.OPENAI)
        assert result == "user: Test message"

    def test_unsupported_join_strategy(self) -> None:
        """Test error on unsupported join strategy."""
        messages = [ChatMessage(role=ChatRole.USER, content="Test")]

        template = ChatPromptTemplate[SimpleInput, str](
            messages=messages,
            format=TemplateFormat.F_STRING,
            input_model=SimpleInput,
        )

        with pytest.raises(ValueError) as exc_info:
            template.join("unsupported", messages)  # type: ignore[arg-type]

        assert "Unsupported" in str(exc_info.value)
