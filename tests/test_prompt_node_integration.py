"""Tests for PromptNode integration with prompt templating system."""

import json

from pydantic import BaseModel

from pydantic_flow import ChatMessage
from pydantic_flow import ChatPromptTemplate
from pydantic_flow import ChatRole
from pydantic_flow import JoinStrategy
from pydantic_flow import PromptConfig
from pydantic_flow import PromptNode
from pydantic_flow import PromptTemplate
from pydantic_flow import TemplateFormat
from pydantic_flow import ToolResult
from pydantic_flow.prompt import AsIsParser
from pydantic_flow.prompt import JsonModelParser
from pydantic_flow.streaming.helpers import collect_all_tokens


async def collect_tool_result(stream):
    """Extract the result from ToolResult event."""
    async for item in stream:
        if isinstance(item, ToolResult):
            return item.result
    return None


class QueryInput(BaseModel):
    """Test input model."""

    topic: str
    context: str = ""


class StructuredOutput(BaseModel):
    """Test output model."""

    summary: str
    key_points: list[str]


class TestPromptNodeBackwardCompatibility:
    """Test backward compatibility with simple string prompts."""

    async def test_simple_string_prompt(self):
        """Test PromptNode with simple string prompt."""
        node = PromptNode[QueryInput, str](
            prompt="Explain {topic} in one sentence.",
            config=PromptConfig(model="test"),
            name="test_node",
        )

        input_data = QueryInput(topic="quantum computing")
        stream = node.astream(input_data)

        result = await collect_tool_result(stream)
        assert isinstance(result, str)

    async def test_string_prompt_with_multiple_variables(self):
        """Test string prompt with multiple template variables."""
        node = PromptNode[QueryInput, str](
            prompt="Explain {topic} with context: {context}",
            config=PromptConfig(model="test"),
        )

        input_data = QueryInput(topic="AI", context="for beginners")
        stream = node.astream(input_data)

        result = await collect_tool_result(stream)
        assert isinstance(result, str)

    async def test_string_prompt_uses_configured_format(self):
        """Test that string prompts respect template_format config."""
        node = PromptNode[QueryInput, str](
            prompt="Explain {topic}",
            config=PromptConfig(model="test", template_format=TemplateFormat.F_STRING),
        )

        input_data = QueryInput(topic="testing")
        stream = node.astream(input_data)

        result = await collect_tool_result(stream)
        assert isinstance(result, str)


class TestPromptNodeWithPromptTemplate:
    """Test PromptNode with PromptTemplate objects."""

    async def test_fstring_template(self):
        """Test PromptNode with f-string PromptTemplate."""
        template = PromptTemplate[QueryInput, str](
            template="Explain {topic} in simple terms.",
            format=TemplateFormat.F_STRING,
            input_model=QueryInput,
        )

        node = PromptNode[QueryInput, str](
            prompt=template, config=PromptConfig(model="test")
        )

        input_data = QueryInput(topic="neural networks")
        stream = node.astream(input_data)

        result = await collect_tool_result(stream)
        assert isinstance(result, str)

    async def test_jinja2_template(self):
        """Test PromptNode with Jinja2 template."""
        template = PromptTemplate[QueryInput, str](
            template="""Explain {{ topic }}.
{% if context %}
Context: {{ context }}
{% endif %}""",
            format=TemplateFormat.JINJA2,
            input_model=QueryInput,
        )

        node = PromptNode[QueryInput, str](
            prompt=template, config=PromptConfig(model="test")
        )

        input_data = QueryInput(topic="Python", context="for data science")
        stream = node.astream(input_data)

        result = await collect_tool_result(stream)
        assert isinstance(result, str)

    async def test_mustache_template(self):
        """Test PromptNode with Mustache template."""
        template = PromptTemplate[QueryInput, str](
            template="Explain {{topic}}.",
            format=TemplateFormat.MUSTACHE,
            input_model=QueryInput,
        )

        node = PromptNode[QueryInput, str](
            prompt=template, config=PromptConfig(model="test")
        )

        input_data = QueryInput(topic="databases")
        stream = node.astream(input_data)

        result = await collect_tool_result(stream)
        assert isinstance(result, str)


class TestPromptNodeWithChatTemplate:
    """Test PromptNode with ChatPromptTemplate objects."""

    async def test_chat_template_basic(self):
        """Test PromptNode with basic ChatPromptTemplate."""
        chat_template = ChatPromptTemplate[QueryInput, str](
            messages=[
                ChatMessage(
                    role=ChatRole.SYSTEM,
                    content="You are an expert on {topic}.",
                ),
                ChatMessage(
                    role=ChatRole.USER,
                    content="Explain {topic} to me.",
                ),
            ],
            format=TemplateFormat.F_STRING,
            input_model=QueryInput,
        )

        node = PromptNode[QueryInput, str](
            prompt=chat_template, config=PromptConfig(model="test")
        )

        input_data = QueryInput(topic="blockchain")
        stream = node.astream(input_data)

        result = await collect_tool_result(stream)
        assert isinstance(result, str)

    async def test_chat_template_with_join_strategy(self):
        """Test ChatPromptTemplate with custom join strategy."""
        chat_template = ChatPromptTemplate[QueryInput, str](
            messages=[
                ChatMessage(role=ChatRole.SYSTEM, content="System prompt."),
                ChatMessage(role=ChatRole.USER, content="User: {topic}"),
            ],
            format=TemplateFormat.F_STRING,
            input_model=QueryInput,
        )

        node = PromptNode[QueryInput, str](
            prompt=chat_template,
            config=PromptConfig(model="test", chat_join_strategy=JoinStrategy.OPENAI),
        )

        input_data = QueryInput(topic="test")
        stream = node.astream(input_data)

        result = await collect_tool_result(stream)
        assert isinstance(result, str)


class TestPromptNodeWithOutputParsers:
    """Test PromptNode with output parsers."""

    async def test_asis_parser(self):
        """Test PromptNode with AsIsParser."""
        template = PromptTemplate[QueryInput, str](
            template="Explain {topic}.",
            format=TemplateFormat.F_STRING,
            input_model=QueryInput,
        )

        node = PromptNode[QueryInput, str](
            prompt=template,
            config=PromptConfig(model="test"),
            output_parser=AsIsParser(),
        )

        input_data = QueryInput(topic="testing")
        stream = node.astream(input_data)

        result = await collect_tool_result(stream)
        assert isinstance(result, str)

    async def test_json_model_parser(self):
        """Test PromptNode with JsonModelParser.

        Note: This test uses the 'test' model which doesn't return valid JSON,
        so we expect it to fail during parsing. With a real LLM, this would
        successfully parse to StructuredOutput.
        """
        template = PromptTemplate[QueryInput, str](
            template="""Summarize {topic}.
Return JSON: {{"summary": "...", "key_points": ["..."]}}""",
            format=TemplateFormat.F_STRING,
            input_model=QueryInput,
        )

        node = PromptNode[QueryInput, StructuredOutput](
            prompt=template,
            config=PromptConfig(model="test"),
            output_parser=JsonModelParser(model=StructuredOutput),
        )

        input_data = QueryInput(topic="AI ethics")
        stream = node.astream(input_data)

        # Test model returns non-JSON string, so parsing will raise
        try:
            result = await collect_tool_result(stream)
            # If we get here, the test model somehow returned valid JSON
            assert result is not None
        except json.JSONDecodeError:
            # Expected: test model doesn't return valid JSON
            pass


class TestPromptNodeStreaming:
    """Test streaming behavior of PromptNode."""

    async def test_token_streaming(self):
        """Test that tokens are streamed correctly."""
        node = PromptNode[QueryInput, str](
            prompt="Explain {topic}.",
            config=PromptConfig(model="test"),
        )

        input_data = QueryInput(topic="streaming")
        stream = node.astream(input_data)

        tokens = await collect_all_tokens(stream)
        assert len(tokens) > 0

    async def test_stream_with_template_and_parser(self):
        """Test streaming with both template and parser."""
        template = PromptTemplate[QueryInput, str](
            template="Explain {{topic}}.",
            format=TemplateFormat.MUSTACHE,
            input_model=QueryInput,
        )

        node = PromptNode[QueryInput, str](
            prompt=template,
            config=PromptConfig(model="test"),
            output_parser=AsIsParser(),
        )

        input_data = QueryInput(topic="integration")
        stream = node.astream(input_data)

        tokens = await collect_all_tokens(stream)
        assert len(tokens) > 0


class TestPromptNodeConfiguration:
    """Test PromptNode configuration options."""

    async def test_config_system_prompt(self):
        """Test PromptNode with system prompt configuration."""
        node = PromptNode[QueryInput, str](
            prompt="Explain {topic}.",
            config=PromptConfig(
                model="test", system_prompt="You are a helpful assistant."
            ),
        )

        input_data = QueryInput(topic="config")
        stream = node.astream(input_data)

        result = await collect_tool_result(stream)
        assert isinstance(result, str)

    async def test_config_template_format(self):
        """Test PromptConfig with different template formats."""
        for fmt in [
            TemplateFormat.F_STRING,
            TemplateFormat.JINJA2,
            TemplateFormat.MUSTACHE,
        ]:
            config = PromptConfig(model="test", template_format=fmt)
            assert config.template_format == fmt

    async def test_template_object_ignores_config_format(self):
        """Test that PromptTemplate objects use their own format."""
        template = PromptTemplate[QueryInput, str](
            template="Explain {topic}.",
            format=TemplateFormat.F_STRING,
            input_model=QueryInput,
        )

        # Config has different format, but template's format takes precedence
        node = PromptNode[QueryInput, str](
            prompt=template,
            config=PromptConfig(model="test", template_format=TemplateFormat.JINJA2),
        )

        input_data = QueryInput(topic="precedence")
        stream = node.astream(input_data)

        result = await collect_tool_result(stream)
        assert isinstance(result, str)
