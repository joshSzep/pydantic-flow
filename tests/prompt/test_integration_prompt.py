"""Integration tests for prompt templates."""

from pydantic import BaseModel

from pydantic_flow.prompt.enums import ChatRole
from pydantic_flow.prompt.enums import JoinStrategy
from pydantic_flow.prompt.enums import TemplateFormat
from pydantic_flow.prompt.parsers import AsIsParser
from pydantic_flow.prompt.parsers import JsonModelParser
from pydantic_flow.prompt.templates import ChatPromptTemplate
from pydantic_flow.prompt.templates import PromptTemplate
from pydantic_flow.prompt.templates import from_template
from pydantic_flow.prompt.types import ChatMessage

EXPECTED_MESSAGE_COUNT = 2
EXPECTED_TEMP = 20.0


class QueryInput(BaseModel):
    """Input model for queries."""

    location: str
    unit: str = "celsius"


class WeatherResult(BaseModel):
    """Weather result model."""

    temperature: float
    condition: str
    location: str


class TestPromptIntegration:
    """Integration tests for prompt templates."""

    def test_simple_prompt_template(self) -> None:
        """Test simple prompt template rendering."""
        template = PromptTemplate[QueryInput, str](
            template="Weather in {location} ({unit})",
            format=TemplateFormat.F_STRING,
            input_model=QueryInput,
        )

        input_data = QueryInput(location="Paris", unit="fahrenheit")
        result = template.render(input_data)

        assert result == "Weather in Paris (fahrenheit)"

    def test_prompt_template_with_extra(self) -> None:
        """Test prompt template with extra variables."""
        template = PromptTemplate[QueryInput, str](
            template="Weather in {location} on {date}",
            format=TemplateFormat.F_STRING,
            input_model=QueryInput,
        )

        input_data = QueryInput(location="Tokyo")
        result = template.render(input_data, extra={"date": "2025-01-15"})

        assert result == "Weather in Tokyo on 2025-01-15"

    def test_chat_prompt_template_rendering(self) -> None:
        """Test chat prompt template rendering."""
        template = ChatPromptTemplate[QueryInput, str](
            messages=[
                ChatMessage(
                    role=ChatRole.SYSTEM,
                    content="You are a weather assistant.",
                ),
                ChatMessage(
                    role=ChatRole.USER,
                    content="What's the weather in {location}?",
                ),
            ],
            format=TemplateFormat.F_STRING,
            input_model=QueryInput,
        )

        input_data = QueryInput(location="London")
        rendered = template.render_messages(input_data)

        assert len(rendered) == EXPECTED_MESSAGE_COUNT
        assert rendered[0].role == ChatRole.SYSTEM
        assert rendered[0].content == "You are a weather assistant."
        assert rendered[1].role == ChatRole.USER
        assert rendered[1].content == "What's the weather in London?"

    def test_from_template_helper(self) -> None:
        """Test from_template helper function."""
        template = from_template(
            template="Hello {name}!",
            format=TemplateFormat.F_STRING,
            input_model=QueryInput,
        )

        assert isinstance(template, PromptTemplate)
        assert template.format == TemplateFormat.F_STRING

    async def test_prompt_with_parser(self) -> None:
        """Test prompt template with output parser."""
        parser = AsIsParser()
        template = PromptTemplate[QueryInput, str](
            template="Weather in {location}",
            format=TemplateFormat.F_STRING,
            input_model=QueryInput,
            output_parser=parser,
        )

        input_data = QueryInput(location="NYC")
        result = await template.render_and_parse(input_data)

        assert result == "Weather in NYC"

    async def test_chat_with_parser(self) -> None:
        """Test chat template with output parser."""
        parser = AsIsParser()
        template = ChatPromptTemplate[QueryInput, str](
            messages=[
                ChatMessage(role=ChatRole.USER, content="Weather in {location}"),
            ],
            format=TemplateFormat.F_STRING,
            input_model=QueryInput,
            output_parser=parser,
        )

        input_data = QueryInput(location="SF")
        result = await template.render_and_parse(input_data)

        assert isinstance(result, str)
        assert "Weather in SF" in result

    def test_mustache_template_integration(self) -> None:
        """Test Mustache template integration."""
        template = PromptTemplate[QueryInput, str](
            template="Weather in {{location}} ({{unit}})",
            format=TemplateFormat.MUSTACHE,
            input_model=QueryInput,
        )

        input_data = QueryInput(location="Berlin", unit="celsius")
        result = template.render(input_data)

        assert result == "Weather in Berlin (celsius)"

    def test_jinja2_template_integration(self) -> None:
        """Test Jinja2 template integration."""
        template = PromptTemplate[QueryInput, str](
            template="Weather in {{ location }} ({{ unit }})",
            format=TemplateFormat.JINJA2,
            input_model=QueryInput,
        )

        input_data = QueryInput(location="Madrid", unit="celsius")
        result = template.render(input_data)

        assert result == "Weather in Madrid (celsius)"

    def test_chat_joining_integration(self) -> None:
        """Test chat message joining integration."""
        template = ChatPromptTemplate[QueryInput, str](
            messages=[
                ChatMessage(role=ChatRole.SYSTEM, content="System message"),
                ChatMessage(role=ChatRole.USER, content="Query about {location}"),
            ],
            format=TemplateFormat.F_STRING,
            input_model=QueryInput,
        )

        input_data = QueryInput(location="Athens")
        rendered = template.render_messages(input_data)
        joined = template.join(JoinStrategy.OPENAI, rendered)

        assert "system: System message" in joined
        assert "user: Query about Athens" in joined

    async def test_json_parser_integration(self) -> None:
        """Test JSON parser integration with template."""
        parser = JsonModelParser[WeatherResult](WeatherResult)
        template = PromptTemplate[QueryInput, WeatherResult](
            template=(
                '{{"temperature": 20.0, "condition": "sunny", '
                '"location": "{location}"}}'
            ),
            format=TemplateFormat.F_STRING,
            input_model=QueryInput,
            output_parser=parser,
        )

        input_data = QueryInput(location="Oslo")
        result = await template.render_and_parse(input_data)

        assert isinstance(result, WeatherResult)
        assert result.location == "Oslo"
        assert result.temperature == EXPECTED_TEMP
        assert result.condition == "sunny"
