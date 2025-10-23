"""Comprehensive demonstration of pydantic-flow's prompt library capabilities.

This example showcases:
- Multiple template formats (f-string, Jinja2, Mustache)
- Chat message templates with different roles
- Output parsers (AsIs, JSON model validation, delimited parsing)
- Template validation and variable collection
- Join strategies for chat messages
- Type-safe input/output with Pydantic models
"""

import asyncio

from pydantic import BaseModel
from pydantic import Field

from pydantic_flow.prompt import AsIsParser
from pydantic_flow.prompt import ChatMessage
from pydantic_flow.prompt import ChatPromptTemplate
from pydantic_flow.prompt import ChatRole
from pydantic_flow.prompt import DelimitedParser
from pydantic_flow.prompt import JoinStrategy
from pydantic_flow.prompt import JsonModelParser
from pydantic_flow.prompt import PromptTemplate
from pydantic_flow.prompt import TemplateFormat
from pydantic_flow.prompt import collect_variables
from pydantic_flow.prompt import from_template
from pydantic_flow.prompt import validate_missing_or_extra


# Define input and output models
class UserQuery(BaseModel):
    """User query input for simple prompts."""

    topic: str
    style: str = "professional"
    max_length: int = 100


class PersonInfo(BaseModel):
    """Personal information input."""

    name: str
    age: int
    occupation: str
    interests: list[str] = Field(default_factory=list)


class StoryOutput(BaseModel):
    """Structured output for generated stories."""

    title: str
    genre: str
    plot_summary: str
    word_count: int


class CodeReviewInput(BaseModel):
    """Input for code review scenarios."""

    code_snippet: str
    language: str
    reviewer_name: str = "Senior Engineer"


async def demo_fstring_template() -> None:
    """Demonstrate f-string template rendering."""
    print("\n" + "=" * 60)
    print("DEMO 1: F-String Template")
    print("=" * 60)

    # Create a simple f-string template
    template = PromptTemplate(
        template="Write a {style} article about {topic} in under {max_length} words.",
        format=TemplateFormat.F_STRING,
        input_model=UserQuery,
        output_parser=AsIsParser(),
    )

    query = UserQuery(topic="machine learning", style="beginner-friendly")
    result = template.render(query)

    print(f"\nInput: {query}")
    print(f"Rendered: {result}")

    # Demonstrate variable collection
    variables = collect_variables(
        "Write a {style} article about {topic} in under {max_length} words.",
        TemplateFormat.F_STRING,
    )
    print(f"\nCollected variables: {variables}")


async def demo_jinja2_template() -> None:
    """Demonstrate Jinja2 template with conditionals and loops."""
    print("\n" + "=" * 60)
    print("DEMO 2: Jinja2 Template with Conditionals")
    print("=" * 60)

    template_str = """Generate a bio for {{ name }}.

{% if age >= 18 %}
{{ name }} is a {{ age }}-year-old {{ occupation }}.
{% else %}
{{ name }} is a young {{ occupation }}.
{% endif %}

{% if interests %}
Interests: {{ interests | join(', ') }}.
{% endif %}"""

    template = PromptTemplate(
        template=template_str,
        format=TemplateFormat.JINJA2,
        input_model=PersonInfo,
    )

    # Test with adult
    person1 = PersonInfo(
        name="Alice",
        age=28,
        occupation="software engineer",
        interests=["hiking", "photography", "cooking"],
    )
    result1 = template.render(person1)
    print(f"\nInput 1: {person1}")
    print(f"Rendered 1:\n{result1}")

    # Test with minor (different conditional path)
    person2 = PersonInfo(
        name="Bob",
        age=16,
        occupation="student",
        interests=["gaming", "robotics"],
    )
    result2 = template.render(person2)
    print(f"\nInput 2: {person2}")
    print(f"Rendered 2:\n{result2}")


async def demo_mustache_template() -> None:
    """Demonstrate Mustache (logic-less) template."""
    print("\n" + "=" * 60)
    print("DEMO 3: Mustache Template")
    print("=" * 60)

    template = from_template(
        "Hello {{name}}! You are a {{occupation}}.",
        format=TemplateFormat.MUSTACHE,
        input_model=PersonInfo,
    )

    person = PersonInfo(name="Carol", age=35, occupation="data scientist")
    result = template.render(person)

    print(f"\nInput: {person}")
    print(f"Rendered: {result}")


async def demo_chat_prompt_template() -> None:
    """Demonstrate chat prompt templates with roles."""
    print("\n" + "=" * 60)
    print("DEMO 4: Chat Prompt Template with Roles")
    print("=" * 60)

    messages = [
        ChatMessage(
            role=ChatRole.SYSTEM,
            content="You are {reviewer_name}, an expert {language} developer.",
        ),
        ChatMessage(
            role=ChatRole.USER,
            content="Please review this {language} code:\n\n{code_snippet}",
        ),
        ChatMessage(
            role=ChatRole.ASSISTANT,
            content="I'll provide a thorough code review focusing on best practices.",
        ),
    ]

    chat_template = ChatPromptTemplate(
        messages=messages,
        format=TemplateFormat.F_STRING,
        input_model=CodeReviewInput,
    )

    code_input = CodeReviewInput(
        code_snippet="def add(a, b):\n    return a + b",
        language="Python",
        reviewer_name="Senior Python Developer",
    )

    rendered_messages = chat_template.render_messages(code_input)

    print("\nRendered Chat Messages:")
    for msg in rendered_messages:
        print(f"\n[{msg.role.value.upper()}]")
        print(msg.content)


async def demo_join_strategies() -> None:
    """Demonstrate different message join strategies."""
    print("\n" + "=" * 60)
    print("DEMO 5: Chat Message Join Strategies")
    print("=" * 60)

    messages = [
        ChatMessage(role=ChatRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(role=ChatRole.USER, content="What is 2+2?"),
        ChatMessage(role=ChatRole.ASSISTANT, content="The answer is 4."),
    ]

    chat_template = ChatPromptTemplate(
        messages=messages,
        format=TemplateFormat.F_STRING,
        input_model=PersonInfo,  # Not used, just for type checking
    )

    # Demonstrate different join strategies
    for strategy in [JoinStrategy.OPENAI, JoinStrategy.ANTHROPIC, JoinStrategy.SIMPLE]:
        result = chat_template.join(strategy, messages)
        print(f"\n{strategy.value.upper()} Strategy:")
        print(result)
        print("-" * 40)


async def demo_json_model_parser() -> None:
    """Demonstrate JSON parsing with Pydantic model validation."""
    print("\n" + "=" * 60)
    print("DEMO 6: JSON Model Parser with Validation")
    print("=" * 60)

    parser = JsonModelParser(StoryOutput, strict=True)

    # Simulate LLM JSON output
    json_output = """{
    "title": "The Lost Key",
    "genre": "mystery",
    "plot_summary": "A detective searches for a mysterious ancient key.",
    "word_count": 5000
}"""

    print(f"Raw JSON output:\n{json_output}")

    parsed = await parser.parse(json_output)
    print(f"\nParsed model: {parsed}")
    print(f"Type: {type(parsed)}")
    print(f"Title: {parsed.title}")
    print(f"Genre: {parsed.genre}")


async def demo_delimited_parser() -> None:
    """Demonstrate delimited text parsing."""
    print("\n" + "=" * 60)
    print("DEMO 7: Delimited Parser")
    print("=" * 60)

    parser = DelimitedParser(sep="|")

    # Simulate LLM output with pipe-delimited values
    delimited_output = "Mystery | The Lost Key | An ancient secret awaits | 5000"

    print(f"Raw delimited output: {delimited_output}")

    parsed = await parser.parse(delimited_output)
    print(f"\nParsed dictionary: {parsed}")
    for key, value in parsed.items():
        print(f"  [{key}]: {value}")


async def demo_template_validation() -> None:
    """Demonstrate template variable validation."""
    print("\n" + "=" * 60)
    print("DEMO 8: Template Variable Validation")
    print("=" * 60)

    template_str = "Hello {name}, you are {age} years old and work as a {occupation}."
    required_vars = collect_variables(template_str, TemplateFormat.F_STRING)

    print(f"Template: {template_str}")
    print(f"Required variables: {required_vars}")

    # Valid case
    valid_data = {"name": "Alice", "age": 30, "occupation": "engineer"}
    try:
        validate_missing_or_extra(required_vars, valid_data)
        print(f"\n✓ Valid data: {valid_data}")
    except ValueError as e:
        print(f"\n✗ Validation error: {e}")

    # Missing variable case
    missing_data = {"name": "Bob", "age": 25}
    try:
        validate_missing_or_extra(required_vars, missing_data)
        print(f"\n✓ Valid data: {missing_data}")
    except ValueError as e:
        print(f"\n✗ Validation error: {e}")

    # Extra variable case
    extra_data = {
        "name": "Carol",
        "age": 28,
        "occupation": "designer",
        "location": "NYC",
    }
    try:
        validate_missing_or_extra(required_vars, extra_data)
        print(f"\n✓ Valid data: {extra_data}")
    except ValueError as e:
        print(f"\n✗ Validation error: {e}")


async def demo_render_and_parse() -> None:
    """Demonstrate combined rendering and parsing."""
    print("\n" + "=" * 60)
    print("DEMO 9: Combined Render and Parse")
    print("=" * 60)

    # Create template that expects JSON output
    template = PromptTemplate(
        template=(
            "Generate a {style} story about {topic}. "
            "Return as JSON with title, genre, plot_summary, and word_count."
        ),
        format=TemplateFormat.F_STRING,
        input_model=UserQuery,
        output_parser=JsonModelParser(StoryOutput),
    )

    query = UserQuery(topic="space exploration", style="sci-fi")
    rendered = template.render(query)

    print(f"Input: {query}")
    print(f"Rendered prompt: {rendered}")
    print("\nNote: In real usage, this prompt would be sent to an LLM,")
    print("and the output_parser would validate the JSON response.")


async def demo_extra_variables() -> None:
    """Demonstrate rendering with extra variables not in the model."""
    print("\n" + "=" * 60)
    print("DEMO 10: Extra Variables Beyond Model")
    print("=" * 60)

    template = PromptTemplate(
        template="User {name} (ID: {user_id}) wants info about {topic}.",
        format=TemplateFormat.F_STRING,
        input_model=UserQuery,
    )

    query = UserQuery(topic="AI safety")
    extra = {"name": "Alice", "user_id": 12345}

    result = template.render(query, extra=extra)

    print(f"Input model: {query}")
    print(f"Extra variables: {extra}")
    print(f"Rendered: {result}")


async def main() -> None:
    """Run all prompt library demonstrations."""
    print("\n" + "=" * 60)
    print("PYDANTIC-FLOW PROMPT LIBRARY DEMONSTRATIONS")
    print("=" * 60)

    demos = [
        demo_fstring_template,
        demo_jinja2_template,
        demo_mustache_template,
        demo_chat_prompt_template,
        demo_join_strategies,
        demo_json_model_parser,
        demo_delimited_parser,
        demo_template_validation,
        demo_render_and_parse,
        demo_extra_variables,
    ]

    for demo in demos:
        await demo()
        await asyncio.sleep(0.1)  # Small pause between demos

    print("\n" + "=" * 60)
    print("All demonstrations completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
