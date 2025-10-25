"""Example: Using PromptNode for streaming LLM calls.

This example shows how to use PromptNode with different template formats:
1. Simple string templates (backward compatible)
2. Full PromptTemplate objects with type safety
3. ChatPromptTemplate for multi-message prompts
4. Output parsers for structured data extraction
"""

import asyncio

from pydantic import BaseModel

from pydantic_flow import ChatMessage
from pydantic_flow import ChatPromptTemplate
from pydantic_flow import ChatRole
from pydantic_flow import PromptConfig
from pydantic_flow import PromptNode
from pydantic_flow import PromptTemplate
from pydantic_flow import TemplateFormat
from pydantic_flow import iter_tokens
from pydantic_flow.prompt import JsonModelParser


class Question(BaseModel):
    """Question input."""

    topic: str
    context: str = ""


class Summary(BaseModel):
    """Structured summary output."""

    key_points: list[str]
    conclusion: str


async def demo_simple_string():
    """Demonstrate PromptNode with simple string template (backward compatible)."""
    print("\n" + "=" * 60)
    print("DEMO 1: Simple String Template (Backward Compatible)")
    print("=" * 60)

    node = PromptNode[Question, str](
        prompt="Explain {topic} in one sentence.",
        config=PromptConfig(
            model="openai:gpt-4",
            system_prompt="You are a helpful teacher. Be concise.",
        ),
        name="simple_explainer",
    )

    question = Question(topic="quantum computing")

    print("\nStreaming explanation:")
    print("-" * 60)

    async for token in iter_tokens(node.astream(question)):
        print(token, end="", flush=True)

    print("\n" + "-" * 60)


async def demo_jinja2_template():
    """Demonstrate PromptNode with Jinja2 template for conditional logic."""
    print("\n" + "=" * 60)
    print("DEMO 2: Jinja2 Template with Conditionals")
    print("=" * 60)

    template = PromptTemplate[Question, str](
        template="""Explain {{ topic }}.
{% if context %}
Context: {{ context }}
{% endif %}
Provide a brief explanation.""",
        format=TemplateFormat.JINJA2,
        input_model=Question,
    )

    node = PromptNode[Question, str](
        prompt=template,
        config=PromptConfig(
            model="openai:gpt-4",
            system_prompt="You are a helpful teacher.",
        ),
        name="jinja_explainer",
    )

    question = Question(topic="neural networks", context="for a high school student")

    print("\nStreaming explanation with context:")
    print("-" * 60)

    async for token in iter_tokens(node.astream(question)):
        print(token, end="", flush=True)

    print("\n" + "-" * 60)


async def demo_chat_template():
    """Demonstrate PromptNode with ChatPromptTemplate."""
    print("\n" + "=" * 60)
    print("DEMO 3: Chat Prompt Template")
    print("=" * 60)

    chat_template = ChatPromptTemplate[Question, str](
        messages=[
            ChatMessage(
                role=ChatRole.SYSTEM,
                content="You are an expert tutor specializing in {topic}.",
            ),
            ChatMessage(
                role=ChatRole.USER,
                content="Please explain {topic} to me in simple terms.",
            ),
        ],
        format=TemplateFormat.F_STRING,
        input_model=Question,
    )

    node = PromptNode[Question, str](
        prompt=chat_template,
        config=PromptConfig(model="openai:gpt-4"),
        name="chat_explainer",
    )

    question = Question(topic="machine learning")

    print("\nStreaming chat response:")
    print("-" * 60)

    async for token in iter_tokens(node.astream(question)):
        print(token, end="", flush=True)

    print("\n" + "-" * 60)


async def demo_output_parser():
    """Demonstrate PromptNode with structured output parser."""
    print("\n" + "=" * 60)
    print("DEMO 4: Output Parser for Structured Data")
    print("=" * 60)

    template = PromptTemplate[Question, str](
        template="""Summarize the key points about {topic}.
Return your response as JSON with this structure:
{{
    "key_points": ["point1", "point2", "point3"],
    "conclusion": "brief conclusion"
}}""",
        format=TemplateFormat.F_STRING,
        input_model=Question,
    )

    node = PromptNode[Question, Summary](
        prompt=template,
        config=PromptConfig(
            model="openai:gpt-4",
            system_prompt="Provide concise, structured summaries.",
        ),
        output_parser=JsonModelParser(model=Summary),
        name="structured_summarizer",
    )

    question = Question(topic="climate change")

    print("\nGetting structured summary...")
    result = await node.run(question)

    print("\nParsed result:")
    print(f"Type: {type(result)}")
    if isinstance(result, Summary):
        print("\nKey Points:")
        for i, point in enumerate(result.key_points, 1):
            print(f"  {i}. {point}")
        print(f"\nConclusion: {result.conclusion}")


async def demo():
    """Run all demonstrations."""
    print("PromptNode Integration Examples")
    print("=" * 60)
    print("\nNote: Set OPENAI_API_KEY to run these examples.")
    print("Uncomment the demo functions below to try them.\n")

    # Uncomment to run:
    # await demo_simple_string()
    # await demo_jinja2_template()
    # await demo_chat_template()
    # await demo_output_parser()


if __name__ == "__main__":
    asyncio.run(demo())
