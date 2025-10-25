"""Comprehensive streaming example showing multiple patterns.

This example demonstrates:
1. Basic streaming with token visibility
2. Structured output with partial fields
3. Non-streaming convenience wrappers
4. Streaming helpers (iter_tokens, collect_all_tokens)
5. Progress item observation
"""

from pydantic import BaseModel
from pydantic_ai import Agent

from pydantic_flow import AgentNode
from pydantic_flow import LLMNode
from pydantic_flow import ProgressType
from pydantic_flow import StreamEnd
from pydantic_flow import TokenChunk
from pydantic_flow import collect_all_tokens
from pydantic_flow import iter_tokens


class Query(BaseModel):
    """User query."""

    question: str


class StructuredAnswer(BaseModel):
    """Structured LLM response."""

    answer: str
    confidence: float
    sources: list[str]


# Create user-supplied pydantic-ai agents
text_agent: Agent[None, str] = Agent(
    "openai:gpt-4",
    instructions="Be concise and helpful. Answer in one sentence.",
)

structured_agent: Agent[None, StructuredAnswer] = Agent(  # type: ignore[assignment]
    "openai:gpt-4",
    output_type=StructuredAnswer,
    instructions="Provide a structured answer with confidence and sources.",
)


async def example_1_basic_streaming():
    """Demonstrate basic token streaming."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Token Streaming")
    print("=" * 60)

    node = AgentNode[Query, str](
        agent=text_agent,
        prompt_template="{question}",
        name="basic_agent",
    )

    query = Query(question="What is the capital of France?")

    print("\nStreaming tokens as they arrive:")
    print("-" * 60)

    # Stream and display tokens
    async for token in iter_tokens(node.astream(query)):
        print(token, end="", flush=True)

    print("\n" + "-" * 60)


async def example_2_structured_output():
    """Demonstrate structured output streaming."""
    print("\n" + "=" * 60)
    print("Example 2: Structured Output")
    print("=" * 60)

    node = LLMNode[Query, StructuredAnswer](
        agent=structured_agent,
        prompt_template="Question: {question}",
        name="structured_agent",
    )

    query = Query(question="What causes seasons on Earth?")

    print("\nStreaming structured response:")
    print("-" * 60)

    # Observe all progress items
    async for item in node.astream(query):
        if isinstance(item, TokenChunk):
            print(item.text, end="", flush=True)
        elif isinstance(item, StreamEnd) and item.result_preview:
            print("\n\nFinal structured result:")
            print(f"  Answer: {item.result_preview.get('answer', 'N/A')}")
            print(f"  Confidence: {item.result_preview.get('confidence', 0)}")
            print(f"  Sources: {item.result_preview.get('sources', [])}")

    print("-" * 60)


async def example_3_non_streaming():
    """Demonstrate non-streaming convenience."""
    print("\n" + "=" * 60)
    print("Example 3: Non-Streaming Convenience")
    print("=" * 60)

    node = AgentNode[Query, str](
        agent=text_agent,
        prompt_template="{question}",
        name="simple_agent",
    )

    query = Query(question="What is photosynthesis?")

    print("\nCalling run() (internally consumes stream):")
    print("-" * 60)

    # Get final result without observing stream
    result = await node.run(query)
    print(result)

    print("-" * 60)


async def example_4_collect_tokens():
    """Demonstrate collecting all tokens."""
    print("\n" + "=" * 60)
    print("Example 4: Collect All Tokens")
    print("=" * 60)

    node = AgentNode[Query, str](
        agent=text_agent,
        prompt_template="{question}",
        name="collect_agent",
    )

    query = Query(question="What is machine learning?")

    print("\nCollecting all tokens into a string:")
    print("-" * 60)

    # Collect all tokens at once
    full_text = await collect_all_tokens(node.astream(query))
    print(full_text)
    print(f"\nTotal length: {len(full_text)} characters")

    print("-" * 60)


async def example_5_progress_observation():
    """Demonstrate observing all progress items."""
    print("\n" + "=" * 60)
    print("Example 5: Progress Item Observation")
    print("=" * 60)

    node = AgentNode[Query, str](
        agent=text_agent,
        prompt_template="{question}",
        name="observe_agent",
    )

    query = Query(question="What is quantum computing?")

    print("\nObserving all progress items:")
    print("-" * 60)

    # Track different progress types
    progress_counts = {}

    async for item in node.astream(query):
        # Count progress types
        progress_counts[item.type] = progress_counts.get(item.type, 0) + 1

        # Display progress
        if item.type == ProgressType.START:
            print(f"[{item.timestamp.strftime('%H:%M:%S')}] Stream started")
        elif isinstance(item, TokenChunk):
            print(item.text, end="", flush=True)
        elif item.type == ProgressType.END:
            print(f"\n[{item.timestamp.strftime('%H:%M:%S')}] Stream ended")

    print("\nProgress summary:")
    for progress_type, count in progress_counts.items():
        print(f"  {progress_type}: {count}")

    print("-" * 60)


async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Pydantic-Flow Streaming Examples")
    print("=" * 60)
    print("\nNote: These examples require OpenAI API key to be set.")
    print("Set OPENAI_API_KEY environment variable to run.\n")

    # Uncomment to run when API key is available:
    # await example_1_basic_streaming()
    # await example_2_structured_output()
    # await example_3_non_streaming()
    # await example_4_collect_tokens()
    # await example_5_progress_observation()

    print("\nTo run these examples:")
    print("1. Set your OPENAI_API_KEY environment variable")
    print("2. Uncomment the example calls in main()")
    print("3. Run: python examples/streaming_comprehensive.py")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
