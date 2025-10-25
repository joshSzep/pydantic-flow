"""Example: Simple streaming LLM node with pydantic-ai agent.

This example demonstrates:
1. Using a user-supplied pydantic-ai agent
2. Streaming tokens as they arrive
3. Getting the final result by consuming the stream
"""

from pydantic import BaseModel
from pydantic_ai import Agent

from pydantic_flow.nodes.agent import AgentNode
from pydantic_flow.streaming.helpers import collect_all_tokens
from pydantic_flow.streaming.helpers import iter_tokens


class Query(BaseModel):
    """User query input."""

    question: str


# Create a user-supplied pydantic-ai agent
agent = Agent(
    "openai:gpt-4",
    instructions="Be concise and helpful. Answer in one sentence.",
)


async def demo_streaming():
    """Demonstrate streaming execution."""
    # Create an agent node
    node = AgentNode[Query, str](
        agent=agent,
        prompt_template="{question}",
        name="answer_node",
    )

    # Create input
    query = Query(question="What is the capital of France?")

    print("Streaming tokens as they arrive:")
    print("-" * 50)

    # Stream and display tokens
    async for token in iter_tokens(node.astream(query)):
        print(token, end="", flush=True)

    print("\n" + "-" * 50)


async def demo_non_streaming():
    """Demonstrate non-streaming execution (consumes stream internally)."""
    node = AgentNode[Query, str](
        agent=agent,
        prompt_template="{question}",
        name="answer_node",
    )

    query = Query(question="What is the capital of France?")

    print("\nNon-streaming (final result only):")
    print("-" * 50)

    # Run synchronously - internally consumes the stream
    result = await node.run(query)
    print(result)

    print("-" * 50)


async def demo_collect_tokens():
    """Demonstrate collecting all tokens into a single string."""
    node = AgentNode[Query, str](
        agent=agent,
        prompt_template="{question}",
        name="answer_node",
    )

    query = Query(question="What is the capital of France?")

    print("\nCollecting all tokens:")
    print("-" * 50)

    # Collect all tokens into a string
    full_text = await collect_all_tokens(node.astream(query))
    print(full_text)

    print("-" * 50)


if __name__ == "__main__":
    print("Pydantic-Flow Streaming Example")
    print("=" * 50)

    # Run demos (these would need API keys configured)
    # asyncio.run(demo_streaming())
    # asyncio.run(demo_non_streaming())
    # asyncio.run(demo_collect_tokens())

    print("\nNote: Uncomment the asyncio.run() calls above to run the demos.")
    print("Make sure to set your OpenAI API key first.")
