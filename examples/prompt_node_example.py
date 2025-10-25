"""Example: Using PromptNode for streaming LLM calls.

This example shows how to use PromptNode, which internally creates
a pydantic-ai agent and provides streaming execution.
"""

from pydantic import BaseModel

from pydantic_flow import PromptConfig
from pydantic_flow import PromptNode
from pydantic_flow import iter_tokens


class Question(BaseModel):
    """Question input."""

    topic: str


async def demo():
    """Demonstrate PromptNode streaming."""
    # Create a PromptNode with custom configuration
    node = PromptNode[Question, str](
        prompt="Explain {topic} in one sentence.",
        config=PromptConfig(
            model="openai:gpt-4",
            system_prompt="You are a helpful teacher. Be concise.",
        ),
        name="explainer",
    )

    question = Question(topic="quantum computing")

    print("Streaming explanation:")
    print("-" * 60)

    # Stream tokens as they arrive
    async for token in iter_tokens(node.astream(question)):
        print(token, end="", flush=True)

    print("\n" + "-" * 60)

    # Or get the complete result
    result = await node.run(question)
    print(f"\nComplete result: {result}")


if __name__ == "__main__":
    print("PromptNode Streaming Example")
    print("=" * 60)
    print("\nNote: Set OPENAI_API_KEY to run this example.")
    print("Uncomment asyncio.run(demo()) below.\n")

    # Uncomment to run:
    # asyncio.run(demo())
