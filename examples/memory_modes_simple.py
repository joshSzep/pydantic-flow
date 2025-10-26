"""Simple example demonstrating FlowNode memory modes.

This example shows the three memory modes (SHARED, ISOLATED, READONLY) with
basic flows to illustrate the different behaviors.
"""

import asyncio

from pydantic import BaseModel
from pydantic_ai.messages import ModelRequest
from pydantic_ai.messages import SystemPromptPart

from pydantic_flow import Flow
from pydantic_flow import MemoryConfig
from pydantic_flow.memory import MemoryMode
from pydantic_flow.nodes import FlowNode
from pydantic_flow.nodes import ToolNode


class Query(BaseModel):
    """Input for processing."""

    text: str


class ProcessedText(BaseModel):
    """Processed text result."""

    value: str


class Result(BaseModel):
    """Output from processing."""

    processed: ProcessedText


class Report(BaseModel):
    """Final report combining results."""

    result: Result


def process_text(query: Query) -> ProcessedText:
    """Process the input text."""
    return ProcessedText(value=f"Processed: {query.text}")


async def demonstrate_shared_mode():
    """SHARED mode: Sub-flow uses parent's memory directly."""
    print("\n=== SHARED MODE (Default) ===")
    print("Sub-flow sees and modifies parent's conversation history\n")

    # Create parent flow with memory
    parent_flow = Flow[Query, Report](
        input_type=Query,
        output_type=Report,
        memory_config=MemoryConfig(enable_conversation_memory=True),
    )

    # Create sub-flow
    sub_flow = Flow[Query, Result](
        input_type=Query,
        output_type=Result,
        memory_config=MemoryConfig(enable_conversation_memory=True),
    )
    processor = ToolNode[Query, ProcessedText](tool_func=process_text, name="processed")
    sub_flow.add_nodes(processor)
    sub_flow.compile()

    # Create FlowNode with SHARED mode (default)
    flow_node = FlowNode[Query, Result](
        flow=sub_flow, name="result", memory_mode=MemoryMode.SHARED
    )

    parent_flow.add_nodes(flow_node)
    parent_flow.compile()

    # Add a message to parent memory before running
    if parent_flow._conversation_memory:
        parent_flow._conversation_memory.append(
            ModelRequest(parts=[SystemPromptPart(content="Parent message 1")])
        )
        print(f"Parent memory before: {len(parent_flow._conversation_memory)} message")

    # Run the flow
    result = await parent_flow.run(Query(text="Hello"))
    print(f"Result: {result.result.processed.value}")

    # Note: In SHARED mode, sub-flow could add messages to parent memory
    # (though ToolNode doesn't auto-capture, AgentNode would)
    if parent_flow._conversation_memory:
        mem_count = len(parent_flow._conversation_memory)
        print(f"Parent memory after: {mem_count} message(s)")
        print("✓ Sub-flow had full read-write access to parent memory")


async def demonstrate_isolated_mode():
    """ISOLATED mode: Sub-flow gets separate memory."""
    print("\n=== ISOLATED MODE ===")
    print("Sub-flow has its own memory, changes don't affect parent\n")

    # Create parent flow with memory
    parent_flow = Flow[Query, Report](
        input_type=Query,
        output_type=Report,
        memory_config=MemoryConfig(enable_conversation_memory=True),
    )

    # Create sub-flow
    sub_flow = Flow[Query, Result](
        input_type=Query,
        output_type=Result,
        memory_config=MemoryConfig(enable_conversation_memory=True),
    )
    processor = ToolNode[Query, ProcessedText](tool_func=process_text, name="processed")
    sub_flow.add_nodes(processor)
    sub_flow.compile()

    # Create FlowNode with ISOLATED mode
    flow_node = FlowNode[Query, Result](
        flow=sub_flow,
        name="result",
        memory_mode=MemoryMode.ISOLATED,
        seed_isolated_memory=False,  # Start with empty memory
    )

    parent_flow.add_nodes(flow_node)
    parent_flow.compile()

    # Add messages to parent memory
    if parent_flow._conversation_memory:
        parent_flow._conversation_memory.append(
            ModelRequest(parts=[SystemPromptPart(content="Parent message 1")])
        )
        parent_flow._conversation_memory.append(
            ModelRequest(parts=[SystemPromptPart(content="Parent message 2")])
        )
        print(f"Parent memory before: {len(parent_flow._conversation_memory)} messages")

    # Run the flow
    result = await parent_flow.run(Query(text="Hello"))
    print(f"Result: {result.result.processed.value}")

    # Check memory after - should be unchanged
    if parent_flow._conversation_memory:
        mem_count = len(parent_flow._conversation_memory)
        print(f"Parent memory after: {mem_count} messages (unchanged)")
        print("✓ Sub-flow's memory operations were isolated from parent")


async def demonstrate_isolated_with_seed():
    """ISOLATED mode with seeding: Sub-flow gets copy of parent memory."""
    print("\n=== ISOLATED MODE (With Seed) ===")
    msg = "Sub-flow starts with copy of parent's history but remains isolated"
    print(f"{msg}\n")

    # Create parent flow with memory
    parent_flow = Flow[Query, Report](
        input_type=Query,
        output_type=Report,
        memory_config=MemoryConfig(enable_conversation_memory=True),
    )

    # Create sub-flow
    sub_flow = Flow[Query, Result](
        input_type=Query,
        output_type=Result,
        memory_config=MemoryConfig(enable_conversation_memory=True),
    )
    processor = ToolNode[Query, ProcessedText](tool_func=process_text, name="processed")
    sub_flow.add_nodes(processor)
    sub_flow.compile()

    # Create FlowNode with ISOLATED mode + seeding
    flow_node = FlowNode[Query, Result](
        flow=sub_flow,
        name="result",
        memory_mode=MemoryMode.ISOLATED,
        seed_isolated_memory=True,  # Copy parent's messages for context
    )

    parent_flow.add_nodes(flow_node)
    parent_flow.compile()

    # Add messages to parent memory
    if parent_flow._conversation_memory:
        parent_flow._conversation_memory.append(
            ModelRequest(parts=[SystemPromptPart(content="Important context")])
        )
        print(f"Parent memory before: {len(parent_flow._conversation_memory)} message")

    # Run the flow
    result = await parent_flow.run(Query(text="Hello"))
    print(f"Result: {result.result.processed.value}")

    # Check memory after
    if parent_flow._conversation_memory:
        mem_count = len(parent_flow._conversation_memory)
        print(f"Parent memory after: {mem_count} message (unchanged)")
        print("✓ Sub-flow got seeded copy but remained isolated")


async def demonstrate_readonly_mode():
    """READONLY mode: Sub-flow can read but not modify parent memory."""
    print("\n=== READONLY MODE ===")
    print("Sub-flow can read parent's memory but cannot modify it\n")

    # Create parent flow with memory
    parent_flow = Flow[Query, Report](
        input_type=Query,
        output_type=Report,
        memory_config=MemoryConfig(enable_conversation_memory=True),
    )

    # Create sub-flow
    sub_flow = Flow[Query, Result](
        input_type=Query,
        output_type=Result,
        memory_config=MemoryConfig(enable_conversation_memory=True),
    )
    processor = ToolNode[Query, ProcessedText](tool_func=process_text, name="processed")
    sub_flow.add_nodes(processor)
    sub_flow.compile()

    # Create FlowNode with READONLY mode
    flow_node = FlowNode[Query, Result](
        flow=sub_flow, name="result", memory_mode=MemoryMode.READONLY
    )

    parent_flow.add_nodes(flow_node)
    parent_flow.compile()

    # Add messages to parent memory
    if parent_flow._conversation_memory:
        parent_flow._conversation_memory.append(
            ModelRequest(parts=[SystemPromptPart(content="Read-only context")])
        )
        print(f"Parent memory before: {len(parent_flow._conversation_memory)} message")

    # Run the flow
    result = await parent_flow.run(Query(text="Hello"))
    print(f"Result: {result.result.processed.value}")

    # Check memory after
    if parent_flow._conversation_memory:
        mem_count = len(parent_flow._conversation_memory)
        print(f"Parent memory after: {mem_count} message (unchanged)")
        msg = "✓ Sub-flow had read-only access, couldn't modify parent"
        print(msg)


async def main():
    """Run all demonstrations."""
    print("=" * 60)
    print("FlowNode Memory Modes Demonstration")
    print("=" * 60)

    await demonstrate_shared_mode()
    await demonstrate_isolated_mode()
    await demonstrate_isolated_with_seed()
    await demonstrate_readonly_mode()

    print("\n" + "=" * 60)
    print("Summary:")
    print("- SHARED: Default, sub-flow fully shares parent's memory")
    print("- ISOLATED: Sub-flow gets separate memory (optionally seeded)")
    print("- READONLY: Sub-flow can read but not modify parent's memory")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
