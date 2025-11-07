"""Example demonstrating AgentNode for combining multiple inputs.

This example shows how to use AgentNode with multiple inputs to merge outputs
from multiple upstream nodes and send them to an LLM for processing.
"""

import asyncio

from pydantic import BaseModel

from pydantic_flow import AgentNode
from pydantic_flow import Flow
from pydantic_flow import ToolNode


# Helper to extract result from stream
async def extract_result_from_stream(stream):
    """Extract final result from async stream of progress items."""
    result = None
    async for item in stream:
        if hasattr(item, "result"):
            result = item.result
    return result


class Query(BaseModel):
    """Input query."""

    topic: str


class ResearchData(BaseModel):
    """Research data from first node."""

    facts: list[str]
    sources: list[str]


class AnalysisData(BaseModel):
    """Analysis data from second node."""

    key_points: list[str]
    conclusion: str


class CombinedReport(BaseModel):
    """Final combined report."""

    research: ResearchData
    analysis: AnalysisData
    summary: str


# Mock functions for research and analysis
async def gather_research(query: Query) -> ResearchData:
    """Simulate gathering research data."""
    return ResearchData(
        facts=[
            f"Fact 1 about {query.topic}",
            f"Fact 2 about {query.topic}",
            f"Fact 3 about {query.topic}",
        ],
        sources=[
            "Source A",
            "Source B",
            "Source C",
        ],
    )


async def perform_analysis(query: Query) -> AnalysisData:
    """Simulate performing analysis."""
    return AnalysisData(
        key_points=[
            f"Key point 1 for {query.topic}",
            f"Key point 2 for {query.topic}",
        ],
        conclusion=f"Overall assessment of {query.topic}",
    )


async def main():
    """Run the merge prompt example."""
    print("=" * 60)
    print("AgentNode Multi-Input Example: Combining Research and Analysis")
    print("=" * 60)
    print()

    # Create nodes for gathering research and analysis
    research_node = ToolNode[Query, ResearchData](
        tool_func=gather_research,
        name="research",
    )

    analysis_node = ToolNode[Query, AnalysisData](
        tool_func=perform_analysis,
        name="analysis",
    )

    # Create an AgentNode that combines both outputs
    # The prompt can reference inputs by index {0}, {1} or by field names
    merge_node = AgentNode.from_prompt(
        model="test",  # In production, use "openai:gpt-4" or similar
        prompt_template="""Based on the following information, create a comprehensive summary:

RESEARCH DATA:
Facts: {0.facts}
Sources: {0.sources}

ANALYSIS DATA:
Key Points: {1.key_points}
Conclusion: {1.conclusion}

Provide a clear, concise summary that integrates both the research
findings and analysis.""",
        inputs=(research_node.output, analysis_node.output),
        name="merge_summary",
    )

    # Build the flow
    flow = Flow(input_type=Query, output_type=CombinedReport)
    flow.add_nodes(research_node, analysis_node, merge_node)

    # Create input
    query = Query(topic="AI workflow orchestration")

    print(f"Query: {query.topic}")
    print()
    print("Processing...")
    print()

    # In a real scenario with API keys:
    # result = await extract_result_from_stream(flow.astream(query)
    # print("Summary:", result.summary)

    # For demonstration without API keys:
    print("✓ Research node would gather facts and sources")
    print("✓ Analysis node would extract key points and conclusions")
    print("✓ AgentNode would combine both into a prompt")
    print("✓ LLM would generate a comprehensive summary")
    print()
    print("Example prompt format:")
    print("-" * 60)

    # Demonstrate prompt formatting
    research_data = gather_research(query)
    analysis_data = perform_analysis(query)

    formatted = merge_node._format_prompt((research_data, analysis_data))
    print(formatted)
    print("-" * 60)


async def main_simple():
    """Demonstrate simple positional references."""
    print()
    print("=" * 60)
    print("Simple Example: Positional References")
    print("=" * 60)
    print()

    # Create simple nodes
    research_node = ToolNode[Query, ResearchData](
        tool_func=gather_research,
        name="research",
    )

    analysis_node = ToolNode[Query, AnalysisData](
        tool_func=perform_analysis,
        name="analysis",
    )

    # Use simple positional references {0} and {1}
    merge_node = AgentNode.from_prompt(
        model="test",
        prompt_template="Summarize this research: {0}\n\nAnd this analysis: {1}",
        inputs=(research_node.output, analysis_node.output),
        name="simple_merge",
    )

    # Demonstrate
    query = Query(topic="Python frameworks")
    research = gather_research(query)
    analysis = perform_analysis(query)

    formatted = merge_node._format_prompt((research, analysis))
    print("Formatted prompt:")
    print("-" * 60)
    print(formatted)
    print("-" * 60)


if __name__ == "__main__":
    print("Pydantic-Flow AgentNode Multi-Input Examples")
    print()
    asyncio.run(main())
    asyncio.run(main_simple())
    print()
    print("Note: To use with real LLMs, set appropriate API keys and use")
    print("      model names like 'openai:gpt-4' or 'anthropic:claude-3'")
