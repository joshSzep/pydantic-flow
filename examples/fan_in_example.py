"""Example demonstrating fan-in patterns with MergeToolNode.

This example shows how to use MergeToolNode and MergeParserNode to combine
outputs from multiple nodes into a single processing step.
"""

import asyncio

from pydantic import BaseModel

from pydantic_flow import Flow
from pydantic_flow import MergeToolNode
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
    """Research results."""

    facts: list[str]
    sources: int


class AnalysisData(BaseModel):
    """Analysis results."""

    insights: list[str]
    confidence: float


class MetadataData(BaseModel):
    """Metadata."""

    timestamp: str
    version: str


class CombinedReport(BaseModel):
    """Combined report from multiple sources."""

    summary: str
    total_facts: int
    total_insights: int
    confidence: float


class FinalOutput(BaseModel):
    """Final output containing all results."""

    research: ResearchData
    analysis: AnalysisData
    metadata: MetadataData
    combined: CombinedReport


def gather_research(query: Query) -> ResearchData:
    """Simulate research gathering."""
    return ResearchData(
        facts=[f"Fact about {query.topic}", f"Another fact about {query.topic}"],
        sources=5,
    )


def perform_analysis(query: Query) -> AnalysisData:
    """Simulate analysis."""
    return AnalysisData(
        insights=[f"Insight on {query.topic}", f"Deep insight on {query.topic}"],
        confidence=0.85,
    )


def collect_metadata(query: Query) -> MetadataData:
    """Simulate metadata collection."""
    return MetadataData(timestamp="2025-10-23T12:00:00Z", version="1.0.0")


def combine_all(
    research: ResearchData, analysis: AnalysisData, metadata: MetadataData
) -> CombinedReport:
    """Combine all data sources into a unified report."""
    summary = f"Report on topic (v{metadata.version})"
    return CombinedReport(
        summary=summary,
        total_facts=len(research.facts),
        total_insights=len(analysis.insights),
        confidence=analysis.confidence,
    )


async def main():
    """Run the fan-in example."""
    print("=" * 60)
    print("Fan-In Pattern Example with MergeToolNode")
    print("=" * 60)

    flow = Flow(input_type=Query, output_type=FinalOutput)

    research_node = ToolNode[Query, ResearchData](
        tool_func=gather_research, name="research"
    )

    analysis_node = ToolNode[Query, AnalysisData](
        tool_func=perform_analysis, name="analysis"
    )

    metadata_node = ToolNode[Query, MetadataData](
        tool_func=collect_metadata, name="metadata"
    )

    merge_node = MergeToolNode[
        ResearchData, AnalysisData, MetadataData, CombinedReport
    ](
        inputs=(research_node.output, analysis_node.output, metadata_node.output),
        tool_func=combine_all,
        name="combined",
    )

    flow.add_nodes(research_node, analysis_node, metadata_node, merge_node)
    flow.compile()

    print("\nRunning flow...")
    result = await extract_result_from_stream(flow.astream(Query(topic="AI Workflows")))

    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"\nResearch ({result.research.sources} sources):")
    for fact in result.research.facts:
        print(f"  - {fact}")

    print(f"\nAnalysis (confidence: {result.analysis.confidence}):")
    for insight in result.analysis.insights:
        print(f"  - {insight}")

    print("\nMetadata:")
    print(f"  - Timestamp: {result.metadata.timestamp}")
    print(f"  - Version: {result.metadata.version}")

    print("\nCombined Report:")
    print(f"  - Summary: {result.combined.summary}")
    print(f"  - Total Facts: {result.combined.total_facts}")
    print(f"  - Total Insights: {result.combined.total_insights}")
    print(f"  - Confidence: {result.combined.confidence}")

    print("\n" + "=" * 60)
    print("Pattern: Query → [Research, Analysis, Metadata] → Merge → Output")
    print("         (fan-out)                              (fan-in)")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
