"""Tests for merge nodes (multi-input fan-in patterns)."""

from pydantic import BaseModel
import pytest

from pydantic_flow import Flow
from pydantic_flow import ParserNode
from pydantic_flow import ToolNode
from tests.conftest import extract_result_from_stream


class Input(BaseModel):
    """Input model for testing."""

    query: str


class DataA(BaseModel):
    """First data type for merging."""

    value_a: str


class DataB(BaseModel):
    """Second data type for merging."""

    value_b: int


class DataC(BaseModel):
    """Third data type for merging."""

    value_c: float


class MergedResult(BaseModel):
    """Result of merging multiple inputs."""

    combined: str
    sum_value: float


class FinalResult(BaseModel):
    """Final output containing all results."""

    node_a: DataA
    node_b: DataB
    node_c: DataC
    merge_node: MergedResult
    final_node: MergedResult


def get_data_a(input_data: Input) -> DataA:
    """Tool function to generate DataA."""
    return DataA(value_a=f"A:{input_data.query}")


async def get_data_a_async(input_data: Input) -> DataA:
    """Async tool function to generate DataA."""
    return DataA(value_a=f"A:{input_data.query}")


def get_data_b(input_data: Input) -> DataB:
    """Tool function to generate DataB."""
    return DataB(value_b=len(input_data.query))


async def get_data_b_async(input_data: Input) -> DataB:
    """Async tool function to generate DataB."""
    return DataB(value_b=len(input_data.query))


def get_data_c(input_data: Input) -> DataC:
    """Tool function to generate DataC."""
    return DataC(value_c=float(len(input_data.query)) * 2.5)


async def get_data_c_async(input_data: Input) -> DataC:
    """Async tool function to generate DataC."""
    return DataC(value_c=float(len(input_data.query)) * 2.5)


def merge_two(data_a: DataA, data_b: DataB) -> MergedResult:
    """Merge two inputs."""
    return MergedResult(
        combined=f"{data_a.value_a}+{data_b.value_b}",
        sum_value=float(data_b.value_b),
    )


async def merge_two_async(data_a: DataA, data_b: DataB) -> MergedResult:
    """Async merge two inputs."""
    return MergedResult(
        combined=f"{data_a.value_a}+{data_b.value_b}",
        sum_value=float(data_b.value_b),
    )


def merge_three(data_a: DataA, data_b: DataB, data_c: DataC) -> MergedResult:
    """Merge three inputs."""
    return MergedResult(
        combined=f"{data_a.value_a}+{data_b.value_b}+{data_c.value_c}",
        sum_value=float(data_b.value_b) + data_c.value_c,
    )


async def merge_three_async(
    data_a: DataA, data_b: DataB, data_c: DataC
) -> MergedResult:
    """Async merge three inputs."""
    return MergedResult(
        combined=f"{data_a.value_a}+{data_b.value_b}+{data_c.value_c}",
        sum_value=float(data_b.value_b) + data_c.value_c,
    )


class TestBasicMerge:
    """Test basic merge node functionality."""

    @pytest.mark.asyncio
    async def test_merge_two_nodes(self):
        """Test merging outputs from two nodes."""
        flow = Flow(input_type=Input, output_type=FinalResult)

        node_a = ToolNode[Input, DataA](tool_func=get_data_a_async, name="node_a")
        node_b = ToolNode[Input, DataB](tool_func=get_data_b_async, name="node_b")

        merge_node = ToolNode[tuple[DataA, DataB], MergedResult](  # type: ignore[type-var]
            inputs=(node_a.output, node_b.output),
            tool_func=merge_two_async,
            name="merge_node",
        )

        async def identity_func(x: MergedResult) -> MergedResult:
            return x

        final_node = ToolNode[MergedResult, MergedResult](
            tool_func=identity_func,
            inputs=(merge_node.output,),
            name="final_node",
        )

        # Placeholder for remaining results
        node_c = ToolNode[Input, DataC](tool_func=get_data_c_async, name="node_c")

        flow.add_nodes(node_a, node_b, node_c, merge_node, final_node)

        result = await extract_result_from_stream(flow.astream(Input(query="test")))

        assert result.node_a.value_a == "A:test"
        assert result.node_b.value_b == 4
        assert result.merge_node.combined == "A:test+4"
        assert result.merge_node.sum_value == 4.0

    @pytest.mark.asyncio
    async def test_merge_three_nodes(self):
        """Test merging outputs from three nodes."""
        flow = Flow(input_type=Input, output_type=FinalResult)

        node_a = ToolNode[Input, DataA](tool_func=get_data_a_async, name="node_a")
        node_b = ToolNode[Input, DataB](tool_func=get_data_b_async, name="node_b")
        node_c = ToolNode[Input, DataC](tool_func=get_data_c_async, name="node_c")

        merge_node = ToolNode[tuple[DataA, DataB, DataC], MergedResult](  # type: ignore[type-var]
            inputs=(node_a.output, node_b.output, node_c.output),
            tool_func=merge_three_async,
            name="merge_node",
        )

        async def identity_func_three(x: MergedResult) -> MergedResult:
            return x

        final_node = ToolNode[MergedResult, MergedResult](
            tool_func=identity_func_three,
            inputs=(merge_node.output,),
            name="final_node",
        )

        flow.add_nodes(node_a, node_b, node_c, merge_node, final_node)

        result = await extract_result_from_stream(flow.astream(Input(query="hello")))

        assert result.node_a.value_a == "A:hello"
        assert result.node_b.value_b == 5
        assert result.node_c.value_c == 12.5
        assert result.merge_node.combined == "A:hello+5+12.5"
        assert result.merge_node.sum_value == 17.5


class TestFanOutFanIn:
    """Test complex fan-out and fan-in patterns."""

    @pytest.mark.asyncio
    async def test_fan_out_fan_in_pattern(self):
        """Test the exact pattern from the user's example: A->B,C then B,C->D."""

        class ProcessedA(BaseModel):
            data: str

        class ProcessedB(BaseModel):
            data_b: str

        class ProcessedC(BaseModel):
            data_c: str

        class ProcessedD(BaseModel):
            merged: str

        class ProcessedE(BaseModel):
            final: str

        class FlowResult(BaseModel):
            a: ProcessedA
            b: ProcessedB
            c: ProcessedC
            d: ProcessedD
            e: ProcessedE

        async def process_a(inp: Input) -> ProcessedA:
            return ProcessedA(data=f"A({inp.query})")

        async def process_b(data: ProcessedA) -> ProcessedB:
            return ProcessedB(data_b=f"B({data.data})")

        async def process_c(data: ProcessedA) -> ProcessedC:
            return ProcessedC(data_c=f"C({data.data})")

        async def process_d(data_b: ProcessedB) -> ProcessedD:
            return ProcessedD(merged=f"D({data_b.data_b})")

        async def process_e(data_d: ProcessedD, data_c: ProcessedC) -> ProcessedE:
            return ProcessedE(final=f"E({data_d.merged},{data_c.data_c})")

        flow = Flow(input_type=Input, output_type=FlowResult)

        # A receives initial input
        node_a = ToolNode[Input, ProcessedA](tool_func=process_a, name="a")

        # A fans out to B and C
        node_b = ToolNode[ProcessedA, ProcessedB](
            tool_func=process_b, inputs=(node_a.output,), name="b"
        )
        node_c = ToolNode[ProcessedA, ProcessedC](
            tool_func=process_c, inputs=(node_a.output,), name="c"
        )

        # B goes to D
        node_d = ToolNode[ProcessedB, ProcessedD](
            tool_func=process_d, inputs=(node_b.output,), name="d"
        )

        # D and C fan in to E
        node_e = ToolNode[tuple[ProcessedD, ProcessedC], ProcessedE](  # type: ignore[type-var]
            inputs=(node_d.output, node_c.output), tool_func=process_e, name="e"
        )

        flow.add_nodes(node_a, node_b, node_c, node_d, node_e)

        # Verify flow is valid
        assert flow is not None
        assert len(flow.nodes) == 5

        # Execute and verify
        result = await extract_result_from_stream(flow.astream(Input(query="test")))

        assert result.a.data == "A(test)"
        assert result.b.data_b == "B(A(test))"
        assert result.c.data_c == "C(A(test))"
        assert result.d.merged == "D(B(A(test)))"
        assert result.e.final == "E(D(B(A(test))),C(A(test)))"


class TestParserNode:
    """Test ParserNode functionality."""

    @pytest.mark.asyncio
    async def test_merge_parser_node(self):
        """Test merging with parser functions."""

        class TextA(BaseModel):
            text: str

        class TextB(BaseModel):
            text: str

        class Parsed(BaseModel):
            combined: str
            length: int

        class FlowResult(BaseModel):
            text_a: TextA
            text_b: TextB
            parsed: Parsed

        async def get_text_a(inp: Input) -> TextA:
            return TextA(text=f"Hello {inp.query}")

        async def get_text_b(inp: Input) -> TextB:
            return TextB(text=f"World {inp.query}")

        def parse_combined(text_a: TextA, text_b: TextB) -> Parsed:
            combined = f"{text_a.text} and {text_b.text}"
            return Parsed(combined=combined, length=len(combined))

        flow = Flow(input_type=Input, output_type=FlowResult)

        node_a = ToolNode[Input, TextA](tool_func=get_text_a, name="text_a")
        node_b = ToolNode[Input, TextB](tool_func=get_text_b, name="text_b")

        parser = ParserNode[tuple[TextA, TextB], Parsed](
            inputs=(node_a.output, node_b.output),
            parser_func=parse_combined,
            name="parsed",
        )

        flow.add_nodes(node_a, node_b, parser)

        result = await extract_result_from_stream(flow.astream(Input(query="test")))

        assert result.text_a.text == "Hello test"
        assert result.text_b.text == "World test"
        assert result.parsed.combined == "Hello test and World test"
        assert result.parsed.length == 25  # "Hello test and World test" is 25 chars


class TestDependencyTracking:
    """Test that dependencies are correctly tracked for merge nodes."""

    def test_merge_node_dependencies(self):
        """Test that merge nodes report all input dependencies."""
        node_a = ToolNode[Input, DataA](tool_func=get_data_a_async, name="node_a")
        node_b = ToolNode[Input, DataB](tool_func=get_data_b_async, name="node_b")
        node_c = ToolNode[Input, DataC](tool_func=get_data_c_async, name="node_c")

        merge_node = ToolNode[tuple[DataA, DataB, DataC], MergedResult](  # type: ignore[type-var]
            inputs=(node_a.output, node_b.output, node_c.output),
            tool_func=merge_three_async,
            name="merge",
        )

        deps = merge_node.dependencies

        assert len(deps) == 3
        assert node_a in deps
        assert node_b in deps
        assert node_c in deps

    def test_execution_order_with_merge(self):
        """Test that execution order is correct with merge nodes."""
        flow = Flow(input_type=Input, output_type=FinalResult)

        node_a = ToolNode[Input, DataA](tool_func=get_data_a_async, name="node_a")
        node_b = ToolNode[Input, DataB](tool_func=get_data_b_async, name="node_b")
        node_c = ToolNode[Input, DataC](tool_func=get_data_c_async, name="node_c")

        merge_node = ToolNode[tuple[DataA, DataB, DataC], MergedResult](  # type: ignore[type-var]
            inputs=(node_a.output, node_b.output, node_c.output),
            tool_func=merge_three_async,
            name="merge_node",
        )

        async def identity_func3(x: MergedResult) -> MergedResult:
            return x

        final_node = ToolNode[MergedResult, MergedResult](
            tool_func=identity_func3,
            inputs=(merge_node.output,),
            name="final_node",
        )

        flow.add_nodes(node_a, node_b, node_c, merge_node, final_node)

        # Verify flow is valid with merge dependencies
        assert flow is not None
        assert len(flow.nodes) == 5
        # Verify merge node depends on all three inputs
        assert len(merge_node.dependencies) == 3


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_merge_with_single_input(self):
        """Test that merge node works with a single input (degenerate case)."""

        class SingleResult(BaseModel):
            node_a: DataA
            merge: DataA

        async def identity(data_a: DataA) -> DataA:
            return data_a

        flow = Flow(input_type=Input, output_type=SingleResult)

        node_a = ToolNode[Input, DataA](tool_func=get_data_a_async, name="node_a")

        merge_node = ToolNode[DataA, DataA](
            inputs=(node_a.output,),
            tool_func=identity,
            name="merge",
        )

        flow.add_nodes(node_a, merge_node)

        result = await extract_result_from_stream(flow.astream(Input(query="test")))

        assert result.node_a.value_a == "A:test"
        assert result.merge.value_a == "A:test"

    @pytest.mark.asyncio
    async def test_multiple_merge_nodes(self):
        """Test multiple merge nodes in the same flow."""

        class MultiMergeResult(BaseModel):
            node_a: DataA
            node_b: DataB
            node_c: DataC
            merge1: MergedResult
            merge2: MergedResult

        async def merge_ab(a: DataA, b: DataB) -> MergedResult:
            return MergedResult(combined=f"{a.value_a}+{b.value_b}", sum_value=0.0)

        async def merge_bc(b: DataB, c: DataC) -> MergedResult:
            return MergedResult(
                combined=f"{b.value_b}+{c.value_c}", sum_value=c.value_c
            )

        flow = Flow(input_type=Input, output_type=MultiMergeResult)

        node_a = ToolNode[Input, DataA](tool_func=get_data_a_async, name="node_a")
        node_b = ToolNode[Input, DataB](tool_func=get_data_b_async, name="node_b")
        node_c = ToolNode[Input, DataC](tool_func=get_data_c_async, name="node_c")

        merge1 = ToolNode[tuple[DataA, DataB], MergedResult](  # type: ignore[type-var]
            inputs=(node_a.output, node_b.output),
            tool_func=merge_ab,
            name="merge1",
        )

        merge2 = ToolNode[tuple[DataB, DataC], MergedResult](  # type: ignore[type-var]
            inputs=(node_b.output, node_c.output),
            tool_func=merge_bc,
            name="merge2",
        )

        flow.add_nodes(node_a, node_b, node_c, merge1, merge2)

        result = await extract_result_from_stream(flow.astream(Input(query="test")))

        assert result.merge1.combined == "A:test+4"
        assert result.merge2.combined == "4+10.0"
