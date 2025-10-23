"""Tests for merge nodes (multi-input fan-in patterns)."""

from pydantic import BaseModel
import pytest

from pydantic_flow import Flow
from pydantic_flow import MergeParserNode
from pydantic_flow import MergePromptNode
from pydantic_flow import MergeToolNode
from pydantic_flow import ToolNode


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


def get_data_b(input_data: Input) -> DataB:
    """Tool function to generate DataB."""
    return DataB(value_b=len(input_data.query))


def get_data_c(input_data: Input) -> DataC:
    """Tool function to generate DataC."""
    return DataC(value_c=float(len(input_data.query)) * 2.5)


def merge_two(data_a: DataA, data_b: DataB) -> MergedResult:
    """Merge two inputs."""
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


class TestBasicMerge:
    """Test basic merge node functionality."""

    @pytest.mark.asyncio
    async def test_merge_two_nodes(self):
        """Test merging outputs from two nodes."""
        flow = Flow(input_type=Input, output_type=FinalResult)

        node_a = ToolNode[Input, DataA](tool_func=get_data_a, name="node_a")
        node_b = ToolNode[Input, DataB](tool_func=get_data_b, name="node_b")

        merge_node = MergeToolNode[DataA, DataB, MergedResult](
            inputs=(node_a.output, node_b.output),
            tool_func=merge_two,
            name="merge_node",
        )

        final_node = ToolNode[MergedResult, MergedResult](
            tool_func=lambda x: x,
            input=merge_node.output,
            name="final_node",
        )

        # Placeholder for remaining results
        node_c = ToolNode[Input, DataC](tool_func=get_data_c, name="node_c")

        flow.add_nodes(node_a, node_b, node_c, merge_node, final_node)

        result = await flow.run(Input(query="test"))

        assert result.node_a.value_a == "A:test"
        assert result.node_b.value_b == 4
        assert result.merge_node.combined == "A:test+4"
        assert result.merge_node.sum_value == 4.0

    @pytest.mark.asyncio
    async def test_merge_three_nodes(self):
        """Test merging outputs from three nodes."""
        flow = Flow(input_type=Input, output_type=FinalResult)

        node_a = ToolNode[Input, DataA](tool_func=get_data_a, name="node_a")
        node_b = ToolNode[Input, DataB](tool_func=get_data_b, name="node_b")
        node_c = ToolNode[Input, DataC](tool_func=get_data_c, name="node_c")

        merge_node = MergeToolNode[DataA, DataB, DataC, MergedResult](
            inputs=(node_a.output, node_b.output, node_c.output),
            tool_func=merge_three,
            name="merge_node",
        )

        final_node = ToolNode[MergedResult, MergedResult](
            tool_func=lambda x: x,
            input=merge_node.output,
            name="final_node",
        )

        flow.add_nodes(node_a, node_b, node_c, merge_node, final_node)

        result = await flow.run(Input(query="hello"))

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

        def process_a(inp: Input) -> ProcessedA:
            return ProcessedA(data=f"A({inp.query})")

        def process_b(data: ProcessedA) -> ProcessedB:
            return ProcessedB(data_b=f"B({data.data})")

        def process_c(data: ProcessedA) -> ProcessedC:
            return ProcessedC(data_c=f"C({data.data})")

        def process_d(data_b: ProcessedB) -> ProcessedD:
            return ProcessedD(merged=f"D({data_b.data_b})")

        def process_e(data_d: ProcessedD, data_c: ProcessedC) -> ProcessedE:
            return ProcessedE(final=f"E({data_d.merged},{data_c.data_c})")

        flow = Flow(input_type=Input, output_type=FlowResult)

        # A receives initial input
        node_a = ToolNode[Input, ProcessedA](tool_func=process_a, name="a")

        # A fans out to B and C
        node_b = ToolNode[ProcessedA, ProcessedB](
            tool_func=process_b, input=node_a.output, name="b"
        )
        node_c = ToolNode[ProcessedA, ProcessedC](
            tool_func=process_c, input=node_a.output, name="c"
        )

        # B goes to D
        node_d = ToolNode[ProcessedB, ProcessedD](
            tool_func=process_d, input=node_b.output, name="d"
        )

        # D and C fan in to E
        node_e = MergeToolNode[ProcessedD, ProcessedC, ProcessedE](
            inputs=(node_d.output, node_c.output), tool_func=process_e, name="e"
        )

        flow.add_nodes(node_a, node_b, node_c, node_d, node_e)

        # Verify execution order
        exec_order = flow.get_execution_order()
        assert exec_order.index("a") < exec_order.index("b")
        assert exec_order.index("a") < exec_order.index("c")
        assert exec_order.index("b") < exec_order.index("d")
        assert exec_order.index("d") < exec_order.index("e")
        assert exec_order.index("c") < exec_order.index("e")

        # Execute and verify
        result = await flow.run(Input(query="test"))

        assert result.a.data == "A(test)"
        assert result.b.data_b == "B(A(test))"
        assert result.c.data_c == "C(A(test))"
        assert result.d.merged == "D(B(A(test)))"
        assert result.e.final == "E(D(B(A(test))),C(A(test)))"


class TestMergeParserNode:
    """Test MergeParserNode functionality."""

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

        def get_text_a(inp: Input) -> TextA:
            return TextA(text=f"Hello {inp.query}")

        def get_text_b(inp: Input) -> TextB:
            return TextB(text=f"World {inp.query}")

        def parse_combined(text_a: TextA, text_b: TextB) -> Parsed:
            combined = f"{text_a.text} and {text_b.text}"
            return Parsed(combined=combined, length=len(combined))

        flow = Flow(input_type=Input, output_type=FlowResult)

        node_a = ToolNode[Input, TextA](tool_func=get_text_a, name="text_a")
        node_b = ToolNode[Input, TextB](tool_func=get_text_b, name="text_b")

        parser = MergeParserNode[TextA, TextB, Parsed](
            inputs=(node_a.output, node_b.output),
            parser_func=parse_combined,
            name="parsed",
        )

        flow.add_nodes(node_a, node_b, parser)

        result = await flow.run(Input(query="test"))

        assert result.text_a.text == "Hello test"
        assert result.text_b.text == "World test"
        assert result.parsed.combined == "Hello test and World test"
        assert result.parsed.length == 25  # "Hello test and World test" is 25 chars


class TestDependencyTracking:
    """Test that dependencies are correctly tracked for merge nodes."""

    def test_merge_node_dependencies(self):
        """Test that merge nodes report all input dependencies."""
        node_a = ToolNode[Input, DataA](tool_func=get_data_a, name="node_a")
        node_b = ToolNode[Input, DataB](tool_func=get_data_b, name="node_b")
        node_c = ToolNode[Input, DataC](tool_func=get_data_c, name="node_c")

        merge_node = MergeToolNode[DataA, DataB, DataC, MergedResult](
            inputs=(node_a.output, node_b.output, node_c.output),
            tool_func=merge_three,
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

        node_a = ToolNode[Input, DataA](tool_func=get_data_a, name="node_a")
        node_b = ToolNode[Input, DataB](tool_func=get_data_b, name="node_b")
        node_c = ToolNode[Input, DataC](tool_func=get_data_c, name="node_c")

        merge_node = MergeToolNode[DataA, DataB, DataC, MergedResult](
            inputs=(node_a.output, node_b.output, node_c.output),
            tool_func=merge_three,
            name="merge_node",
        )

        final_node = ToolNode[MergedResult, MergedResult](
            tool_func=lambda x: x,
            input=merge_node.output,
            name="final_node",
        )

        flow.add_nodes(node_a, node_b, node_c, merge_node, final_node)

        exec_order = flow.get_execution_order()

        merge_idx = exec_order.index("merge_node")
        assert exec_order.index("node_a") < merge_idx
        assert exec_order.index("node_b") < merge_idx
        assert exec_order.index("node_c") < merge_idx
        assert exec_order.index("merge_node") < exec_order.index("final_node")


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_merge_with_single_input(self):
        """Test that merge node works with a single input (degenerate case)."""

        class SingleResult(BaseModel):
            node_a: DataA
            merge: DataA

        def identity(data_a: DataA) -> DataA:
            return data_a

        flow = Flow(input_type=Input, output_type=SingleResult)

        node_a = ToolNode[Input, DataA](tool_func=get_data_a, name="node_a")

        merge_node = MergeToolNode[DataA, DataA](
            inputs=(node_a.output,),
            tool_func=identity,
            name="merge",
        )

        flow.add_nodes(node_a, merge_node)

        result = await flow.run(Input(query="test"))

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

        def merge_ab(a: DataA, b: DataB) -> MergedResult:
            return MergedResult(combined=f"{a.value_a}+{b.value_b}", sum_value=0.0)

        def merge_bc(b: DataB, c: DataC) -> MergedResult:
            return MergedResult(
                combined=f"{b.value_b}+{c.value_c}", sum_value=c.value_c
            )

        flow = Flow(input_type=Input, output_type=MultiMergeResult)

        node_a = ToolNode[Input, DataA](tool_func=get_data_a, name="node_a")
        node_b = ToolNode[Input, DataB](tool_func=get_data_b, name="node_b")
        node_c = ToolNode[Input, DataC](tool_func=get_data_c, name="node_c")

        merge1 = MergeToolNode[DataA, DataB, MergedResult](
            inputs=(node_a.output, node_b.output),
            tool_func=merge_ab,
            name="merge1",
        )

        merge2 = MergeToolNode[DataB, DataC, MergedResult](
            inputs=(node_b.output, node_c.output),
            tool_func=merge_bc,
            name="merge2",
        )

        flow.add_nodes(node_a, node_b, node_c, merge1, merge2)

        result = await flow.run(Input(query="test"))

        assert result.merge1.combined == "A:test+4"
        assert result.merge2.combined == "4+10.0"


class TestMergePromptNode:
    """Test MergePromptNode initialization and structure."""

    def test_merge_prompt_node_initialization(self):
        """Test that MergePromptNode initializes correctly."""
        node_a = ToolNode[Input, DataA](tool_func=get_data_a, name="node_a")
        node_b = ToolNode[Input, DataB](tool_func=get_data_b, name="node_b")

        merge_prompt = MergePromptNode[DataA, DataB, str](
            inputs=(node_a.output, node_b.output),
            prompt="Combine {0} and {1}",
            model="openai:gpt-4",
            name="merge_prompt",
        )

        assert merge_prompt.name == "merge_prompt"
        assert merge_prompt.prompt == "Combine {0} and {1}"
        assert merge_prompt.model == "openai:gpt-4"
        assert len(merge_prompt.inputs) == 2

    def test_merge_prompt_node_dependencies(self):
        """Test that MergePromptNode tracks dependencies correctly."""
        node_a = ToolNode[Input, DataA](tool_func=get_data_a, name="node_a")
        node_b = ToolNode[Input, DataB](tool_func=get_data_b, name="node_b")

        merge_prompt = MergePromptNode[DataA, DataB, str](
            inputs=(node_a.output, node_b.output),
            prompt="Combine {0} and {1}",
            name="merge_prompt",
        )

        deps = merge_prompt.dependencies
        assert len(deps) == 2
        assert node_a in deps
        assert node_b in deps

    @pytest.mark.asyncio
    async def test_merge_prompt_node_not_implemented(self):
        """Test that MergePromptNode run raises NotImplementedError."""
        node_a = ToolNode[Input, DataA](tool_func=get_data_a, name="node_a")
        node_b = ToolNode[Input, DataB](tool_func=get_data_b, name="node_b")

        merge_prompt = MergePromptNode[DataA, DataB, str](
            inputs=(node_a.output, node_b.output),
            prompt="Combine {0} and {1}",
            name="merge_prompt",
        )

        data_a = DataA(value_a="test_a")
        data_b = DataB(value_b=42)

        with pytest.raises(NotImplementedError) as exc_info:
            await merge_prompt.run((data_a, data_b))

        assert "LLM integration not yet implemented" in str(exc_info.value)
