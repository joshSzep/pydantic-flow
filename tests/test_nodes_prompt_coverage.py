"""Additional tests for nodes/prompt.py to achieve full coverage."""

from pydantic import BaseModel
import pytest

from pydantic_flow import ChatMessage
from pydantic_flow import ChatPromptTemplate
from pydantic_flow import ChatRole
from pydantic_flow import InterruptDecision
from pydantic_flow import InterruptionRequested
from pydantic_flow import JoinStrategy
from pydantic_flow import NonFatalError
from pydantic_flow import ProgressItem
from pydantic_flow import PromptConfig
from pydantic_flow import PromptNode
from pydantic_flow import PromptTemplate
from pydantic_flow import StreamEnd
from pydantic_flow import StreamStart
from pydantic_flow import TemplateFormat
from pydantic_flow import TokenChunk
from pydantic_flow import ToolResult


class SimpleInput(BaseModel):
    """Simple input model for testing."""

    name: str
    age: int


class StructuredOutput(BaseModel):
    """Structured output model."""

    greeting: str
    status: str


class FailingParser:
    """Output parser that always fails."""

    async def parse(self, text: str) -> str:
        """Parse that raises an exception."""
        raise ValueError("Parser failed")


@pytest.mark.asyncio
class TestPromptNodeCoverage:
    """Tests for uncovered paths in PromptNode."""

    async def test_with_result_type(self) -> None:
        """Test PromptNode with result_type configured."""
        config = PromptConfig(
            model="test",
            result_type=StructuredOutput,
        )
        node = PromptNode[SimpleInput, StructuredOutput](
            prompt="Generate a greeting for {name}",
            config=config,
        )

        assert node.config.result_type == StructuredOutput
        assert node._agent is not None

    async def test_create_checkpoint(self) -> None:
        """Test checkpoint creation."""
        node = PromptNode[SimpleInput, str](
            prompt="Hello {name}",
            name="test_node",
        )

        checkpoint = node._create_checkpoint("test-run-123")

        assert checkpoint.run_id == "test-run-123"
        assert checkpoint.interrupted_node_id == "test_node"
        assert checkpoint.wave_number == 0
        assert checkpoint.reason == "hitl_interrupt"

    async def test_chat_template_rendering(self) -> None:
        """Test chat template message rendering path."""
        messages = [
            ChatMessage(role=ChatRole.SYSTEM, content="You are helpful"),
            ChatMessage(role=ChatRole.USER, content="Hello {name}"),
        ]
        template = ChatPromptTemplate[SimpleInput, str](
            messages=messages,
            input_model=SimpleInput,
            format=TemplateFormat.F_STRING,
        )
        config = PromptConfig(
            model="test",
            chat_join_strategy=JoinStrategy.ANTHROPIC,
        )
        node = PromptNode[SimpleInput, str](
            prompt=template,
            config=config,
        )

        input_data = SimpleInput(name="Alice", age=30)
        result = None
        async for item in node.astream(input_data):
            if isinstance(item, ToolResult):
                result = item.result

        assert result is not None

    async def test_interrupt_on_stream_start(self) -> None:
        """Test interruption at StreamStart event."""
        node = PromptNode[SimpleInput, str](
            prompt="Hello {name}",
            name="test_node",
        )

        async def interrupt_handler(item: ProgressItem) -> InterruptDecision:
            if isinstance(item, StreamStart):
                return InterruptDecision(should_interrupt=True, reason="Testing")
            return InterruptDecision(should_interrupt=False)

        node.register_interrupt_handler(interrupt_handler)

        input_data = SimpleInput(name="Bob", age=25)

        with pytest.raises(InterruptionRequested) as exc_info:
            async for _ in node.astream(input_data):
                pass

        assert isinstance(exc_info.value, InterruptionRequested)
        assert exc_info.value.decision.reason == "Testing"
        assert exc_info.value.snapshot.interrupted_node_id == "test_node"

    async def test_interrupt_on_token_chunk(self) -> None:
        """Test interruption during token streaming."""
        node = PromptNode[SimpleInput, str](
            prompt="Hello {name}",
            name="test_node",
        )

        token_count = 0

        async def interrupt_handler(item: ProgressItem) -> InterruptDecision:
            nonlocal token_count
            if isinstance(item, TokenChunk):
                token_count += 1
                if token_count >= 1:
                    return InterruptDecision(
                        should_interrupt=True, reason="Token limit"
                    )
            return InterruptDecision(should_interrupt=False)

        node.register_interrupt_handler(interrupt_handler)

        input_data = SimpleInput(name="Charlie", age=35)

        with pytest.raises(InterruptionRequested) as exc_info:
            async for _ in node.astream(input_data):
                pass

        assert isinstance(exc_info.value, InterruptionRequested)
        assert exc_info.value.decision.reason == "Token limit"

    async def test_interrupt_on_tool_result(self) -> None:
        """Test interruption at ToolResult event."""
        node = PromptNode[SimpleInput, str](
            prompt="Hello {name}",
            name="test_node",
        )

        async def interrupt_handler(item: ProgressItem) -> InterruptDecision:
            if isinstance(item, ToolResult):
                return InterruptDecision(should_interrupt=True, reason="Result check")
            return InterruptDecision(should_interrupt=False)

        node.register_interrupt_handler(interrupt_handler)

        input_data = SimpleInput(name="Dana", age=28)

        with pytest.raises(InterruptionRequested) as exc_info:
            async for _ in node.astream(input_data):
                pass

        assert isinstance(exc_info.value, InterruptionRequested)
        assert exc_info.value.decision.reason == "Result check"

    async def test_output_parser_applied(self) -> None:
        """Test that output parser is applied to results."""

        class UpperParser:
            async def parse(self, text: str) -> str:
                return text.upper()

        node = PromptNode[SimpleInput, str](
            prompt="Hello {name}",
            output_parser=UpperParser(),
        )

        input_data = SimpleInput(name="Eve", age=40)
        result = None
        async for item in node.astream(input_data):
            if isinstance(item, ToolResult):
                result = item.result

        assert result is not None
        assert result == result.upper()

    async def test_output_parser_failure(self) -> None:
        """Test error handling when output parser fails."""
        node = PromptNode[SimpleInput, str](
            prompt="Hello {name}",
            output_parser=FailingParser(),
        )

        input_data = SimpleInput(name="Frank", age=45)
        got_error = False

        try:
            async for item in node.astream(input_data):
                if isinstance(item, NonFatalError):
                    got_error = True
                    assert "LLM execution failed" in item.message
        except ValueError:
            got_error = True

        assert got_error

    async def test_result_with_model_dump(self) -> None:
        """Test result preview generation with Pydantic model."""
        config = PromptConfig(
            model="test",
        )

        class ModelDumpParser:
            async def parse(self, text: str) -> StructuredOutput:
                return StructuredOutput(greeting=text, status="parsed")

        node = PromptNode[SimpleInput, StructuredOutput](
            prompt="Generate for {name}",
            config=config,
            output_parser=ModelDumpParser(),
        )

        input_data = SimpleInput(name="Grace", age=50)
        stream_end = None
        async for item in node.astream(input_data):
            if isinstance(item, StreamEnd):
                stream_end = item

        assert stream_end is not None
        assert stream_end.result_preview is not None
        assert isinstance(stream_end.result_preview, dict)

    async def test_result_without_model_dump(self) -> None:
        """Test result preview generation with simple string result."""
        node = PromptNode[SimpleInput, str](
            prompt="Hello {name}",
        )

        input_data = SimpleInput(name="Henry", age=55)
        stream_end = None
        async for item in node.astream(input_data):
            if isinstance(item, StreamEnd):
                stream_end = item

        assert stream_end is not None
        assert stream_end.result_preview is not None
        assert "value" in stream_end.result_preview

    async def test_exception_generates_error_event(self) -> None:
        """Test that exceptions generate NonFatalError events."""

        class ErrorParser:
            async def parse(self, text: str) -> str:
                raise RuntimeError("Critical error")

        node = PromptNode[SimpleInput, str](
            prompt="Hello {name}",
            output_parser=ErrorParser(),
        )

        input_data = SimpleInput(name="Iris", age=60)
        got_error_event = False
        exception_raised = False

        try:
            async for item in node.astream(input_data):
                if isinstance(item, NonFatalError):
                    got_error_event = True
                    assert "LLM execution failed" in item.message
                    assert not item.recoverable
        except RuntimeError:
            exception_raised = True

        assert got_error_event
        assert exception_raised

    async def test_interruption_reraise(self) -> None:
        """Test that InterruptionRequested exceptions are re-raised."""
        node = PromptNode[SimpleInput, str](
            prompt="Hello {name}",
            name="test_node",
        )

        async def interrupt_immediately(item: ProgressItem) -> InterruptDecision:
            if isinstance(item, StreamStart):
                return InterruptDecision(should_interrupt=True, reason="Immediate")
            return InterruptDecision(should_interrupt=False)

        node.register_interrupt_handler(interrupt_immediately)

        input_data = SimpleInput(name="Jack", age=65)

        with pytest.raises(InterruptionRequested):
            async for _ in node.astream(input_data):
                pass

    async def test_simple_prompt_template_rendering(self) -> None:
        """Test simple PromptTemplate (non-chat) rendering path."""
        template = PromptTemplate[SimpleInput, str](
            template="Hello {name}, you are {age} years old",
            input_model=SimpleInput,
            format=TemplateFormat.F_STRING,
        )
        node = PromptNode[SimpleInput, str](
            prompt=template,
        )

        input_data = SimpleInput(name="Karen", age=33)
        result = None
        async for item in node.astream(input_data):
            if isinstance(item, ToolResult):
                result = item.result

        assert result is not None

    async def test_interrupt_on_stream_end(self) -> None:
        """Test interruption at StreamEnd event."""
        node = PromptNode[SimpleInput, str](
            prompt="Hello {name}",
            name="test_node",
        )

        async def interrupt_on_end(item: ProgressItem) -> InterruptDecision:
            if isinstance(item, StreamEnd):
                return InterruptDecision(should_interrupt=True, reason="End check")
            return InterruptDecision(should_interrupt=False)

        node.register_interrupt_handler(interrupt_on_end)

        input_data = SimpleInput(name="Laura", age=42)

        with pytest.raises(InterruptionRequested) as exc_info:
            async for _ in node.astream(input_data):
                pass

        assert isinstance(exc_info.value, InterruptionRequested)
        assert exc_info.value.decision.reason == "End check"
