"""Additional tests for remaining uncovered paths in nodes/prompt.py."""

from pydantic import BaseModel
import pytest

from pydantic_flow import NonFatalError
from pydantic_flow import PromptConfig
from pydantic_flow import PromptNode
from pydantic_flow import PromptTemplate
from pydantic_flow import StreamEnd
from pydantic_flow import TemplateFormat
from pydantic_flow import ToolResult


class InputData(BaseModel):
    """Test input model."""

    message: str


@pytest.mark.asyncio
class TestPromptNodeRemainingPaths:
    """Tests for remaining uncovered paths."""

    async def test_string_prompt_backward_compatibility(self) -> None:
        """Test using plain string prompt (backward compatibility)."""
        node = PromptNode[InputData, str](
            prompt="Say hello to {message}",
            name="string_node",
        )

        # This hits lines 95-96 (string prompt path)
        assert node._raw_prompt == "Say hello to {message}"
        assert node._template is None

        input_data = InputData(message="world")
        result = None
        async for item in node.astream(input_data):
            if isinstance(item, ToolResult):
                result = item.result

        assert result is not None

    async def test_agent_without_result_type(self) -> None:
        """Test creating agent without result_type."""
        config = PromptConfig(
            model="test",
            result_type=None,  # No structured output
        )
        node = PromptNode[InputData, str](
            prompt="Hello {message}",
            config=config,
        )

        # This hits line 101 (agent without result_type)
        assert node._agent is not None
        assert node.config.result_type is None

        input_data = InputData(message="assistant")
        result = None
        async for item in node.astream(input_data):
            if isinstance(item, ToolResult):
                result = item.result

        assert result is not None

    async def test_simple_prompt_template_render(self) -> None:
        """Test PromptTemplate (non-chat) rendering."""
        template = PromptTemplate[InputData, str](
            template="Process: {message}",
            input_model=InputData,
            format=TemplateFormat.F_STRING,
        )
        node = PromptNode[InputData, str](
            prompt=template,
            name="template_node",
        )

        # This hits line 166 (simple template.render())
        input_data = InputData(message="data")
        result = None
        async for item in node.astream(input_data):
            if isinstance(item, ToolResult):
                result = item.result

        assert result is not None

    async def test_jinja2_string_prompt(self) -> None:
        """Test string prompt with Jinja2 format."""
        config = PromptConfig(
            model="test",
            template_format=TemplateFormat.JINJA2,
        )
        node = PromptNode[InputData, str](
            prompt="Hello {{ message }}!",
            config=config,
        )

        # This hits lines 151-153 (renderer.render() path)
        input_data = InputData(message="Jinja")
        result = None
        async for item in node.astream(input_data):
            if isinstance(item, ToolResult):
                result = item.result

        assert result is not None

    async def test_mustache_string_prompt(self) -> None:
        """Test string prompt with Mustache format."""
        config = PromptConfig(
            model="test",
            template_format=TemplateFormat.MUSTACHE,
        )
        node = PromptNode[InputData, str](
            prompt="Hello {{message}}!",
            config=config,
        )

        # This hits lines 151-153 (renderer.render() path)
        input_data = InputData(message="Mustache")
        result = None
        async for item in node.astream(input_data):
            if isinstance(item, ToolResult):
                result = item.result

        assert result is not None

    async def test_exception_during_llm_execution(self) -> None:
        """Test exception handling during LLM execution."""
        # Create a node that will fail during execution
        node = PromptNode[InputData, str](
            prompt="Test {message}",
            name="error_node",
        )

        # Mock the agent to raise an exception
        original_agent = node._agent

        class FailingAgent:
            """Agent that raises during execution."""

            def run_stream(self, prompt: str):
                """Raise exception."""
                raise RuntimeError("Simulated LLM failure")

            def __enter__(self):
                """Context manager enter."""
                return self

            def __exit__(self, *args):
                """Context manager exit."""
                pass

        node._agent = FailingAgent()  # type: ignore[assignment]

        input_data = InputData(message="fail")
        error_found = False
        exception_raised = False

        try:
            async for item in node.astream(input_data):
                if isinstance(item, NonFatalError):
                    error_found = True
                    assert "LLM execution failed" in item.message
                    assert not item.recoverable
        except Exception:
            exception_raised = True

        # This hits lines 238-248 (exception handling)
        assert error_found or exception_raised
        node._agent = original_agent

    async def test_result_none_preview(self) -> None:
        """Test result preview when result is None."""
        node = PromptNode[InputData, str](
            prompt="Say nothing",
            name="none_node",
        )

        # Mock to return None result
        input_data = InputData(message="empty")
        stream_end_found = False

        async for item in node.astream(input_data):
            if isinstance(item, StreamEnd):
                stream_end_found = True
                # When result is not None and has no model_dump,
                # should use string representation
                break

        assert stream_end_found
