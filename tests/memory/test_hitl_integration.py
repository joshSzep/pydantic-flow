"""Tests for conversation memory HITL integration."""

import contextlib
from unittest.mock import MagicMock

from pydantic import BaseModel
import pytest

from pydantic_flow.core.errors import FlowCheckpoint
from pydantic_flow.core.errors import InterruptionRequested
from pydantic_flow.flow.flow import Flow
from pydantic_flow.memory.config import MemoryConfig
from pydantic_flow.memory.memory import ConversationMemory
from pydantic_flow.nodes.base import BaseNode
from pydantic_flow.streaming.events import InterruptDecision
from pydantic_flow.streaming.events import ProgressItem
from pydantic_flow.streaming.events import StreamEnd
from pydantic_flow.streaming.events import StreamStart
from pydantic_flow.streaming.events import ToolResult


class SimpleInput(BaseModel):
    """Test input model."""

    value: int


class SimpleOutput(BaseModel):
    """Test output model."""

    result: int


class CounterNode(BaseNode[SimpleInput, int]):
    """Simple test node that counts."""

    async def astream(self, input_data: SimpleInput):
        """Stream a simple result."""
        yield StreamStart(run_id="test", node_id=self.name)
        result = input_data.value * 2
        yield ToolResult(
            run_id="test",
            node_id=self.name,
            tool_name=self.name,
            result=result,
        )
        yield StreamEnd(
            run_id="test",
            node_id=self.name,
            result_preview={"value": result},
        )


def test_flow_checkpoint_with_memory():
    """Test FlowCheckpoint can store conversation memory."""
    # Create mock messages
    mock_msg1 = MagicMock()
    mock_msg2 = MagicMock()

    checkpoint = FlowCheckpoint(
        flow_id="test-flow",
        run_id="test-run",
        interrupted_node_id="node1",
        node_states={"node1": 42},
        edge_history=[],
        conversation_memory=[mock_msg1, mock_msg2],
    )

    assert checkpoint.conversation_memory is not None
    assert len(checkpoint.conversation_memory) == 2
    assert checkpoint.conversation_memory[0] == mock_msg1
    assert checkpoint.conversation_memory[1] == mock_msg2


def test_flow_checkpoint_without_memory():
    """Test FlowCheckpoint works without conversation memory."""
    checkpoint = FlowCheckpoint(
        flow_id="test-flow",
        run_id="test-run",
        interrupted_node_id="node1",
        node_states={"node1": 42},
        edge_history=[],
    )

    assert checkpoint.conversation_memory is None


def test_flow_create_checkpoint_with_memory():
    """Test Flow._create_checkpoint captures conversation memory."""
    flow = Flow[SimpleInput, SimpleOutput](
        input_type=SimpleInput,
        output_type=SimpleOutput,
        memory_config=MemoryConfig(enable_conversation_memory=True),
    )

    # Add messages to memory
    mock_msg = MagicMock()
    assert flow._conversation_memory is not None
    flow._conversation_memory.append(mock_msg)

    checkpoint = flow._create_checkpoint("node1", "run-123")

    assert checkpoint.flow_id == flow.flow_id
    assert checkpoint.interrupted_node_id == "node1"
    assert checkpoint.run_id == "run-123"
    assert checkpoint.conversation_memory is not None
    assert len(checkpoint.conversation_memory) == 1
    assert checkpoint.conversation_memory[0] == mock_msg


def test_flow_create_checkpoint_without_memory():
    """Test Flow._create_checkpoint works without memory enabled."""
    flow = Flow[SimpleInput, SimpleOutput](
        input_type=SimpleInput,
        output_type=SimpleOutput,
        memory_config=MemoryConfig(enable_conversation_memory=False),
    )

    checkpoint = flow._create_checkpoint("node1", "run-123")

    assert checkpoint.conversation_memory is None


@pytest.mark.asyncio
async def test_flow_resume_restores_memory():
    """Test Flow.resume restores conversation memory from checkpoint."""
    # Create flow with memory
    flow = Flow[SimpleInput, SimpleOutput](
        input_type=SimpleInput,
        output_type=SimpleOutput,
        memory_config=MemoryConfig(enable_conversation_memory=True),
    )

    # Add a simple node
    node = CounterNode(name="counter")
    flow.add_nodes(node)
    flow.compile()

    # Create checkpoint with memory
    mock_msg1 = MagicMock()
    mock_msg2 = MagicMock()
    checkpoint = FlowCheckpoint(
        flow_id=flow.flow_id,
        run_id="test-run",
        interrupted_node_id="counter",
        node_states={},
        edge_history=[],
        conversation_memory=[mock_msg1, mock_msg2],
    )

    # Memory should be empty initially (just created)
    assert flow._conversation_memory is not None
    assert len(flow._conversation_memory) == 0

    # Resume should restore memory
    with contextlib.suppress(Exception):
        # We expect this to fail since counter is in execution_order
        # but has no result yet - that's okay, we're testing memory restore
        await flow.resume(checkpoint, SimpleInput(value=1))

    # Check memory was restored
    assert flow._conversation_memory is not None
    assert len(flow._conversation_memory) == 2
    restored_messages = flow._conversation_memory.get()
    assert restored_messages[0] == mock_msg1
    assert restored_messages[1] == mock_msg2


@pytest.mark.asyncio
async def test_flow_resume_creates_memory_if_needed():
    """Test Flow.resume creates memory if checkpoint has it but flow doesn't."""
    # Create flow WITHOUT memory
    flow = Flow[SimpleInput, SimpleOutput](
        input_type=SimpleInput,
        output_type=SimpleOutput,
        memory_config=MemoryConfig(enable_conversation_memory=False),
    )

    # Add a simple node
    node = CounterNode(name="counter")
    flow.add_nodes(node)
    flow.compile()

    # Create checkpoint WITH memory
    mock_msg = MagicMock()
    checkpoint = FlowCheckpoint(
        flow_id=flow.flow_id,
        run_id="test-run",
        interrupted_node_id="counter",
        node_states={},
        edge_history=[],
        conversation_memory=[mock_msg],
    )

    # Memory should be None initially
    assert flow._conversation_memory is None

    # Resume should create memory
    with contextlib.suppress(Exception):
        # We expect this to fail - that's okay, we're testing memory creation
        await flow.resume(checkpoint, SimpleInput(value=1))

    # Check memory was created and populated
    assert flow._conversation_memory is not None
    assert len(flow._conversation_memory) == 1
    assert flow._conversation_memory.get()[0] == mock_msg


@pytest.mark.asyncio
async def test_interruption_captures_memory():
    """Test that InterruptionRequested captures conversation memory."""
    # Create flow with memory
    flow = Flow[SimpleInput, SimpleOutput](
        input_type=SimpleInput,
        output_type=SimpleOutput,
        memory_config=MemoryConfig(enable_conversation_memory=True),
    )

    # Add mock message to memory
    mock_msg = MagicMock()
    assert flow._conversation_memory is not None
    flow._conversation_memory.append(mock_msg)

    # Add simple node with name matching output field
    node = CounterNode(name="result")
    flow.add_nodes(node)
    flow.compile()

    async def always_interrupt(item: ProgressItem) -> InterruptDecision:
        return InterruptDecision.interrupt(reason="Test interrupt")

    flow.register_interrupt_handler(always_interrupt)

    # Run should raise InterruptionRequested with memory captured
    with pytest.raises(InterruptionRequested) as exc_info:
        await flow.run(SimpleInput(value=1))

    assert isinstance(exc_info.value, InterruptionRequested)
    checkpoint = exc_info.value.checkpoint
    assert checkpoint.conversation_memory is not None
    assert len(checkpoint.conversation_memory) == 1
    assert checkpoint.conversation_memory[0] == mock_msg


def test_conversation_memory_serialization():
    """Test that ConversationMemory with ModelMessages can be serialized."""
    # Create memory with mock messages
    mock_msg1 = MagicMock()
    mock_msg2 = MagicMock()
    memory = ConversationMemory(initial_messages=[mock_msg1, mock_msg2])

    # Create checkpoint
    checkpoint = FlowCheckpoint(
        flow_id="test",
        run_id="test",
        interrupted_node_id="node1",
        node_states={},
        edge_history=[],
        conversation_memory=memory.get(),
    )

    # Verify we can serialize to dict (Pydantic validation)
    checkpoint_dict = checkpoint.model_dump()
    assert "conversation_memory" in checkpoint_dict
    assert checkpoint_dict["conversation_memory"] is not None

    # Verify we can reconstruct
    checkpoint2 = FlowCheckpoint(**checkpoint_dict)
    assert checkpoint2.conversation_memory is not None
    assert len(checkpoint2.conversation_memory) == 2
