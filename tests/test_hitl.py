"""Tests for HITL (Human-in-the-Loop) functionality."""

from pydantic import BaseModel
import pytest

from pydantic_flow import ApprovalNode
from pydantic_flow import Flow
from pydantic_flow import HandlerPriority
from pydantic_flow import HumanInputRequest
from pydantic_flow import HumanNode
from pydantic_flow import HumanResponse
from pydantic_flow import InterruptDecision
from pydantic_flow import InterruptionRequested
from pydantic_flow.hitl.interrupts import FlowCheckpoint
from pydantic_flow.nodes.base import BaseNode
from pydantic_flow.streaming.base import ProgressItem
from pydantic_flow.streaming.core_events import StreamStart


class SimpleInput(BaseModel):
    """Test input model."""

    text: str


class SimpleOutput(BaseModel):
    """Test output model."""

    result: str


class ReviewResult(BaseModel):
    """Test review result model."""

    approved: bool
    comments: str = ""


# Test InterruptDecision


def test_interrupt_decision_proceed():
    """Test creating a proceed decision."""
    decision = InterruptDecision.proceed(reason="All good")
    assert not decision.should_interrupt
    assert decision.reason == "All good"
    assert decision.replacement_value is None


def test_interrupt_decision_interrupt():
    """Test creating an interrupt decision."""
    decision = InterruptDecision.interrupt(
        reason="Need review",
        replacement_value="test_value",
        metadata={"key": "value"},
    )
    assert decision.should_interrupt
    assert decision.reason == "Need review"
    assert decision.replacement_value == "test_value"
    assert decision.metadata == {"key": "value"}


# Test FlowCheckpoint


def test_flow_checkpoint_creation():
    """Test creating a flow checkpoint."""
    checkpoint = FlowCheckpoint(
        flow_id="test-flow-123",
        run_id="run-456",
        interrupted_node_id="node-1",
        node_states={"node-1": {"key": "value"}},
        edge_history=[("node-1", "node-2")],
        metadata={"test": "data"},
    )
    assert checkpoint.flow_id == "test-flow-123"
    assert checkpoint.run_id == "run-456"
    assert checkpoint.interrupted_node_id == "node-1"
    assert checkpoint.node_states == {"node-1": {"key": "value"}}
    assert checkpoint.edge_history == [("node-1", "node-2")]
    assert checkpoint.metadata == {"test": "data"}


# Test Node-Level Interrupt Handlers


@pytest.mark.asyncio
async def test_node_register_interrupt_handler():
    """Test registering interrupt handlers on a node."""

    class TestNode(BaseNode[SimpleInput, SimpleOutput]):
        async def astream(self, input_data):
            yield StreamStart(run_id="", node_id=self.name)

    node = TestNode(name="test")

    # Initially no handlers
    assert len(node._interrupt_handlers) == 0

    # Register a handler
    async def handler(item: ProgressItem) -> InterruptDecision:
        return InterruptDecision.proceed()

    node.register_interrupt_handler(handler, priority=50)
    assert len(node._interrupt_handlers) == 1
    assert node._interrupt_handlers[0].priority == 50

    # Register another with different priority
    node.register_interrupt_handler(handler, priority=25)
    assert len(node._interrupt_handlers) == 2
    # Should be sorted by priority
    assert node._interrupt_handlers[0].priority == 25
    assert node._interrupt_handlers[1].priority == 50


@pytest.mark.asyncio
async def test_node_clear_interrupt_handlers():
    """Test clearing interrupt handlers."""

    class TestNode(BaseNode[SimpleInput, SimpleOutput]):
        async def astream(self, input_data):
            yield StreamStart(run_id="", node_id=self.name)

    node = TestNode(name="test")

    async def handler(item: ProgressItem) -> InterruptDecision:
        return InterruptDecision.proceed()

    node.register_interrupt_handler(handler)
    assert len(node._interrupt_handlers) == 1

    node.clear_interrupt_handlers()
    assert len(node._interrupt_handlers) == 0


@pytest.mark.asyncio
async def test_node_check_interrupt_handlers_proceed():
    """Test _check_interrupt_handlers when handler returns proceed."""

    class TestNode(BaseNode[SimpleInput, SimpleOutput]):
        async def astream(self, input_data):
            yield StreamStart(run_id="", node_id=self.name)

    node = TestNode(name="test")

    async def handler1(item: ProgressItem) -> InterruptDecision:
        return InterruptDecision.proceed(reason="handler1 ok")

    async def handler2(item: ProgressItem) -> InterruptDecision:
        return InterruptDecision.proceed(reason="handler2 ok")

    node.register_interrupt_handler(handler1, priority=10)
    node.register_interrupt_handler(handler2, priority=20)

    item = StreamStart(run_id="test", node_id="test")
    decision = await node._check_interrupt_handlers(item)

    assert not decision.should_interrupt


@pytest.mark.asyncio
async def test_node_check_interrupt_handlers_interrupt():
    """Test _check_interrupt_handlers when handler requests interrupt."""

    class TestNode(BaseNode[SimpleInput, SimpleOutput]):
        async def astream(self, input_data):
            yield StreamStart(run_id="", node_id=self.name)

    node = TestNode(name="test")

    async def handler1(item: ProgressItem) -> InterruptDecision:
        return InterruptDecision.proceed()

    async def handler2(item: ProgressItem) -> InterruptDecision:
        return InterruptDecision.interrupt(reason="Stop here")

    async def handler3(item: ProgressItem) -> InterruptDecision:
        return InterruptDecision.proceed()

    # handler2 has lowest priority, should execute first
    node.register_interrupt_handler(handler1, priority=30)
    node.register_interrupt_handler(handler2, priority=10)
    node.register_interrupt_handler(handler3, priority=50)

    item = StreamStart(run_id="test", node_id="test")
    decision = await node._check_interrupt_handlers(item)

    # Should interrupt from handler2
    assert decision.should_interrupt
    assert decision.reason == "Stop here"


# Test Flow-Level HITL


def test_flow_has_flow_id():
    """Test that flows have unique flow_id."""
    flow1 = Flow(input_type=SimpleInput, output_type=SimpleOutput)
    flow2 = Flow(input_type=SimpleInput, output_type=SimpleOutput)

    assert flow1.flow_id != ""
    assert flow2.flow_id != ""
    assert flow1.flow_id != flow2.flow_id


def test_flow_register_interrupt_handler():
    """Test registering flow-level interrupt handlers."""
    flow = Flow(input_type=SimpleInput, output_type=SimpleOutput)

    async def handler(item: ProgressItem) -> InterruptDecision:
        return InterruptDecision.proceed()

    flow.register_interrupt_handler(handler, priority=HandlerPriority.NORMAL)
    assert len(flow._interrupt_handlers) == 1


def test_flow_clear_interrupt_handlers():
    """Test clearing flow-level interrupt handlers."""
    flow = Flow(input_type=SimpleInput, output_type=SimpleOutput)

    async def handler(item: ProgressItem) -> InterruptDecision:
        return InterruptDecision.proceed()

    flow.register_interrupt_handler(handler)
    assert len(flow._interrupt_handlers) == 1

    flow.clear_interrupt_handlers()
    assert len(flow._interrupt_handlers) == 0


def test_flow_create_checkpoint():
    """Test flow checkpoint creation."""
    flow = Flow(input_type=SimpleInput, output_type=SimpleOutput)
    flow._results = {"node1": "result1"}
    flow._edge_history = [("node1", "node2")]

    checkpoint = flow._create_checkpoint("node2", "run-123")

    assert checkpoint.flow_id == flow.flow_id
    assert checkpoint.run_id == "run-123"
    assert checkpoint.interrupted_node_id == "node2"
    assert checkpoint.node_states == {"node1": "result1"}
    assert checkpoint.edge_history == [("node1", "node2")]


# Test HumanNode


@pytest.mark.asyncio
async def test_human_node_always_interrupts():
    """Test that HumanNode always raises InterruptionRequested."""
    node = HumanNode[SimpleInput, HumanResponse](
        prompt="Please review this",
        name="human_review",
    )

    input_data = SimpleInput(text="test")

    with pytest.raises(InterruptionRequested) as exc_info:
        async for _ in node.astream(input_data):
            pass

    exc: InterruptionRequested = exc_info.value  # type: ignore
    assert exc.checkpoint.interrupted_node_id == "human_review"
    assert "Human input required" in exc.decision.reason


@pytest.mark.asyncio
async def test_human_node_dynamic_prompt():
    """Test HumanNode with dynamic prompt function."""
    node = HumanNode[SimpleInput, HumanResponse](
        prompt=lambda data: f"Review: {data.text}",
        name="human_review",
    )

    input_data = SimpleInput(text="important content")

    with pytest.raises(InterruptionRequested) as exc_info:
        async for _ in node.astream(input_data):
            pass

    exc: InterruptionRequested = exc_info.value  # type: ignore
    request = exc.decision.replacement_value
    assert isinstance(request, HumanInputRequest)
    assert request.prompt == "Review: important content"


@pytest.mark.asyncio
async def test_human_node_with_options():
    """Test HumanNode with selection options."""
    node = HumanNode[SimpleInput, HumanResponse](
        prompt="Choose an option",
        input_type="choice",
        options=["option1", "option2", "option3"],
        name="human_choice",
    )

    input_data = SimpleInput(text="test")

    with pytest.raises(InterruptionRequested) as exc_info:
        async for _ in node.astream(input_data):
            pass

    exc: InterruptionRequested = exc_info.value  # type: ignore
    request = exc.decision.replacement_value
    assert request.input_type == "choice"
    assert request.options == ["option1", "option2", "option3"]


def test_human_node_parse_response_with_parser():
    """Test HumanNode response parsing with custom parser."""
    node = HumanNode[SimpleInput, ReviewResult](
        prompt="Review this",
        response_parser=lambda resp: ReviewResult(
            approved=resp.approved, comments=str(resp.value)
        ),
    )

    response = HumanResponse(value="Looks good", approved=True)
    result = node.parse_response(response)

    assert isinstance(result, ReviewResult)
    assert result.approved is True
    assert result.comments == "Looks good"


def test_human_node_parse_response_without_parser():
    """Test HumanNode response parsing without parser."""
    node = HumanNode[SimpleInput, HumanResponse](
        prompt="Review this",
    )

    response = HumanResponse(value="test", approved=True)
    result = node.parse_response(response)

    assert result is response


def test_human_node_parse_response_without_parser_wrong_type():
    """Test HumanNode response parsing without parser returns response as-is."""
    node = HumanNode[SimpleInput, ReviewResult](
        prompt="Review this",
    )

    response = HumanResponse(value="test", approved=True)

    # Without parser, returns response as-is (caller responsible for type safety)
    result = node.parse_response(response)
    assert result is response


# Test ApprovalNode


@pytest.mark.asyncio
async def test_approval_node_always_interrupts():
    """Test that ApprovalNode always interrupts."""
    node = ApprovalNode[SimpleInput](
        prompt="Approve this action?",
        name="approval",
    )

    input_data = SimpleInput(text="test")

    with pytest.raises(InterruptionRequested) as exc_info:
        async for _ in node.astream(input_data):
            pass

    exc: InterruptionRequested = exc_info.value  # type: ignore
    request = exc.decision.replacement_value
    assert request.input_type == "approval"
    assert request.options == ["approve", "reject"]


@pytest.mark.asyncio
async def test_approval_node_dynamic_prompt():
    """Test ApprovalNode with dynamic prompt."""
    node = ApprovalNode[SimpleInput](
        prompt=lambda data: f"Approve action on: {data.text}?",
        name="approval",
    )

    input_data = SimpleInput(text="user-123")

    with pytest.raises(InterruptionRequested) as exc_info:
        async for _ in node.astream(input_data):
            pass

    exc: InterruptionRequested = exc_info.value  # type: ignore
    request = exc.decision.replacement_value
    assert request.prompt == "Approve action on: user-123?"


# Test HandlerPriority


def test_handler_priority_values():
    """Test HandlerPriority enum values."""
    assert HandlerPriority.CRITICAL == 0
    assert HandlerPriority.HIGH == 26
    assert HandlerPriority.NORMAL == 51
    assert HandlerPriority.LOW == 76


def test_handler_priority_ordering():
    """Test that handlers are executed in priority order."""
    priorities = [
        HandlerPriority.LOW,
        HandlerPriority.CRITICAL,
        HandlerPriority.HIGH,
        HandlerPriority.NORMAL,
    ]
    sorted_priorities = sorted(priorities)

    assert sorted_priorities == [
        HandlerPriority.CRITICAL,
        HandlerPriority.HIGH,
        HandlerPriority.NORMAL,
        HandlerPriority.LOW,
    ]


# Test HumanInputRequest


def test_human_input_request_creation():
    """Test creating HumanInputRequest."""
    request = HumanInputRequest(
        prompt="Enter your name",
        context={"user_id": "123"},
        input_type="text",
        options=None,
        metadata={"source": "test"},
    )

    assert request.prompt == "Enter your name"
    assert request.context == {"user_id": "123"}
    assert request.input_type == "text"
    assert request.options is None
    assert request.metadata == {"source": "test"}


def test_human_input_request_defaults():
    """Test HumanInputRequest default values."""
    request = HumanInputRequest(prompt="Test prompt")

    assert request.prompt == "Test prompt"
    assert request.context == {}
    assert request.input_type == "text"
    assert request.options is None
    assert request.metadata == {}


# Test HumanResponse


def test_human_response_creation():
    """Test creating HumanResponse."""
    response = HumanResponse(
        value="user input",
        approved=True,
        metadata={"timestamp": "2025-01-01"},
    )

    assert response.value == "user input"
    assert response.approved is True
    assert response.metadata == {"timestamp": "2025-01-01"}


def test_human_response_defaults():
    """Test HumanResponse default values."""
    response = HumanResponse(value="test")

    assert response.value == "test"
    assert response.approved is True
    assert response.metadata == {}
