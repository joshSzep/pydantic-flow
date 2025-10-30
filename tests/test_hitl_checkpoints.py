"""Tests for HITL checkpoint integration with CheckpointStore."""

from pydantic import BaseModel
import pytest

from pydantic_flow import Flow
from pydantic_flow import HandlerPriority
from pydantic_flow import InterruptDecision
from pydantic_flow import InterruptionRequested
from pydantic_flow import PromptNode
from pydantic_flow.core.run_config import RunConfig
from pydantic_flow.hitl.checkpoints.interface import CheckpointQuery
from pydantic_flow.hitl.checkpoints.interface import RunId
from pydantic_flow.hitl.checkpoints.interface import filter_interrupted
from pydantic_flow.hitl.checkpoints.interface import list_interrupted
from pydantic_flow.hitl.checkpoints.memory import InMemoryCheckpointStore
from pydantic_flow.streaming.events import ProgressItem
from pydantic_flow.streaming.events import StreamEnd


class SimpleInput(BaseModel):
    """Test input model."""

    text: str


class SimpleOutput(BaseModel):
    """Test output model."""

    result: str


@pytest.mark.asyncio
async def test_interrupted_checkpoint_has_metadata():
    """Test that interrupted checkpoints have is_interrupted and metadata."""
    flow = Flow(input_type=SimpleInput, output_type=SimpleOutput)
    node = PromptNode[SimpleInput, SimpleOutput](
        name="processor", prompt="Process: {text}"
    )
    flow.add_nodes(node)

    store = InMemoryCheckpointStore()
    run_id = "test_interrupt_001"
    config = RunConfig(checkpoint_store=store, run_id=run_id)

    # Register interrupt handler with reason and metadata
    async def interrupt_with_metadata(item: ProgressItem) -> InterruptDecision:
        if isinstance(item, StreamEnd):
            return InterruptDecision.interrupt(
                reason="Human review required",
                metadata={"review_type": "final", "priority": "high"},
            )
        return InterruptDecision.proceed()

    flow.register_interrupt_handler(
        callback=interrupt_with_metadata, priority=HandlerPriority.HIGH
    )

    # Run flow - should be interrupted
    with pytest.raises(InterruptionRequested):
        await flow.run(SimpleInput(text="test data"), config=config)

    # Verify checkpoint in store has interrupt metadata
    query = CheckpointQuery(run_id=RunId(run_id))
    checkpoints, _ = await store.list(query)
    assert len(checkpoints) == 1

    envelope = checkpoints[0]
    assert envelope.is_interrupted is True
    assert envelope.interrupt_reason == "Human review required"
    assert envelope.interrupt_metadata is not None
    assert envelope.interrupt_metadata["review_type"] == "final"
    assert envelope.interrupt_metadata["priority"] == "high"


@pytest.mark.asyncio
async def test_checkpoint_saved_with_node_state():
    """Test that checkpoint is saved with correct node state on interruption."""
    # Create a simple flow to test checkpoint persistence
    flow = Flow(input_type=SimpleInput, output_type=SimpleOutput)
    node = PromptNode[SimpleInput, SimpleOutput](
        name="processor", prompt="Process: {text}"
    )
    flow.add_nodes(node)

    store = InMemoryCheckpointStore()
    run_id = "test_checkpoint_state"
    config = RunConfig(checkpoint_store=store, run_id=run_id)

    # Interrupt after node completes
    async def interrupt_after_completion(item: ProgressItem) -> InterruptDecision:
        if isinstance(item, StreamEnd):
            return InterruptDecision.interrupt(reason="Review after completion")
        return InterruptDecision.proceed()

    flow.register_interrupt_handler(
        callback=interrupt_after_completion, priority=HandlerPriority.HIGH
    )

    # Run and catch interruption
    input_data = SimpleInput(text="test")
    with pytest.raises(InterruptionRequested) as exc_info:
        await flow.run(input_data, config=config)

    exception: InterruptionRequested = exc_info.value  # type: ignore[assignment]
    checkpoint_id = exception.checkpoint.metadata["checkpoint_id"]

    # Verify node state was captured
    assert "processor" in exception.checkpoint.node_states
    assert exception.checkpoint.node_states["processor"] is not None

    # Verify checkpoint exists in store
    envelope = await store.get(RunId(run_id), checkpoint_id)
    assert envelope is not None
    assert envelope.is_interrupted is True
    assert envelope.interrupt_reason == "Review after completion"


@pytest.mark.asyncio
async def test_checkpoint_envelope_structure():
    """Test that CheckpointEnvelope has correct structure after interrupt."""
    flow = Flow(input_type=SimpleInput, output_type=SimpleOutput)
    node = PromptNode[SimpleInput, SimpleOutput](
        name="processor", prompt="Process: {text}"
    )
    flow.add_nodes(node)

    store = InMemoryCheckpointStore()
    run_id = "test_envelope_structure"
    config = RunConfig(checkpoint_store=store, run_id=run_id)

    # Interrupt after node completes
    async def interrupt_at_end(item: ProgressItem) -> InterruptDecision:
        if isinstance(item, StreamEnd):
            return InterruptDecision.interrupt(reason="Test envelope structure")
        return InterruptDecision.proceed()

    flow.register_interrupt_handler(callback=interrupt_at_end, priority=0)

    # Run and catch interruption
    input_data = SimpleInput(text="test")
    with pytest.raises(InterruptionRequested):
        await flow.run(input_data, config=config)

    # Get envelope from store and verify structure
    query = CheckpointQuery(run_id=RunId(run_id))
    checkpoints, _ = await store.list(query)
    assert len(checkpoints) == 1

    envelope = checkpoints[0]
    assert envelope.run_id == RunId(run_id)
    assert envelope.node_id == "processor"
    assert envelope.is_interrupted is True
    assert envelope.interrupt_reason == "Test envelope structure"
    assert envelope.checkpoint.flow_id == flow.flow_id
    assert "processor" in envelope.checkpoint.node_states


@pytest.mark.asyncio
async def test_filter_interrupted_checkpoints():
    """Test filtering interrupted checkpoints from a list."""
    flow = Flow(input_type=SimpleInput, output_type=SimpleOutput)
    node = PromptNode[SimpleInput, SimpleOutput](
        name="processor", prompt="Process: {text}"
    )
    flow.add_nodes(node)

    store = InMemoryCheckpointStore()

    # Create interrupted checkpoint
    async def interrupt_handler(item: ProgressItem) -> InterruptDecision:
        return InterruptDecision.interrupt(reason="Test interrupt")

    flow.register_interrupt_handler(callback=interrupt_handler, priority=0)

    run_id_1 = "test_filter_001"
    config_1 = RunConfig(checkpoint_store=store, run_id=run_id_1)
    with pytest.raises(InterruptionRequested):
        await flow.run(SimpleInput(text="test1"), config=config_1)

    # Create normal checkpoint by manually saving
    flow.clear_interrupt_handlers()
    run_id_2 = "test_filter_002"
    from pydantic_flow.hitl.checkpoints.interface import CheckpointEnvelope
    from pydantic_flow.hitl.checkpoints.interface import CheckpointId
    from pydantic_flow.hitl.checkpoints.interface import generate_checkpoint_id
    from pydantic_flow.hitl.interrupts import FlowCheckpoint

    normal_checkpoint = FlowCheckpoint(
        flow_id=flow.flow_id,
        run_id=run_id_2,
        interrupted_node_id="processor",
        node_states={},
        edge_history=[],
    )
    normal_envelope = CheckpointEnvelope(
        id=CheckpointId(generate_checkpoint_id()),
        run_id=RunId(run_id_2),
        checkpoint=normal_checkpoint,
        is_interrupted=False,
    )
    await store.save(normal_envelope)

    # Get all checkpoints and filter
    all_checkpoints, _ = await store.list(CheckpointQuery(limit=100))
    interrupted = filter_interrupted(all_checkpoints)

    assert len(all_checkpoints) == 2
    assert len(interrupted) == 1
    assert interrupted[0].is_interrupted is True


@pytest.mark.asyncio
async def test_list_interrupted_helper():
    """Test the list_interrupted helper function."""
    flow = Flow(input_type=SimpleInput, output_type=SimpleOutput)
    node = PromptNode[SimpleInput, SimpleOutput](
        name="processor", prompt="Process: {text}"
    )
    flow.add_nodes(node)

    store = InMemoryCheckpointStore()

    # Create interrupted checkpoint
    async def interrupt_handler(item: ProgressItem) -> InterruptDecision:
        return InterruptDecision.interrupt(reason="Test")

    flow.register_interrupt_handler(callback=interrupt_handler, priority=0)

    run_id = "test_list_001"
    config = RunConfig(checkpoint_store=store, run_id=run_id)
    with pytest.raises(InterruptionRequested):
        await flow.run(SimpleInput(text="test"), config=config)

    # Use list_interrupted helper
    query = CheckpointQuery(run_id=RunId(run_id))
    interrupted_checkpoints, _ = await list_interrupted(store, query)

    assert len(interrupted_checkpoints) == 1
    assert interrupted_checkpoints[0].is_interrupted is True
    assert interrupted_checkpoints[0].interrupt_reason == "Test"


@pytest.mark.asyncio
async def test_backwards_compatibility_without_interrupt_fields():
    """Test that checkpoints without interrupt fields still work."""
    from pydantic_flow.hitl.checkpoints.interface import CheckpointEnvelope
    from pydantic_flow.hitl.checkpoints.interface import CheckpointId
    from pydantic_flow.hitl.checkpoints.interface import RunId
    from pydantic_flow.hitl.checkpoints.interface import generate_checkpoint_id
    from pydantic_flow.hitl.interrupts import FlowCheckpoint

    # Create envelope without interrupt fields (old format)
    checkpoint = FlowCheckpoint(
        flow_id="test-flow",
        run_id="test-run",
        interrupted_node_id="node1",
        node_states={},
        edge_history=[],
    )

    # This should work with defaults
    envelope = CheckpointEnvelope(
        id=CheckpointId(generate_checkpoint_id()),
        run_id=RunId("test-run"),
        checkpoint=checkpoint,
    )

    assert envelope.is_interrupted is False
    assert envelope.interrupt_reason is None
    assert envelope.interrupt_metadata is None


@pytest.mark.asyncio
async def test_interrupt_reason_without_metadata():
    """Test interrupt with reason but no metadata."""
    flow = Flow(input_type=SimpleInput, output_type=SimpleOutput)
    node = PromptNode[SimpleInput, SimpleOutput](
        name="processor", prompt="Process: {text}"
    )
    flow.add_nodes(node)

    store = InMemoryCheckpointStore()
    run_id = "test_reason_only"
    config = RunConfig(checkpoint_store=store, run_id=run_id)

    # Interrupt with reason only
    async def interrupt_with_reason(item: ProgressItem) -> InterruptDecision:
        return InterruptDecision.interrupt(reason="Simple reason")

    flow.register_interrupt_handler(callback=interrupt_with_reason, priority=0)

    with pytest.raises(InterruptionRequested):
        await flow.run(SimpleInput(text="test"), config=config)

    query = CheckpointQuery(run_id=RunId(run_id))
    checkpoints, _ = await store.list(query)
    envelope = checkpoints[0]

    assert envelope.is_interrupted is True
    assert envelope.interrupt_reason == "Simple reason"
    assert envelope.interrupt_metadata is None


@pytest.mark.asyncio
async def test_multiple_interrupted_checkpoints_same_run():
    """Test multiple interruptions in the same run create separate checkpoints."""
    flow = Flow(input_type=SimpleInput, output_type=SimpleOutput)
    node1 = PromptNode[SimpleInput, SimpleOutput](name="step1", prompt="Step 1: {text}")
    node2 = PromptNode[SimpleOutput, SimpleOutput](
        name="step2", prompt="Step 2: {result}", input=node1.output
    )
    flow.add_nodes(node1, node2)

    store = InMemoryCheckpointStore()
    run_id = "test_multiple_001"
    input_data = SimpleInput(text="test")

    # First interrupt at step1
    interrupt_count = 0

    async def interrupt_first_time(item: ProgressItem) -> InterruptDecision:
        nonlocal interrupt_count
        if isinstance(item, StreamEnd):
            interrupt_count += 1
            if interrupt_count == 1:
                return InterruptDecision.interrupt(reason="First interrupt")
        return InterruptDecision.proceed()

    flow.register_interrupt_handler(callback=interrupt_first_time, priority=0)

    config = RunConfig(checkpoint_store=store, run_id=run_id)
    with pytest.raises(InterruptionRequested):
        await flow.run(input_data, config=config)

    # Verify one checkpoint
    query = CheckpointQuery(run_id=RunId(run_id))
    checkpoints, _ = await store.list(query)
    assert len(checkpoints) == 1
    assert checkpoints[0].interrupt_reason == "First interrupt"
