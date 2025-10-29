"""Integration tests for checkpoint persistence with Flow execution."""

from pydantic import BaseModel
import pytest

from pydantic_flow import Flow
from pydantic_flow import PromptNode
from pydantic_flow.checkpoints.interface import CheckpointQuery
from pydantic_flow.checkpoints.interface import RunId
from pydantic_flow.checkpoints.memory import InMemoryCheckpointStore
from pydantic_flow.core.errors import InterruptionRequested
from pydantic_flow.core.run_config import RunConfig
from pydantic_flow.streaming.events import InterruptDecision
from pydantic_flow.streaming.events import ProgressItem


class SimpleInput(BaseModel):
    """Test input model."""

    value: str


class SimpleOutput(BaseModel):
    """Test output model."""

    result: str


@pytest.mark.asyncio
async def test_checkpoint_persisted_on_interruption():
    """Test that checkpoints are persisted when flow is interrupted."""
    # Create flow
    flow = Flow(input_type=SimpleInput, output_type=SimpleOutput)
    node = PromptNode[SimpleInput, SimpleOutput](
        name="processor", prompt="Process: {value}"
    )
    flow.add_nodes(node)

    # Create checkpoint store
    store = InMemoryCheckpointStore()
    run_id = "test_run_001"
    config = RunConfig(checkpoint_store=store, run_id=run_id)

    # Register interrupt handler
    async def always_interrupt(item: ProgressItem) -> InterruptDecision:
        return InterruptDecision(should_interrupt=True, reason="Test interrupt")

    flow.register_interrupt_handler(callback=always_interrupt, priority=0)

    # Run flow - should be interrupted
    with pytest.raises(InterruptionRequested) as exc_info:
        await flow.run(SimpleInput(value="test"), config=config)

    # Verify checkpoint was persisted
    exception: InterruptionRequested = exc_info.value  # type: ignore[assignment]
    checkpoint = exception.checkpoint
    assert checkpoint.metadata is not None
    assert "checkpoint_id" in checkpoint.metadata
    assert "run_id" in checkpoint.metadata

    # Verify checkpoint in store
    query = CheckpointQuery(run_id=RunId(run_id))
    checkpoints, _ = await store.list(query)
    assert len(checkpoints) == 1
    assert checkpoints[0].run_id == RunId(run_id)


@pytest.mark.asyncio
async def test_checkpoint_not_persisted_without_store():
    """Test that checkpoints are not persisted when no store configured."""
    # Create flow
    flow = Flow(input_type=SimpleInput, output_type=SimpleOutput)
    node = PromptNode[SimpleInput, SimpleOutput](
        name="processor", prompt="Process: {value}"
    )
    flow.add_nodes(node)

    # No checkpoint store configured
    config = RunConfig(run_id="test_run_002")

    # Register interrupt handler
    async def always_interrupt(item: ProgressItem) -> InterruptDecision:
        return InterruptDecision(should_interrupt=True, reason="Test interrupt")

    flow.register_interrupt_handler(callback=always_interrupt, priority=0)

    # Run flow - should be interrupted
    with pytest.raises(InterruptionRequested) as exc_info:
        await flow.run(SimpleInput(value="test"), config=config)

    # Verify no checkpoint_id in metadata (not persisted)
    exception: InterruptionRequested = exc_info.value  # type: ignore[assignment]
    checkpoint = exception.checkpoint
    assert checkpoint.metadata is None or "checkpoint_id" not in checkpoint.metadata


@pytest.mark.asyncio
async def test_run_id_generated_if_not_provided():
    """Test that run_id is auto-generated if not provided in config."""
    # Create flow
    flow = Flow(input_type=SimpleInput, output_type=SimpleOutput)
    node = PromptNode[SimpleInput, SimpleOutput](
        name="processor", prompt="Process: {value}"
    )
    flow.add_nodes(node)

    # Create checkpoint store but no run_id
    store = InMemoryCheckpointStore()
    config = RunConfig(checkpoint_store=store)  # run_id not provided

    # Register interrupt handler
    async def always_interrupt(item: ProgressItem) -> InterruptDecision:
        return InterruptDecision(should_interrupt=True, reason="Test interrupt")

    flow.register_interrupt_handler(callback=always_interrupt, priority=0)

    # Run flow - should be interrupted
    with pytest.raises(InterruptionRequested) as exc_info:
        await flow.run(SimpleInput(value="test"), config=config)

    # Verify run_id was generated and checkpoint persisted
    exception: InterruptionRequested = exc_info.value  # type: ignore[assignment]
    checkpoint = exception.checkpoint
    assert checkpoint.metadata is not None
    assert "run_id" in checkpoint.metadata
    run_id = checkpoint.metadata["run_id"]
    assert run_id is not None
    assert len(run_id) > 0  # Generated UUID

    # Verify checkpoint in store
    query = CheckpointQuery(run_id=RunId(run_id))
    checkpoints, _ = await store.list(query)
    assert len(checkpoints) == 1


@pytest.mark.asyncio
async def test_multiple_checkpoints_same_run():
    """Test that multiple interruptions create multiple checkpoints."""
    # This would require resuming and interrupting again
    # For now, just test that multiple flows with same run_id accumulate checkpoints
    store = InMemoryCheckpointStore()
    run_id = "test_run_multi"

    for i in range(3):
        flow = Flow(input_type=SimpleInput, output_type=SimpleOutput)
        node = PromptNode[SimpleInput, SimpleOutput](
            name=f"processor_{i}", prompt="Process: {value}"
        )
        flow.add_nodes(node)

        config = RunConfig(checkpoint_store=store, run_id=run_id)

        async def always_interrupt(item: ProgressItem) -> InterruptDecision:
            return InterruptDecision(should_interrupt=True, reason="Test")

        flow.register_interrupt_handler(callback=always_interrupt, priority=0)

        with pytest.raises(InterruptionRequested):
            await flow.run(SimpleInput(value=f"test_{i}"), config=config)

    # Verify all checkpoints were saved
    query = CheckpointQuery(run_id=RunId(run_id))
    checkpoints, _ = await store.list(query)
    assert len(checkpoints) == 3
