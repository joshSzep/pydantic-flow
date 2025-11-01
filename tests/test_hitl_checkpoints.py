"""Tests for HITL checkpoint integration with V2 system."""

from pathlib import Path

from pydantic import BaseModel
import pytest

from pydantic_flow import Flow
from pydantic_flow import HandlerPriority
from pydantic_flow import InterruptDecision
from pydantic_flow import PromptNode
from pydantic_flow.checkpoints import CheckpointInspector
from pydantic_flow.checkpoints import SnapshotReason
from pydantic_flow.checkpoints import SQLiteCheckpointBackend
from pydantic_flow.checkpoints import SQLiteCheckpointConfig
from pydantic_flow.checkpoints.types import RunId
from pydantic_flow.core.run_config import RunConfig
from pydantic_flow.hitl.interrupts import InterruptionRequested
from pydantic_flow.streaming.base import ProgressItem
from pydantic_flow.streaming.core_events import StreamEnd


class SimpleInput(BaseModel):
    """Test input model."""

    text: str


class SimpleOutput(BaseModel):
    """Test output model."""

    result: str


@pytest.mark.asyncio
async def test_interrupted_checkpoint_saved():
    """Test that interrupted flows create V2 checkpoints with HITL_INTERRUPT reason."""
    config = SQLiteCheckpointConfig(db_path=Path(":memory:"))
    backend = SQLiteCheckpointBackend(config)
    await backend.initialize()

    try:
        inspector = CheckpointInspector(backend)
        flow = Flow(input_type=SimpleInput, output_type=SimpleOutput)
        node = PromptNode[SimpleInput, SimpleOutput](
            name="processor",
            prompt="Process: {text}",
            output_type=SimpleOutput,
        )
        flow.add_nodes(node)

        run_id = "test_interrupt_001"
        run_config = RunConfig(checkpoint_backend=backend, run_id=run_id)

        async def interrupt_handler(item: ProgressItem) -> InterruptDecision:
            if isinstance(item, StreamEnd):
                return InterruptDecision.interrupt(
                    reason="Test interrupt on completion"
                )
            return InterruptDecision.proceed()

        flow.register_interrupt_handler(
            callback=interrupt_handler, priority=HandlerPriority.HIGH
        )

        with pytest.raises(InterruptionRequested) as exc_info:
            await flow.run(SimpleInput(text="test data"), run_config)

        exception = exc_info.value
        assert isinstance(exception, InterruptionRequested)
        assert exception.snapshot is not None
        assert exception.snapshot.reason == SnapshotReason.HITL_INTERRUPT
        assert exception.snapshot.run_id == run_id

        runs = await inspector.list_interrupted_runs()
        assert len(runs) == 1
        assert runs[0].run_id == run_id
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_checkpoint_has_conversation_head():
    """Test that interrupted checkpoints can reference conversation messages."""
    config = SQLiteCheckpointConfig(db_path=Path(":memory:"))
    backend = SQLiteCheckpointBackend(config)
    await backend.initialize()

    try:
        inspector = CheckpointInspector(backend)
        flow = Flow(input_type=SimpleInput, output_type=SimpleOutput)
        node = PromptNode[SimpleInput, SimpleOutput](
            name="processor",
            prompt="Process: {text}",
            output_type=SimpleOutput,
        )
        flow.add_nodes(node)

        run_id = RunId("test_conversation_001")
        run_config = RunConfig(checkpoint_backend=backend, run_id=run_id)

        async def interrupt_handler(item: ProgressItem) -> InterruptDecision:
            if isinstance(item, StreamEnd):
                return InterruptDecision.interrupt(
                    reason="Test interrupt for conversation"
                )
            return InterruptDecision.proceed()

        flow.register_interrupt_handler(callback=interrupt_handler, priority=0)

        with pytest.raises(InterruptionRequested):
            await flow.run(SimpleInput(text="test"), run_config)

        runs = await inspector.list_interrupted_runs()
        assert len(runs) == 1

        snapshot = await inspector.get_interrupt_snapshot(run_id)
        assert snapshot is not None

        # Conversation may be empty if ConversationMemory is not configured
        # Just verify the API works without errors
        conversation = await inspector.get_conversation_at_interrupt(run_id)
        assert isinstance(conversation, list)
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_checkpoint_inspection_apis():
    """Test that checkpoint inspection APIs work correctly."""
    config = SQLiteCheckpointConfig(db_path=Path(":memory:"))
    backend = SQLiteCheckpointBackend(config)
    await backend.initialize()

    try:
        inspector = CheckpointInspector(backend)
        flow = Flow(input_type=SimpleInput, output_type=SimpleOutput)
        node = PromptNode[SimpleInput, SimpleOutput](
            name="processor",
            prompt="Process: {text}",
            output_type=SimpleOutput,
        )
        flow.add_nodes(node)

        run_id = RunId("test_inspect_001")
        run_config = RunConfig(checkpoint_backend=backend, run_id=run_id)

        async def interrupt_handler(item: ProgressItem) -> InterruptDecision:
            if isinstance(item, StreamEnd):
                return InterruptDecision.interrupt(
                    reason="Test interrupt for inspection"
                )
            return InterruptDecision.proceed()

        flow.register_interrupt_handler(callback=interrupt_handler, priority=0)

        with pytest.raises(InterruptionRequested):
            await flow.run(SimpleInput(text="test"), run_config)

        interrupted_runs = await inspector.list_interrupted_runs()
        assert len(interrupted_runs) == 1
        assert interrupted_runs[0].run_id == run_id

        snapshot = await inspector.get_interrupt_snapshot(run_id)
        assert snapshot is not None
        assert snapshot.reason == SnapshotReason.HITL_INTERRUPT

        conversation = await inspector.get_conversation_at_interrupt(run_id)
        assert isinstance(conversation, list)
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_multiple_runs_separate_interrupts():
    """Test that multiple runs create separate interrupted checkpoints."""
    config = SQLiteCheckpointConfig(db_path=Path(":memory:"))
    backend = SQLiteCheckpointBackend(config)
    await backend.initialize()

    try:
        inspector = CheckpointInspector(backend)
        flow = Flow(input_type=SimpleInput, output_type=SimpleOutput)
        node = PromptNode[SimpleInput, SimpleOutput](
            name="processor",
            prompt="Process: {text}",
            output_type=SimpleOutput,
        )
        flow.add_nodes(node)

        async def interrupt_handler(item: ProgressItem) -> InterruptDecision:
            if isinstance(item, StreamEnd):
                return InterruptDecision.interrupt(
                    reason="Test interrupt for multiple runs"
                )
            return InterruptDecision.proceed()

        flow.register_interrupt_handler(callback=interrupt_handler, priority=0)

        for i in range(3):
            run_id = RunId(f"test_run_{i:03d}")
            run_config = RunConfig(checkpoint_backend=backend, run_id=run_id)

            with pytest.raises(InterruptionRequested):
                await flow.run(SimpleInput(text=f"test_{i}"), run_config)

        interrupted_runs = await inspector.list_interrupted_runs()
        assert len(interrupted_runs) == 3

        run_ids = {run.run_id for run in interrupted_runs}
        assert run_ids == {
            RunId("test_run_000"),
            RunId("test_run_001"),
            RunId("test_run_002"),
        }
    finally:
        await backend.close()

    await backend.close()
