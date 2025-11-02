"""Integration tests for checkpoint persistence with Flow execution."""

from pathlib import Path

from pydantic import BaseModel
import pytest

from pydantic_flow import Flow
from pydantic_flow import PromptNode
from pydantic_flow.checkpoints import CheckpointInspector
from pydantic_flow.checkpoints import SnapshotReason
from pydantic_flow.checkpoints import SQLiteCheckpointBackend
from pydantic_flow.checkpoints import SQLiteCheckpointConfig
from pydantic_flow.checkpoints.types import RunId
from pydantic_flow.core.run_config import RunConfig
from pydantic_flow.hitl.decisions import InterruptDecision
from pydantic_flow.hitl.interrupts import InterruptionRequested
from pydantic_flow.streaming.base import ProgressItem
from pydantic_flow.streaming.core_events import StreamEnd
from tests.conftest import extract_result_from_stream


class SimpleInput(BaseModel):
    """Test input model."""

    value: str


class ProcessorOutput(BaseModel):
    """Processor result."""

    result: str


class SimpleOutput(BaseModel):
    """Test output model."""

    processor: ProcessorOutput


@pytest.mark.skip(reason="Requires LLM execution for PromptNode interrupts")
@pytest.mark.asyncio
async def test_checkpoint_persisted_on_interruption():
    """Test that checkpoints are persisted when flow is interrupted."""
    config = SQLiteCheckpointConfig(db_path=Path(":memory:"))
    backend = SQLiteCheckpointBackend(config)
    await backend.initialize()

    try:
        inspector = CheckpointInspector(backend)

        # Create flow
        flow = Flow(input_type=SimpleInput, output_type=SimpleOutput)
        node = PromptNode[SimpleInput, ProcessorOutput](
            name="processor",
            prompt="Process: {value}",
            output_type=ProcessorOutput,
        )
        flow.add_nodes(node)

        run_id = RunId("test_run_001")
        run_config = RunConfig(checkpoint_backend=backend, run_id=run_id)

        # Register interrupt handler
        async def always_interrupt(item: ProgressItem) -> InterruptDecision:
            if isinstance(item, StreamEnd):
                return InterruptDecision.interrupt(reason="Test interrupt")
            return InterruptDecision.proceed()

        flow.register_interrupt_handler(callback=always_interrupt, priority=0)

        # Run flow - should be interrupted
        with pytest.raises(InterruptionRequested) as exc_info:
            await extract_result_from_stream(
                flow.astream(SimpleInput(value="test"), run_config)
            )

        # Verify snapshot was created
        exception = exc_info.value
        assert isinstance(exception, InterruptionRequested)
        assert exception.snapshot is not None
        assert exception.snapshot.reason == SnapshotReason.HITL_INTERRUPT
        assert exception.snapshot.run_id == run_id

        # Verify checkpoint in storage
        interrupted_runs = await inspector.list_interrupted_runs()
        assert len(interrupted_runs) == 1
        assert interrupted_runs[0].run_id == run_id

        snapshot = await inspector.get_interrupt_snapshot(run_id)
        assert snapshot is not None
        assert snapshot.reason == SnapshotReason.HITL_INTERRUPT
    finally:
        await backend.close()


@pytest.mark.skip(reason="Requires LLM execution for PromptNode interrupts")
@pytest.mark.asyncio
async def test_checkpoint_not_persisted_without_backend():
    """Test that flow works without checkpoint backend configured."""
    # Create flow without checkpoint backend
    flow = Flow(input_type=SimpleInput, output_type=SimpleOutput)
    node = PromptNode[SimpleInput, ProcessorOutput](
        name="processor",
        prompt="Process: {value}",
        output_type=ProcessorOutput,
    )
    flow.add_nodes(node)

    # No checkpoint backend configured
    config = RunConfig(run_id="test_run_002")

    # Register interrupt handler
    async def always_interrupt(item: ProgressItem) -> InterruptDecision:
        if isinstance(item, StreamEnd):
            return InterruptDecision.interrupt(reason="Test interrupt")
        return InterruptDecision.proceed()

    flow.register_interrupt_handler(callback=always_interrupt, priority=0)

    # Run flow - should be interrupted but with V1 exception
    # (since no V2 backend is configured)
    from pydantic_flow.hitl.interrupts import InterruptionRequested

    with pytest.raises(InterruptionRequested):
        await extract_result_from_stream(
            flow.astream(SimpleInput(value="test"), config=config)
        )


@pytest.mark.skip(reason="Requires LLM execution for PromptNode interrupts")
@pytest.mark.asyncio
async def test_run_id_generated_if_not_provided():
    """Test that run_id is auto-generated if not provided in config."""
    config = SQLiteCheckpointConfig(db_path=Path(":memory:"))
    backend = SQLiteCheckpointBackend(config)
    await backend.initialize()

    try:
        inspector = CheckpointInspector(backend)

        # Create flow
        flow = Flow(input_type=SimpleInput, output_type=SimpleOutput)
        node = PromptNode[SimpleInput, ProcessorOutput](
            name="processor",
            prompt="Process: {value}",
            output_type=ProcessorOutput,
        )
        flow.add_nodes(node)

        # Checkpoint backend but no run_id (will be auto-generated)
        run_config = RunConfig(checkpoint_backend=backend)

        # Register interrupt handler
        async def always_interrupt(item: ProgressItem) -> InterruptDecision:
            if isinstance(item, StreamEnd):
                return InterruptDecision.interrupt(reason="Test interrupt")
            return InterruptDecision.proceed()

        flow.register_interrupt_handler(callback=always_interrupt, priority=0)

        # Run flow - should be interrupted
        with pytest.raises(InterruptionRequested) as exc_info:
            await extract_result_from_stream(
                flow.astream(SimpleInput(value="test"), run_config)
            )

        # Verify run_id was generated
        exception = exc_info.value
        assert isinstance(exception, InterruptionRequested)
        assert exception.snapshot is not None
        generated_run_id = exception.snapshot.run_id
        assert generated_run_id is not None
        assert len(str(generated_run_id)) > 0

        # Verify checkpoint in storage
        interrupted_runs = await inspector.list_interrupted_runs()
        assert len(interrupted_runs) == 1
        assert interrupted_runs[0].run_id == generated_run_id
    finally:
        await backend.close()


@pytest.mark.skip(reason="Requires LLM execution for PromptNode interrupts")
@pytest.mark.asyncio
async def test_multiple_interrupts_same_backend():
    """Test that multiple flows with same backend create separate runs."""
    config = SQLiteCheckpointConfig(db_path=Path(":memory:"))
    backend = SQLiteCheckpointBackend(config)
    await backend.initialize()

    try:
        inspector = CheckpointInspector(backend)

        run_ids = []
        for i in range(3):
            # Create output type with processor field matching the node name
            class ProcessorResult(BaseModel):
                result: str

            output_type = type(
                f"Output_{i}",
                (BaseModel,),
                {
                    f"processor_{i}": (ProcessorResult, ...),
                    "__annotations__": {f"processor_{i}": ProcessorResult},
                },
            )

            flow = Flow(input_type=SimpleInput, output_type=output_type)
            node = PromptNode[SimpleInput, ProcessorResult](
                name=f"processor_{i}",
                prompt="Process: {value}",
                output_type=ProcessorResult,
            )
            flow.add_nodes(node)

            run_id = RunId(f"test_run_multi_{i}")
            run_ids.append(run_id)
            run_config = RunConfig(checkpoint_backend=backend, run_id=run_id)

            async def always_interrupt(item: ProgressItem) -> InterruptDecision:
                if isinstance(item, StreamEnd):
                    return InterruptDecision.interrupt(reason="Test")
                return InterruptDecision.proceed()

            flow.register_interrupt_handler(callback=always_interrupt, priority=0)

            with pytest.raises(InterruptionRequested):
                await extract_result_from_stream(
                    flow.astream(SimpleInput(value=f"test_{i}"), run_config)
                )

        # Verify all runs were saved as interrupted
        interrupted_runs = await inspector.list_interrupted_runs()
        assert len(interrupted_runs) == 3

        # Verify each run has its snapshot
        for run_id in run_ids:
            snapshot = await inspector.get_interrupt_snapshot(run_id)
            assert snapshot is not None
            assert snapshot.reason == SnapshotReason.HITL_INTERRUPT
    finally:
        await backend.close()
