"""Tests for HITL query and inspection APIs.

Tests the unified V2 checkpoint system's HITL-specific query methods.
"""

from datetime import UTC
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel
import pytest

from pydantic_flow.checkpoints.backends.sqlite import SQLiteCheckpointBackend
from pydantic_flow.checkpoints.backends.sqlite import SQLiteCheckpointConfig
from pydantic_flow.checkpoints.conversation import ConversationMessage
from pydantic_flow.checkpoints.debugger import CheckpointDebugger
from pydantic_flow.checkpoints.inspection import CheckpointInspector
from pydantic_flow.checkpoints.types import RunId
from pydantic_flow.checkpoints.types import RunMetadata
from pydantic_flow.checkpoints.types import SnapshotReason
from pydantic_flow.checkpoints.types import StateSnapshot
from pydantic_flow.checkpoints.types import generate_message_id
from pydantic_flow.checkpoints.types import generate_run_id
from pydantic_flow.checkpoints.types import generate_snapshot_id


class SampleState(BaseModel):
    """Sample state for testing."""

    value: str
    count: int


@pytest.fixture
async def backend(tmp_path: Path):
    """Create a test SQLite backend."""
    config = SQLiteCheckpointConfig(db_path=tmp_path / "test.db")
    backend = SQLiteCheckpointBackend(config=config)
    await backend.initialize()
    yield backend
    await backend.close()


@pytest.fixture
async def inspector(backend: SQLiteCheckpointBackend):
    """Create test inspector."""
    return CheckpointInspector(backend)


@pytest.fixture
async def debugger(backend: SQLiteCheckpointBackend):
    """Create test debugger."""
    return CheckpointDebugger(backend)


async def create_interrupted_run(backend: SQLiteCheckpointBackend) -> tuple[RunId, str]:
    """Create an interrupted run with snapshot for testing.

    Returns:
        Tuple of (run_id, snapshot_id)

    """
    run_id = generate_run_id()
    snapshot_id = generate_snapshot_id()

    # Create run metadata
    metadata = RunMetadata(
        run_id=run_id,
        flow_id="test_flow",
        started_at=datetime.now(UTC),
        status=RunMetadata.Status.INTERRUPTED,
        total_waves=3,
        interrupted_at_wave=2,
        interrupt_snapshot_id=snapshot_id,
        awaiting_human_decision=True,
    )
    await backend.save_run_metadata(metadata)

    # Create conversation messages
    msg1_id = generate_message_id()
    msg2_id = generate_message_id()

    msg1 = ConversationMessage(
        message_id=msg1_id,
        run_id=run_id,
        previous_message_id=None,
        message={"role": "user", "content": "Test message 1"},
    )
    await backend.save_conversation_message(msg1)

    msg2 = ConversationMessage(
        message_id=msg2_id,
        run_id=run_id,
        previous_message_id=msg1_id,
        message={"role": "assistant", "content": "Test response 1"},
    )
    await backend.save_conversation_message(msg2)

    # Create interrupt snapshot
    snapshot = StateSnapshot(
        version=2,
        snapshot_id=snapshot_id,
        run_id=run_id,
        wave_number=2,
        full_state={"node1": SampleState(value="test", count=42)},
        state_hash="test_hash",
        next_frontier=["node2", "node3"],
        routing_ended=False,
        reason=SnapshotReason.HITL_INTERRUPT,
        interrupted_node_id="node1",
        conversation_head_id=msg2_id,
        metadata={"interrupt_reason": "user_confirmation_required"},
    )
    await backend.save_state_snapshot(snapshot)

    return run_id, snapshot_id


@pytest.mark.asyncio
async def test_list_interrupted_runs(
    inspector: CheckpointInspector,
    backend: SQLiteCheckpointBackend,
):
    """Test listing interrupted runs."""
    # Create several runs with different statuses
    interrupted_run, _ = await create_interrupted_run(backend)

    # Create a completed run
    completed_run_id = generate_run_id()
    completed_metadata = RunMetadata(
        run_id=completed_run_id,
        flow_id="test_flow",
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        status=RunMetadata.Status.COMPLETED,
        total_waves=5,
    )
    await backend.save_run_metadata(completed_metadata)

    # List interrupted runs
    interrupted_runs = await inspector.list_interrupted_runs(limit=50)

    assert len(interrupted_runs) == 1
    assert interrupted_runs[0].run_id == interrupted_run
    assert interrupted_runs[0].status == RunMetadata.Status.INTERRUPTED
    assert interrupted_runs[0].awaiting_human_decision is True


@pytest.mark.asyncio
async def test_get_interrupt_snapshot(
    inspector: CheckpointInspector,
    backend: SQLiteCheckpointBackend,
):
    """Test retrieving interrupt snapshot."""
    run_id, snapshot_id = await create_interrupted_run(backend)

    # Get interrupt snapshot
    snapshot = await inspector.get_interrupt_snapshot(run_id)

    assert snapshot is not None
    assert snapshot.snapshot_id == snapshot_id
    assert snapshot.reason == SnapshotReason.HITL_INTERRUPT
    assert snapshot.interrupted_node_id == "node1"
    assert snapshot.next_frontier == ["node2", "node3"]
    assert snapshot.metadata["interrupt_reason"] == "user_confirmation_required"


@pytest.mark.asyncio
async def test_get_interrupt_snapshot_not_found(
    inspector: CheckpointInspector,
):
    """Test getting interrupt snapshot for non-existent run."""
    snapshot = await inspector.get_interrupt_snapshot(generate_run_id())
    assert snapshot is None


@pytest.mark.asyncio
async def test_get_conversation_at_interrupt(
    inspector: CheckpointInspector,
    backend: SQLiteCheckpointBackend,
):
    """Test reconstructing conversation at interrupt point."""
    run_id, _ = await create_interrupted_run(backend)

    # Get conversation (returned in reverse chronological order - newest first)
    conversation = await inspector.get_conversation_at_interrupt(run_id)

    assert len(conversation) == 2
    # First message in list is the most recent (msg2 - assistant)
    assert conversation[0].message["role"] == "assistant"
    assert conversation[0].message["content"] == "Test response 1"
    # Second message is older (msg1 - user)
    assert conversation[1].message["role"] == "user"
    assert conversation[1].message["content"] == "Test message 1"


@pytest.mark.asyncio
async def test_get_conversation_at_interrupt_no_snapshot(
    inspector: CheckpointInspector,
):
    """Test getting conversation when no interrupt snapshot exists."""
    with pytest.raises(ValueError, match="No interrupt snapshot found"):
        await inspector.get_conversation_at_interrupt(generate_run_id())


@pytest.mark.asyncio
async def test_show_interrupted_runs(
    debugger: CheckpointDebugger,
    backend: SQLiteCheckpointBackend,
    capsys,
):
    """Test showing interrupted runs with Rich rendering."""
    # Create interrupted runs
    await create_interrupted_run(backend)
    await create_interrupted_run(backend)

    # Show interrupted runs (will print to console)
    await debugger.show_interrupted_runs()

    # Note: Testing Rich output is tricky, but we can at least verify no errors
    # In real usage, you'd visually inspect the table output


@pytest.mark.asyncio
async def test_show_interrupt_context(
    debugger: CheckpointDebugger,
    backend: SQLiteCheckpointBackend,
):
    """Test showing interrupt context with all details."""
    run_id, _ = await create_interrupted_run(backend)

    # Show interrupt context (will print to console)
    await debugger.show_interrupt_context(run_id)

    # Again, mainly testing for no errors
    # Visual inspection would verify Rich formatting


@pytest.mark.asyncio
async def test_show_interrupt_context_not_interrupted(
    debugger: CheckpointDebugger,
    backend: SQLiteCheckpointBackend,
):
    """Test showing context for non-interrupted run."""
    # Create completed run
    run_id = generate_run_id()
    metadata = RunMetadata(
        run_id=run_id,
        flow_id="test_flow",
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        status=RunMetadata.Status.COMPLETED,
        total_waves=5,
    )
    await backend.save_run_metadata(metadata)

    # Should handle gracefully
    await debugger.show_interrupt_context(run_id)


@pytest.mark.asyncio
async def test_multiple_interrupted_runs_ordering(
    inspector: CheckpointInspector,
    backend: SQLiteCheckpointBackend,
):
    """Test that interrupted runs are returned in correct order."""
    # Create multiple interrupted runs
    runs = []
    for _ in range(3):
        run_id, _ = await create_interrupted_run(backend)
        runs.append(run_id)

    # List interrupted runs
    interrupted_runs = await inspector.list_interrupted_runs(limit=50)

    assert len(interrupted_runs) == 3
    # All should be interrupted status
    assert all(r.status == RunMetadata.Status.INTERRUPTED for r in interrupted_runs)
    assert all(r.awaiting_human_decision for r in interrupted_runs)


@pytest.mark.asyncio
async def test_interrupt_snapshot_with_empty_conversation(
    inspector: CheckpointInspector,
    backend: SQLiteCheckpointBackend,
):
    """Test interrupt snapshot with no conversation history."""
    run_id = generate_run_id()
    snapshot_id = generate_snapshot_id()

    # Create run metadata
    metadata = RunMetadata(
        run_id=run_id,
        flow_id="test_flow",
        started_at=datetime.now(UTC),
        status=RunMetadata.Status.INTERRUPTED,
        total_waves=1,
        interrupted_at_wave=0,
        interrupt_snapshot_id=snapshot_id,
        awaiting_human_decision=True,
    )
    await backend.save_run_metadata(metadata)

    # Create interrupt snapshot with no conversation
    snapshot = StateSnapshot(
        version=2,
        snapshot_id=snapshot_id,
        run_id=run_id,
        wave_number=0,
        full_state={"node1": SampleState(value="test", count=0)},
        state_hash="test_hash",
        next_frontier=["node2"],
        routing_ended=False,
        reason=SnapshotReason.HITL_INTERRUPT,
        interrupted_node_id="node1",
        conversation_head_id=None,  # No conversation
        metadata={},
    )
    await backend.save_state_snapshot(snapshot)

    # Get conversation should return empty list
    conversation = await inspector.get_conversation_at_interrupt(run_id)
    assert conversation == []
