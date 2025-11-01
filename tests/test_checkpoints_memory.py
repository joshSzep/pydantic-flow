"""Tests for in-memory checkpoint backend (V2)."""

from pathlib import Path

import pytest

from pydantic_flow.checkpoints import CheckpointInspector
from pydantic_flow.checkpoints import RunId
from pydantic_flow.checkpoints import SnapshotReason
from pydantic_flow.checkpoints import SQLiteCheckpointBackend
from pydantic_flow.checkpoints import SQLiteCheckpointConfig
from pydantic_flow.checkpoints import StateSnapshot
from pydantic_flow.checkpoints.types import generate_snapshot_id


@pytest.mark.asyncio
async def test_memory_backend_is_empty_initially():
    """Test that a new in-memory backend is empty."""
    config = SQLiteCheckpointConfig(db_path=Path(":memory:"))
    backend = SQLiteCheckpointBackend(config)
    await backend.initialize()
    inspector = CheckpointInspector(backend)

    runs = await inspector.list_interrupted_runs()
    assert len(runs) == 0

    await backend.close()


@pytest.mark.asyncio
async def test_memory_backend_isolation():
    """Test that different in-memory backend instances don't share data."""
    config1 = SQLiteCheckpointConfig(db_path=Path(":memory:"))
    config2 = SQLiteCheckpointConfig(db_path=Path(":memory:"))
    backend1 = SQLiteCheckpointBackend(config1)
    backend2 = SQLiteCheckpointBackend(config2)
    await backend1.initialize()
    await backend2.initialize()

    snapshot = StateSnapshot(
        snapshot_id=generate_snapshot_id(),
        run_id=RunId("run1"),
        wave_number=0,
        full_state={},
        state_hash="test_hash",
        next_frontier=[],
        routing_ended=False,
        reason=SnapshotReason.HITL_INTERRUPT,
    )
    await backend1.save_state_snapshot(snapshot)

    inspector2 = CheckpointInspector(backend2)
    runs = await inspector2.list_interrupted_runs()
    assert len(runs) == 0

    await backend1.close()
    await backend2.close()
