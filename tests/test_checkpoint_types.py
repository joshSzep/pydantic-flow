"""Tests for checkpoint types and state snapshots."""

from __future__ import annotations

from pydantic import BaseModel

from pydantic_flow.checkpoints.types import DELETED_KEY
from pydantic_flow.checkpoints.types import CheckpointId
from pydantic_flow.checkpoints.types import RunId
from pydantic_flow.checkpoints.types import SnapshotId
from pydantic_flow.checkpoints.types import StateSnapshot
from pydantic_flow.checkpoints.types import generate_checkpoint_id
from pydantic_flow.checkpoints.types import generate_run_id
from pydantic_flow.checkpoints.types import generate_snapshot_id


class SampleState(BaseModel):
    """Test state model."""

    value: int
    name: str


def test_generate_checkpoint_id():
    """Test checkpoint ID generation."""
    id1 = generate_checkpoint_id()
    id2 = generate_checkpoint_id()

    assert isinstance(id1, CheckpointId)
    assert isinstance(id2, CheckpointId)
    assert id1 != id2


def test_generate_run_id():
    """Test run ID generation."""
    id1 = generate_run_id()
    id2 = generate_run_id()

    assert isinstance(id1, RunId)
    assert isinstance(id2, RunId)
    assert id1 != id2


def test_generate_snapshot_id():
    """Test snapshot ID generation."""
    id1 = generate_snapshot_id()
    id2 = generate_snapshot_id()

    assert isinstance(id1, SnapshotId)
    assert isinstance(id2, SnapshotId)
    assert id1 != id2


def test_state_snapshot_creation():
    """Test StateSnapshot creation."""
    snapshot = StateSnapshot(
        snapshot_id=generate_snapshot_id(),
        run_id=generate_run_id(),
        wave_number=0,
        full_state={
            "node1": SampleState(value=1, name="test1"),
            "node2": SampleState(value=2, name="test2"),
        },
        state_hash="abc123",
        next_frontier=["node3"],
        routing_ended=False,
    )

    assert snapshot.wave_number == 0
    assert snapshot.full_state is not None
    assert len(snapshot.full_state) == 2
    assert "node1" in snapshot.full_state
    assert snapshot.next_frontier == ["node3"]
    assert not snapshot.routing_ended


def test_state_snapshot_serialization():
    """Test StateSnapshot serialization/deserialization."""
    snapshot = StateSnapshot(
        snapshot_id=generate_snapshot_id(),
        run_id=generate_run_id(),
        wave_number=5,
        full_state={
            "node1": SampleState(value=42, name="answer"),
        },
        state_hash="hash123",
        next_frontier=["node2", "node3"],
        routing_ended=False,
    )

    data = snapshot.serialize()
    assert isinstance(data, bytes)

    restored = StateSnapshot.deserialize(data)
    assert restored.wave_number == 5
    assert restored.full_state is not None
    assert "node1" in restored.full_state
    # Type is preserved via msgpack extension types
    assert isinstance(restored.full_state["node1"], SampleState)
    assert restored.full_state["node1"].value == 42  # type: ignore[attr-defined]
    assert restored.next_frontier == ["node2", "node3"]


def test_state_snapshot_with_deltas():
    """Test StateSnapshot with delta fields."""
    snapshot = StateSnapshot(
        snapshot_id=generate_snapshot_id(),
        run_id=generate_run_id(),
        wave_number=3,
        forward_delta={
            "node1": SampleState(value=10, name="updated"),
        },
        reverse_delta={
            "node1": SampleState(value=5, name="original"),
        },
        full_state=None,
        state_hash="delta_hash",
        next_frontier=["node2"],
        routing_ended=False,
    )

    assert snapshot.forward_delta is not None
    assert snapshot.reverse_delta is not None
    assert snapshot.full_state is None


def test_state_hash_computation():
    """Test state hash computation."""
    snapshot = StateSnapshot(
        snapshot_id=generate_snapshot_id(),
        run_id=generate_run_id(),
        wave_number=0,
        full_state={},
        state_hash="placeholder",
        next_frontier=[],
        routing_ended=False,
    )

    state: dict[str, BaseModel] = {
        "node1": SampleState(value=1, name="test"),
        "node2": SampleState(value=2, name="test2"),
    }

    hash1 = snapshot.compute_state_hash(state)
    hash2 = snapshot.compute_state_hash(state)

    assert hash1 == hash2
    assert isinstance(hash1, str)
    assert len(hash1) == 64


def test_deleted_key_sentinel():
    """Test DeletedKey sentinel."""
    assert repr(DELETED_KEY) == "<DeletedKey>"
