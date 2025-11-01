"""Tests for delta computation."""

from __future__ import annotations

from pydantic import BaseModel

from pydantic_flow.checkpoints.delta import DeltaComputer
from pydantic_flow.checkpoints.types import DELETED_KEY
from pydantic_flow.checkpoints.types import DeletedKey


class SampleState(BaseModel):
    """Test state model."""

    value: int
    name: str


def test_compute_forward_delta_empty():
    """Test forward delta with no changes."""
    state1 = {"node1": SampleState(value=1, name="test")}
    state2 = {"node1": SampleState(value=1, name="test")}

    delta = DeltaComputer.compute_forward_delta(state1, state2)
    assert len(delta) == 0


def test_compute_forward_delta_changed():
    """Test forward delta with changed values."""
    state1 = {"node1": SampleState(value=1, name="old")}
    state2 = {"node1": SampleState(value=2, name="new")}

    delta = DeltaComputer.compute_forward_delta(state1, state2)
    assert len(delta) == 1
    assert "node1" in delta
    node1 = delta["node1"]
    assert isinstance(node1, SampleState)
    assert node1.value == 2
    assert node1.name == "new"


def test_compute_forward_delta_added():
    """Test forward delta with added keys."""
    state1 = {"node1": SampleState(value=1, name="test")}
    state2 = {
        "node1": SampleState(value=1, name="test"),
        "node2": SampleState(value=2, name="new"),
    }

    delta = DeltaComputer.compute_forward_delta(state1, state2)
    assert len(delta) == 1
    assert "node2" in delta
    node2 = delta["node2"]
    assert isinstance(node2, SampleState)
    assert node2.value == 2


def test_compute_reverse_delta_unchanged():
    """Test reverse delta with no changes."""
    state1 = {"node1": SampleState(value=1, name="test")}
    state2 = {"node1": SampleState(value=1, name="test")}

    delta = DeltaComputer.compute_reverse_delta(state1, state2)
    assert len(delta) == 0


def test_compute_reverse_delta_changed():
    """Test reverse delta with changed values."""
    state1 = {"node1": SampleState(value=1, name="old")}
    state2 = {"node1": SampleState(value=2, name="new")}

    delta = DeltaComputer.compute_reverse_delta(state1, state2)
    assert len(delta) == 1
    assert "node1" in delta
    assert isinstance(delta["node1"], SampleState)
    assert delta["node1"].value == 1
    assert delta["node1"].name == "old"


def test_compute_reverse_delta_deleted():
    """Test reverse delta with deleted keys."""
    state1 = {"node1": SampleState(value=1, name="test")}
    state2 = {
        "node1": SampleState(value=1, name="test"),
        "node2": SampleState(value=2, name="new"),
    }

    delta = DeltaComputer.compute_reverse_delta(state1, state2)
    assert len(delta) == 1
    assert "node2" in delta
    assert isinstance(delta["node2"], DeletedKey)


def test_apply_forward_delta():
    """Test applying forward delta."""
    base_state = {"node1": SampleState(value=1, name="old")}
    delta = {"node1": SampleState(value=2, name="new")}

    result = DeltaComputer.apply_forward_delta(base_state, delta)
    node1 = result["node1"]
    assert isinstance(node1, SampleState)
    assert node1.value == 2
    assert node1.name == "new"


def test_apply_forward_delta_adds_keys():
    """Test applying forward delta adds new keys."""
    base_state = {"node1": SampleState(value=1, name="test")}
    delta = {"node2": SampleState(value=2, name="new")}

    result = DeltaComputer.apply_forward_delta(base_state, delta)
    assert len(result) == 2
    assert "node1" in result
    assert "node2" in result
    node2 = result["node2"]
    assert isinstance(node2, SampleState)
    assert node2.value == 2


def test_apply_reverse_delta():
    """Test applying reverse delta."""
    current_state = {"node1": SampleState(value=2, name="new")}
    delta: dict[str, BaseModel | DeletedKey] = {
        "node1": SampleState(value=1, name="old")
    }

    result = DeltaComputer.apply_reverse_delta(current_state, delta)
    node1 = result["node1"]
    assert isinstance(node1, SampleState)
    assert node1.value == 1
    assert node1.name == "old"


def test_apply_reverse_delta_deletes_keys():
    """Test applying reverse delta with DeletedKey."""
    current_state = {
        "node1": SampleState(value=1, name="test"),
        "node2": SampleState(value=2, name="new"),
    }
    delta: dict[str, BaseModel | DeletedKey] = {"node2": DELETED_KEY}

    result = DeltaComputer.apply_reverse_delta(current_state, delta)
    assert len(result) == 1
    assert "node1" in result
    assert "node2" not in result


def test_roundtrip_forward_reverse():
    """Test forward then reverse delta roundtrip."""
    state1 = {
        "node1": SampleState(value=1, name="old"),
        "node2": SampleState(value=2, name="test"),
    }
    state2 = {
        "node1": SampleState(value=10, name="new"),
        "node3": SampleState(value=3, name="added"),
    }

    forward_delta = DeltaComputer.compute_forward_delta(state1, state2)
    reverse_delta = DeltaComputer.compute_reverse_delta(state1, state2)

    forward_result = DeltaComputer.apply_forward_delta(state1, forward_delta)

    forward_node1 = forward_result["node1"]
    assert isinstance(forward_node1, SampleState)
    assert forward_node1.value == 10
    assert "node3" in forward_result

    reverse_result = DeltaComputer.apply_reverse_delta(forward_result, reverse_delta)

    reverse_node1 = reverse_result["node1"]
    assert isinstance(reverse_node1, SampleState)
    assert reverse_node1.value == 1
    assert reverse_node1.name == "old"
    assert "node2" in reverse_result
    assert "node3" not in reverse_result
