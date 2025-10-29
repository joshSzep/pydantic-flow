"""Shared fixtures for checkpoint tests."""

import pytest

from pydantic_flow.checkpoints.interface import CheckpointEnvelope
from pydantic_flow.checkpoints.interface import CheckpointId
from pydantic_flow.checkpoints.interface import RunId
from pydantic_flow.core.errors import FlowCheckpoint


@pytest.fixture
def sample_checkpoint() -> FlowCheckpoint:
    """Create a sample checkpoint for testing."""
    return FlowCheckpoint(
        flow_id="test_flow",
        run_id="test_run_123",
        interrupted_node_id="node_1",
        node_states={"node_1": {"value": 42}},
        edge_history=[("start", "node_1")],
        metadata={"test": "data"},
    )


@pytest.fixture
def sample_envelope(sample_checkpoint: FlowCheckpoint) -> CheckpointEnvelope:
    """Create a sample checkpoint envelope for testing."""
    return CheckpointEnvelope(
        id=CheckpointId("checkpoint_001"),
        run_id=RunId("test_run_123"),
        node_id="node_1",
        checkpoint=sample_checkpoint,
    )
