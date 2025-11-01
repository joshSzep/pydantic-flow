"""Shared fixtures for checkpoint tests."""

import pytest

# V1 checkpoint imports commented out - V1 system deprecated
# from pydantic_flow.hitl.checkpoints.interface import CheckpointEnvelope
# from pydantic_flow.hitl.checkpoints.interface import CheckpointId
# from pydantic_flow.hitl.checkpoints.interface import RunId
# from pydantic_flow.hitl.interrupts import FlowCheckpoint


@pytest.fixture(autouse=True)
def _disable_telemetry_exports(monkeypatch: pytest.MonkeyPatch):
    """Disable telemetry OTLP exports by default in tests.

    Tests should not attempt to connect to real OTLP endpoints.
    Individual tests can override this by setting the env var explicitly.
    """
    # Clear any OTLP endpoint env var that might cause connection attempts
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    # Disable telemetry by default - tests that need it can enable explicitly
    monkeypatch.setenv("PFLOW_TELEMETRY_ENABLED", "false")


# V1 checkpoint fixtures commented out - these tests need migration to V2
# @pytest.fixture
# def sample_checkpoint() -> FlowCheckpoint:
#     """Create a sample checkpoint for testing."""
#     return FlowCheckpoint(
#         flow_id="test_flow",
#         run_id="test_run_123",
#         interrupted_node_id="node_1",
#         node_states={"node_1": {"value": 42}},
#         edge_history=[("start", "node_1")],
#         metadata={"test": "data"},
#     )
#
#
# @pytest.fixture
# def sample_envelope(sample_checkpoint: FlowCheckpoint) -> CheckpointEnvelope:
#     """Create a sample checkpoint envelope for testing."""
#     return CheckpointEnvelope(
#         id=CheckpointId("checkpoint_001"),
#         run_id=RunId("test_run_123"),
#         node_id="node_1",
#         checkpoint=sample_checkpoint,
#     )
