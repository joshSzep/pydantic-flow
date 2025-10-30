"""Tests for InMemoryCheckpointStore implementation."""

import pytest

from pydantic_flow.hitl.checkpoints.memory import InMemoryCheckpointStore
from tests.test_checkpoints_conformance import CheckpointStoreConformanceTests


class TestInMemoryCheckpointStore(CheckpointStoreConformanceTests):
    """Run conformance tests against InMemoryCheckpointStore."""

    @pytest.fixture
    def store(self):
        """Provide InMemoryCheckpointStore instance."""
        return InMemoryCheckpointStore()

    @pytest.mark.asyncio
    async def test_store_is_empty_initially(self):
        """Test that a new store is empty."""
        from pydantic_flow.hitl.checkpoints.interface import CheckpointQuery
        from pydantic_flow.hitl.checkpoints.interface import RunId

        store = InMemoryCheckpointStore()

        query = CheckpointQuery(run_id=RunId("any_run"))
        results, cursor = await store.list(query)
        assert len(results) == 0
        assert cursor is None

    @pytest.mark.asyncio
    async def test_store_isolation(self):
        """Test that different store instances don't share data."""
        from pydantic_flow.hitl.checkpoints.interface import CheckpointEnvelope
        from pydantic_flow.hitl.checkpoints.interface import CheckpointId
        from pydantic_flow.hitl.checkpoints.interface import CheckpointQuery
        from pydantic_flow.hitl.checkpoints.interface import RunId
        from pydantic_flow.hitl.interrupts import FlowCheckpoint

        store1 = InMemoryCheckpointStore()
        store2 = InMemoryCheckpointStore()

        # Save to store1
        checkpoint = FlowCheckpoint(
            flow_id="test",
            run_id="run1",
            interrupted_node_id="node1",
            node_states={},
            edge_history=[],
        )
        envelope = CheckpointEnvelope(
            id=CheckpointId("cp1"), run_id=RunId("run1"), checkpoint=checkpoint
        )
        await store1.save(envelope)

        # Verify store2 is empty
        query = CheckpointQuery(run_id=RunId("run1"))
        results, _ = await store2.list(query)
        assert len(results) == 0
