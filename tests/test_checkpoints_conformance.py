"""Conformance tests that all checkpoint stores must pass.

These tests define the contract that all CheckpointStore implementations
must satisfy. Each store backend should pass these tests.
"""

from datetime import UTC
from datetime import datetime
from datetime import timedelta

import pytest

from pydantic_flow.checkpoints.interface import CheckpointConflict
from pydantic_flow.checkpoints.interface import CheckpointEnvelope
from pydantic_flow.checkpoints.interface import CheckpointId
from pydantic_flow.checkpoints.interface import CheckpointQuery
from pydantic_flow.checkpoints.interface import RunId
from pydantic_flow.checkpoints.interface import SortOrder


class CheckpointStoreConformanceTests:
    """Base class for checkpoint store conformance tests.

    Subclass this and implement the store_fixture to run all conformance
    tests against your checkpoint store implementation.
    """

    @pytest.fixture
    def store(self):
        """Override this fixture to provide your store implementation."""
        raise NotImplementedError("Must implement store fixture")

    @pytest.mark.asyncio
    async def test_save_and_get_checkpoint(self, store, sample_envelope):
        """Test saving and retrieving a checkpoint."""
        # Save checkpoint
        saved = await store.save(sample_envelope)
        assert saved.id == sample_envelope.id
        assert saved.run_id == sample_envelope.run_id

        # Retrieve checkpoint
        retrieved = await store.get(
            run_id=sample_envelope.run_id, checkpoint_id=sample_envelope.id
        )
        assert retrieved is not None
        assert retrieved.id == sample_envelope.id
        assert retrieved.checkpoint.flow_id == "test_flow"
        assert retrieved.checkpoint.node_states == {"node_1": {"value": 42}}

    @pytest.mark.asyncio
    async def test_save_duplicate_without_overwrite_raises(
        self, store, sample_envelope
    ):
        """Test that saving duplicate checkpoint without overwrite raises error."""
        # Save first time
        await store.save(sample_envelope)

        # Try to save again without overwrite
        with pytest.raises(CheckpointConflict):
            await store.save(sample_envelope, overwrite=False)

    @pytest.mark.asyncio
    async def test_save_duplicate_with_overwrite_succeeds(self, store, sample_envelope):
        """Test that saving duplicate checkpoint with overwrite succeeds."""
        # Save first time
        await store.save(sample_envelope)

        # Update checkpoint data
        modified_envelope = sample_envelope.model_copy(deep=True)
        modified_envelope.checkpoint.metadata = {"updated": "value"}

        # Save again with overwrite
        saved = await store.save(modified_envelope, overwrite=True)
        assert saved.checkpoint.metadata == {"updated": "value"}

        # Verify it was updated
        retrieved = await store.get(
            run_id=sample_envelope.run_id, checkpoint_id=sample_envelope.id
        )
        assert retrieved is not None
        assert retrieved.checkpoint.metadata == {"updated": "value"}

    @pytest.mark.asyncio
    async def test_get_nonexistent_checkpoint_returns_none(self, store):
        """Test that getting a nonexistent checkpoint returns None."""
        result = await store.get(
            run_id=RunId("nonexistent_run"), checkpoint_id=CheckpointId("nonexistent")
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_latest_returns_most_recent_checkpoint(
        self, store, sample_checkpoint
    ):
        """Test that latest returns the most recent checkpoint for a run."""
        run_id = RunId("test_run_latest")

        # Use fixed base timestamp to avoid flakiness
        base_time = datetime.now(UTC)

        # Create multiple checkpoints with different timestamps
        envelope1 = CheckpointEnvelope(
            id=CheckpointId("checkpoint_001"),
            run_id=run_id,
            node_id="node_1",
            created_at=base_time - timedelta(minutes=10),
            checkpoint=sample_checkpoint,
        )

        envelope2 = CheckpointEnvelope(
            id=CheckpointId("checkpoint_002"),
            run_id=run_id,
            node_id="node_2",
            created_at=base_time - timedelta(minutes=5),
            checkpoint=sample_checkpoint,
        )

        envelope3 = CheckpointEnvelope(
            id=CheckpointId("checkpoint_003"),
            run_id=run_id,
            node_id="node_3",
            created_at=base_time,
            checkpoint=sample_checkpoint,
        )

        # Save in random order
        await store.save(envelope2)
        await store.save(envelope1)
        await store.save(envelope3)

        # Get latest
        latest = await store.latest(run_id=run_id)
        assert latest is not None
        assert latest.id == CheckpointId("checkpoint_003")

    @pytest.mark.asyncio
    async def test_latest_with_node_id_filter(self, store, sample_checkpoint):
        """Test that latest can filter by node_id."""
        run_id = RunId("test_run_node_filter")

        # Create checkpoints for different nodes
        envelope1 = CheckpointEnvelope(
            id=CheckpointId("checkpoint_001"),
            run_id=run_id,
            node_id="node_a",
            created_at=datetime.now(UTC) - timedelta(minutes=5),
            checkpoint=sample_checkpoint,
        )

        envelope2 = CheckpointEnvelope(
            id=CheckpointId("checkpoint_002"),
            run_id=run_id,
            node_id="node_b",
            created_at=datetime.now(UTC),
            checkpoint=sample_checkpoint,
        )

        await store.save(envelope1)
        await store.save(envelope2)

        # Get latest for specific node
        latest_a = await store.latest(run_id=run_id, node_id="node_a")
        assert latest_a is not None
        assert latest_a.id == CheckpointId("checkpoint_001")
        assert latest_a.node_id == "node_a"

    @pytest.mark.asyncio
    async def test_latest_returns_none_when_empty(self, store):
        """Test that latest returns None when no checkpoints exist."""
        result = await store.latest(run_id=RunId("nonexistent_run"))
        assert result is None

    @pytest.mark.asyncio
    async def test_list_with_run_id_filter(self, store, sample_checkpoint):
        """Test listing checkpoints filtered by run_id."""
        run_id_1 = RunId("test_run_1")
        run_id_2 = RunId("test_run_2")

        # Create checkpoints for different runs
        envelope1 = CheckpointEnvelope(
            id=CheckpointId("checkpoint_001"),
            run_id=run_id_1,
            checkpoint=sample_checkpoint,
        )

        envelope2 = CheckpointEnvelope(
            id=CheckpointId("checkpoint_002"),
            run_id=run_id_2,
            checkpoint=sample_checkpoint,
        )

        envelope3 = CheckpointEnvelope(
            id=CheckpointId("checkpoint_003"),
            run_id=run_id_1,
            checkpoint=sample_checkpoint,
        )

        await store.save(envelope1)
        await store.save(envelope2)
        await store.save(envelope3)

        # List checkpoints for run_id_1
        query = CheckpointQuery(run_id=run_id_1)
        results, _ = await store.list(query)

        assert len(results) == 2
        assert all(e.run_id == run_id_1 for e in results)

    @pytest.mark.asyncio
    async def test_list_with_limit(self, store, sample_checkpoint):
        """Test listing checkpoints with pagination limit."""
        run_id = RunId("test_run_pagination")

        # Create multiple checkpoints
        for i in range(5):
            envelope = CheckpointEnvelope(
                id=CheckpointId(f"checkpoint_{i:03d}"),
                run_id=run_id,
                checkpoint=sample_checkpoint,
            )
            await store.save(envelope)

        # List with limit
        query = CheckpointQuery(run_id=run_id, limit=3)
        results, cursor = await store.list(query)

        assert len(results) == 3
        assert cursor is not None  # Should have more results

    @pytest.mark.asyncio
    async def test_list_with_cursor_pagination(self, store, sample_checkpoint):
        """Test pagination using cursors."""
        run_id = RunId("test_run_cursor")

        # Create multiple checkpoints
        for i in range(5):
            envelope = CheckpointEnvelope(
                id=CheckpointId(f"checkpoint_{i:03d}"),
                run_id=run_id,
                checkpoint=sample_checkpoint,
            )
            await store.save(envelope)

        # Get first page
        query1 = CheckpointQuery(run_id=run_id, limit=2)
        page1, cursor1 = await store.list(query1)
        assert len(page1) == 2
        assert cursor1 is not None

        # Get second page
        query2 = CheckpointQuery(run_id=run_id, limit=2, cursor=cursor1)
        page2, _ = await store.list(query2)
        assert len(page2) == 2

        # Verify different results
        page1_ids = {e.id for e in page1}
        page2_ids = {e.id for e in page2}
        assert page1_ids.isdisjoint(page2_ids)

    @pytest.mark.asyncio
    async def test_list_sort_order_descending(self, store, sample_checkpoint):
        """Test listing checkpoints in descending order (most recent first)."""
        run_id = RunId("test_run_sort_desc")

        # Create checkpoints with specific timestamps
        envelope1 = CheckpointEnvelope(
            id=CheckpointId("checkpoint_001"),
            run_id=run_id,
            created_at=datetime.now(UTC) - timedelta(minutes=10),
            checkpoint=sample_checkpoint,
        )

        envelope2 = CheckpointEnvelope(
            id=CheckpointId("checkpoint_002"),
            run_id=run_id,
            created_at=datetime.now(UTC) - timedelta(minutes=5),
            checkpoint=sample_checkpoint,
        )

        await store.save(envelope1)
        await store.save(envelope2)

        # List in descending order (default)
        query = CheckpointQuery(run_id=run_id, sort_order=SortOrder.DESC)
        results, _ = await store.list(query)

        assert len(results) == 2
        assert results[0].id == CheckpointId("checkpoint_002")
        assert results[1].id == CheckpointId("checkpoint_001")

    @pytest.mark.asyncio
    async def test_list_sort_order_ascending(self, store, sample_checkpoint):
        """Test listing checkpoints in ascending order (oldest first)."""
        run_id = RunId("test_run_sort_asc")

        # Create checkpoints with specific timestamps
        envelope1 = CheckpointEnvelope(
            id=CheckpointId("checkpoint_001"),
            run_id=run_id,
            created_at=datetime.now(UTC) - timedelta(minutes=10),
            checkpoint=sample_checkpoint,
        )

        envelope2 = CheckpointEnvelope(
            id=CheckpointId("checkpoint_002"),
            run_id=run_id,
            created_at=datetime.now(UTC) - timedelta(minutes=5),
            checkpoint=sample_checkpoint,
        )

        await store.save(envelope1)
        await store.save(envelope2)

        # List in ascending order
        query = CheckpointQuery(run_id=run_id, sort_order=SortOrder.ASC)
        results, _ = await store.list(query)

        assert len(results) == 2
        assert results[0].id == CheckpointId("checkpoint_001")
        assert results[1].id == CheckpointId("checkpoint_002")

    @pytest.mark.asyncio
    async def test_list_with_time_range(self, store, sample_checkpoint):
        """Test listing checkpoints within a time range."""
        run_id = RunId("test_run_time_range")
        now = datetime.now(UTC)

        # Create checkpoints at different times
        envelope1 = CheckpointEnvelope(
            id=CheckpointId("checkpoint_001"),
            run_id=run_id,
            created_at=now - timedelta(hours=2),
            checkpoint=sample_checkpoint,
        )

        envelope2 = CheckpointEnvelope(
            id=CheckpointId("checkpoint_002"),
            run_id=run_id,
            created_at=now - timedelta(hours=1),
            checkpoint=sample_checkpoint,
        )

        envelope3 = CheckpointEnvelope(
            id=CheckpointId("checkpoint_003"),
            run_id=run_id,
            created_at=now,
            checkpoint=sample_checkpoint,
        )

        await store.save(envelope1)
        await store.save(envelope2)
        await store.save(envelope3)

        # Query for checkpoints in the last 90 minutes
        query = CheckpointQuery(
            run_id=run_id, since=now - timedelta(minutes=90), until=now
        )
        results, _ = await store.list(query)

        assert len(results) == 2
        assert CheckpointId("checkpoint_001") not in [e.id for e in results]

    @pytest.mark.asyncio
    async def test_delete_checkpoint(self, store, sample_envelope):
        """Test deleting a checkpoint."""
        # Save checkpoint
        await store.save(sample_envelope)

        # Delete it
        deleted = await store.delete(
            run_id=sample_envelope.run_id, checkpoint_id=sample_envelope.id
        )
        assert deleted is True

        # Verify it's gone
        retrieved = await store.get(
            run_id=sample_envelope.run_id, checkpoint_id=sample_envelope.id
        )
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_checkpoint_returns_false(self, store):
        """Test that deleting a nonexistent checkpoint returns False."""
        deleted = await store.delete(
            run_id=RunId("nonexistent_run"), checkpoint_id=CheckpointId("nonexistent")
        )
        assert deleted is False

    @pytest.mark.asyncio
    async def test_purge_by_run_id(self, store, sample_checkpoint):
        """Test purging all checkpoints for a run."""
        run_id = RunId("test_run_purge")

        # Create multiple checkpoints
        for i in range(3):
            envelope = CheckpointEnvelope(
                id=CheckpointId(f"checkpoint_{i:03d}"),
                run_id=run_id,
                checkpoint=sample_checkpoint,
            )
            await store.save(envelope)

        # Purge all checkpoints for the run
        count = await store.purge(run_id=run_id)
        assert count == 3

        # Verify they're gone
        query = CheckpointQuery(run_id=run_id)
        results, _ = await store.list(query)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_healthcheck_returns_true_when_healthy(self, store):
        """Test that healthcheck returns True when store is operational."""
        is_healthy = await store.healthcheck()
        assert is_healthy is True

    @pytest.mark.asyncio
    async def test_concurrent_saves(self, store, sample_checkpoint):
        """Test that concurrent saves don't corrupt data."""
        import asyncio

        run_id = RunId("test_run_concurrent")

        # Create multiple checkpoints concurrently
        async def save_checkpoint(idx: int):
            envelope = CheckpointEnvelope(
                id=CheckpointId(f"checkpoint_{idx:03d}"),
                run_id=run_id,
                checkpoint=sample_checkpoint,
            )
            return await store.save(envelope)

        # Save 10 checkpoints concurrently
        results = await asyncio.gather(*[save_checkpoint(i) for i in range(10)])
        assert len(results) == 10

        # Verify all were saved
        query = CheckpointQuery(run_id=run_id)
        retrieved, _ = await store.list(query)
        assert len(retrieved) == 10
