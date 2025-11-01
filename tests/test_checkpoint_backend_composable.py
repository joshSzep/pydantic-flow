"""Tests for composable checkpoint backends."""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
from pathlib import Path
import tempfile

from pydantic import BaseModel
import pytest

from pydantic_flow.checkpoints.backends.composable import MultiConsumerConfig
from pydantic_flow.checkpoints.backends.composable import MultiConsumerStorage
from pydantic_flow.checkpoints.backends.composable import TieredStorage
from pydantic_flow.checkpoints.backends.composable import TieredStorageConfig
from pydantic_flow.checkpoints.backends.sqlite import SQLiteCheckpointBackend
from pydantic_flow.checkpoints.backends.sqlite import SQLiteCheckpointConfig
from pydantic_flow.checkpoints.types import RunMetadata
from pydantic_flow.checkpoints.types import StateSnapshot
from pydantic_flow.checkpoints.types import generate_run_id
from pydantic_flow.checkpoints.types import generate_snapshot_id


class SampleState(BaseModel):
    """Sample state for testing."""

    value: int
    name: str


@pytest.mark.asyncio
async def test_multi_consumer_basic() -> None:
    """Test basic multi-consumer storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        primary_config = SQLiteCheckpointConfig(db_path=Path(tmpdir) / "primary.db")
        replica1_config = SQLiteCheckpointConfig(db_path=Path(tmpdir) / "replica1.db")
        replica2_config = SQLiteCheckpointConfig(db_path=Path(tmpdir) / "replica2.db")

        primary = SQLiteCheckpointBackend(primary_config)
        replica1 = SQLiteCheckpointBackend(replica1_config)
        replica2 = SQLiteCheckpointBackend(replica2_config)

        storage = MultiConsumerStorage(primary=primary, replicas=[replica1, replica2])

        await storage.initialize()

        run_id = generate_run_id()
        snapshot = StateSnapshot(
            snapshot_id=generate_snapshot_id(),
            run_id=run_id,
            wave_number=0,
            full_state={"node1": SampleState(value=1, name="test")},
            state_hash="hash123",
            next_frontier=["node2"],
            routing_ended=False,
        )

        await storage.save_state_snapshot(snapshot)

        primary_snapshot = await primary.get_state_snapshot(run_id, 0)
        replica1_snapshot = await replica1.get_state_snapshot(run_id, 0)
        replica2_snapshot = await replica2.get_state_snapshot(run_id, 0)

        assert primary_snapshot is not None
        assert replica1_snapshot is not None
        assert replica2_snapshot is not None

        assert primary_snapshot.snapshot_id == snapshot.snapshot_id
        assert replica1_snapshot.snapshot_id == snapshot.snapshot_id
        assert replica2_snapshot.snapshot_id == snapshot.snapshot_id

        await storage.close()


@pytest.mark.asyncio
async def test_multi_consumer_replica_failure() -> None:
    """Test multi-consumer with replica failure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        primary_config = SQLiteCheckpointConfig(db_path=Path(tmpdir) / "primary.db")
        replica_config = SQLiteCheckpointConfig(db_path=Path(tmpdir) / "replica.db")

        primary = SQLiteCheckpointBackend(primary_config)
        replica = SQLiteCheckpointBackend(replica_config)

        config = MultiConsumerConfig(fail_on_replica_error=False)
        storage = MultiConsumerStorage(
            primary=primary, replicas=[replica], config=config
        )

        await storage.initialize()

        await replica.close()

        run_id = generate_run_id()
        snapshot = StateSnapshot(
            snapshot_id=generate_snapshot_id(),
            run_id=run_id,
            wave_number=0,
            full_state={},
            state_hash="hash",
            next_frontier=[],
            routing_ended=True,
        )

        await storage.save_state_snapshot(snapshot)

        primary_snapshot = await primary.get_state_snapshot(run_id, 0)
        assert primary_snapshot is not None

        await storage.close()


@pytest.mark.asyncio
async def test_multi_consumer_reads_from_primary() -> None:
    """Test that reads only come from primary."""
    with tempfile.TemporaryDirectory() as tmpdir:
        primary_config = SQLiteCheckpointConfig(db_path=Path(tmpdir) / "primary.db")
        replica_config = SQLiteCheckpointConfig(db_path=Path(tmpdir) / "replica.db")

        primary = SQLiteCheckpointBackend(primary_config)
        replica = SQLiteCheckpointBackend(replica_config)

        storage = MultiConsumerStorage(primary=primary, replicas=[replica])

        await storage.initialize()

        run_id = generate_run_id()
        snapshot = StateSnapshot(
            snapshot_id=generate_snapshot_id(),
            run_id=run_id,
            wave_number=0,
            full_state={"node1": SampleState(value=1, name="primary")},
            state_hash="hash1",
            next_frontier=[],
            routing_ended=True,
        )

        await primary.save_state_snapshot(snapshot)

        replica_snapshot = StateSnapshot(
            snapshot_id=generate_snapshot_id(),
            run_id=run_id,
            wave_number=0,
            full_state={"node1": SampleState(value=2, name="replica")},
            state_hash="hash2",
            next_frontier=[],
            routing_ended=True,
        )

        await replica.save_state_snapshot(replica_snapshot)

        retrieved = await storage.get_state_snapshot(run_id, 0)
        assert retrieved is not None
        assert retrieved.full_state["node1"].name == "primary"  # type: ignore[union-attr]

        await storage.close()


@pytest.mark.asyncio
async def test_tiered_storage_hot_fallback() -> None:
    """Test tiered storage with hot/cold fallback."""
    with tempfile.TemporaryDirectory() as tmpdir:
        hot_config = SQLiteCheckpointConfig(db_path=Path(tmpdir) / "hot.db")
        cold_config = SQLiteCheckpointConfig(db_path=Path(tmpdir) / "cold.db")

        hot = SQLiteCheckpointBackend(hot_config)
        cold = SQLiteCheckpointBackend(cold_config)

        storage = TieredStorage(hot=hot, cold=cold)

        await storage.initialize()

        run_id = generate_run_id()
        cold_snapshot = StateSnapshot(
            snapshot_id=generate_snapshot_id(),
            run_id=run_id,
            wave_number=0,
            full_state={"node1": SampleState(value=1, name="cold")},
            state_hash="hash",
            next_frontier=[],
            routing_ended=True,
        )

        await cold.save_state_snapshot(cold_snapshot)

        retrieved = await storage.get_state_snapshot(run_id, 0)
        assert retrieved is not None
        assert retrieved.full_state["node1"].name == "cold"  # type: ignore[union-attr]

        await storage.close()


@pytest.mark.asyncio
async def test_tiered_storage_prefer_hot() -> None:
    """Test tiered storage prefers hot over cold."""
    with tempfile.TemporaryDirectory() as tmpdir:
        hot_config = SQLiteCheckpointConfig(db_path=Path(tmpdir) / "hot.db")
        cold_config = SQLiteCheckpointConfig(db_path=Path(tmpdir) / "cold.db")

        hot = SQLiteCheckpointBackend(hot_config)
        cold = SQLiteCheckpointBackend(cold_config)

        storage = TieredStorage(hot=hot, cold=cold)

        await storage.initialize()

        run_id = generate_run_id()
        snapshot_id = generate_snapshot_id()

        hot_snapshot = StateSnapshot(
            snapshot_id=snapshot_id,
            run_id=run_id,
            wave_number=0,
            full_state={"node1": SampleState(value=1, name="hot")},
            state_hash="hash1",
            next_frontier=[],
            routing_ended=True,
        )

        cold_snapshot = StateSnapshot(
            snapshot_id=generate_snapshot_id(),
            run_id=run_id,
            wave_number=0,
            full_state={"node1": SampleState(value=2, name="cold")},
            state_hash="hash2",
            next_frontier=[],
            routing_ended=True,
        )

        await hot.save_state_snapshot(hot_snapshot)
        await cold.save_state_snapshot(cold_snapshot)

        retrieved = await storage.get_state_snapshot(run_id, 0)
        assert retrieved is not None
        assert retrieved.full_state["node1"].name == "hot"  # type: ignore[union-attr]

        await storage.close()


@pytest.mark.asyncio
async def test_tiered_storage_move_to_cold() -> None:
    """Test moving data from hot to cold storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        hot_config = SQLiteCheckpointConfig(db_path=Path(tmpdir) / "hot.db")
        cold_config = SQLiteCheckpointConfig(db_path=Path(tmpdir) / "cold.db")

        hot = SQLiteCheckpointBackend(hot_config)
        cold = SQLiteCheckpointBackend(cold_config)

        storage = TieredStorage(hot=hot, cold=cold)

        await storage.initialize()

        run_id = generate_run_id()

        metadata = RunMetadata(
            run_id=run_id,
            flow_id="test_flow",
            started_at=datetime.now(UTC),
            status=RunMetadata.Status.COMPLETED,
            total_waves=2,
        )

        await hot.save_run_metadata(metadata)

        snapshot1 = StateSnapshot(
            snapshot_id=generate_snapshot_id(),
            run_id=run_id,
            wave_number=0,
            full_state={"node1": SampleState(value=1, name="wave0")},
            state_hash="hash0",
            next_frontier=["node2"],
            routing_ended=False,
        )

        snapshot2 = StateSnapshot(
            snapshot_id=generate_snapshot_id(),
            run_id=run_id,
            wave_number=1,
            full_state={"node2": SampleState(value=2, name="wave1")},
            state_hash="hash1",
            next_frontier=[],
            routing_ended=True,
        )

        await hot.save_state_snapshot(snapshot1)
        await hot.save_state_snapshot(snapshot2)

        await storage.move_to_cold(run_id)

        hot_metadata = await hot.get_run_metadata(run_id)
        assert hot_metadata is None

        cold_metadata = await cold.get_run_metadata(run_id)
        assert cold_metadata is not None
        assert cold_metadata.run_id == run_id

        cold_snapshot = await cold.get_state_snapshot(run_id, 0)
        assert cold_snapshot is not None

        await storage.close()


@pytest.mark.asyncio
async def test_tiered_storage_no_fallback() -> None:
    """Test tiered storage without fallback."""
    with tempfile.TemporaryDirectory() as tmpdir:
        hot_config = SQLiteCheckpointConfig(db_path=Path(tmpdir) / "hot.db")
        cold_config = SQLiteCheckpointConfig(db_path=Path(tmpdir) / "cold.db")

        hot = SQLiteCheckpointBackend(hot_config)
        cold = SQLiteCheckpointBackend(cold_config)

        config = TieredStorageConfig(fallback_on_cold_miss=False)
        storage = TieredStorage(hot=hot, cold=cold, config=config)

        await storage.initialize()

        run_id = generate_run_id()
        cold_snapshot = StateSnapshot(
            snapshot_id=generate_snapshot_id(),
            run_id=run_id,
            wave_number=0,
            full_state={"node1": SampleState(value=1, name="cold")},
            state_hash="hash",
            next_frontier=[],
            routing_ended=True,
        )

        await cold.save_state_snapshot(cold_snapshot)

        retrieved = await storage.get_state_snapshot(run_id, 0)
        assert retrieved is None

        await storage.close()
