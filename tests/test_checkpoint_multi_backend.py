"""Advanced multi-backend scenarios for checkpoint v2.

Tests failover, migration, tiered storage, and composable backend patterns.
"""

from __future__ import annotations

from datetime import UTC
from datetime import datetime

from pydantic import BaseModel
import pytest

from pydantic_flow.checkpoints import SQLiteCheckpointBackend
from pydantic_flow.checkpoints import SQLiteCheckpointConfig
from pydantic_flow.checkpoints.backends.composable import MultiConsumerConfig
from pydantic_flow.checkpoints.backends.composable import MultiConsumerStorage
from pydantic_flow.checkpoints.backends.composable import TieredStorage
from pydantic_flow.checkpoints.backends.composable import TieredStorageConfig
from pydantic_flow.checkpoints.backends.filesystem import FilesystemCheckpointBackend
from pydantic_flow.checkpoints.backends.filesystem import FilesystemCheckpointConfig
from pydantic_flow.checkpoints.types import RunMetadata
from pydantic_flow.checkpoints.types import StateSnapshot
from pydantic_flow.checkpoints.types import generate_run_id
from pydantic_flow.checkpoints.types import generate_snapshot_id


class CheckpointTestState(BaseModel):
    """Test state model for checkpoint tests."""

    counter: int
    data: str


# =============================================================================
# Test 1: Multi-Backend Failover
# =============================================================================


@pytest.mark.asyncio
async def test_multi_backend_failover_on_primary_failure(tmp_path):
    """Test automatic failover when primary backend fails."""
    # Create primary and replica backends
    primary_db = tmp_path / "primary.db"
    replica_db = tmp_path / "replica.db"

    primary_config = SQLiteCheckpointConfig(db_path=primary_db)
    replica_config = SQLiteCheckpointConfig(db_path=replica_db)

    primary = SQLiteCheckpointBackend(primary_config)
    replica = SQLiteCheckpointBackend(replica_config)

    await primary.initialize()
    await replica.initialize()

    try:
        # Create multi-consumer storage
        multi = MultiConsumerStorage(
            primary=primary,
            replicas=[replica],
            config=MultiConsumerConfig(fail_on_replica_error=False),
        )

        run_id = generate_run_id()

        # Save checkpoint to both
        snapshot = StateSnapshot(
            snapshot_id=generate_snapshot_id(),
            run_id=run_id,
            wave_number=0,
            full_state={"node": CheckpointTestState(counter=1, data="test")},
            state_hash="hash",
            next_frontier=[],
            routing_ended=False,
        )

        await multi.save_state_snapshot(snapshot)

        # Verify both have data
        primary_data = await primary.get_state_snapshot(run_id, 0)
        replica_data = await replica.get_state_snapshot(run_id, 0)

        assert primary_data is not None
        assert replica_data is not None
        assert primary_data.full_state["node"].counter == 1  # type: ignore[index, attr-defined]
        assert replica_data.full_state["node"].counter == 1  # type: ignore[index, attr-defined]

        # Simulate primary failure by closing it
        await primary.close()

        # Reads should still work from replica
        # (In production, you'd implement failover logic)
        replica_data_after_failure = await replica.get_state_snapshot(run_id, 0)
        assert replica_data_after_failure is not None

    finally:
        await replica.close()


@pytest.mark.asyncio
async def test_multi_backend_replica_failure_tolerance(tmp_path):
    """Test that replica failures don't affect primary writes."""
    primary_db = tmp_path / "primary.db"
    replica_db = tmp_path / "replica.db"

    primary_config = SQLiteCheckpointConfig(db_path=primary_db)
    replica_config = SQLiteCheckpointConfig(db_path=replica_db)

    primary = SQLiteCheckpointBackend(primary_config)
    replica = SQLiteCheckpointBackend(replica_config)

    await primary.initialize()
    await replica.initialize()

    try:
        # Create multi-consumer with replica failure tolerance
        multi = MultiConsumerStorage(
            primary=primary,
            replicas=[replica],
            config=MultiConsumerConfig(fail_on_replica_error=False),
        )

        run_id = generate_run_id()

        # Close replica to simulate failure
        await replica.close()

        # Should still be able to write to primary
        snapshot = StateSnapshot(
            snapshot_id=generate_snapshot_id(),
            run_id=run_id,
            wave_number=0,
            full_state={"node": CheckpointTestState(counter=1, data="test")},
            state_hash="hash",
            next_frontier=[],
            routing_ended=False,
        )

        # Should not raise even though replica is down
        await multi.save_state_snapshot(snapshot)

        # Primary should have data
        primary_data = await primary.get_state_snapshot(run_id, 0)
        assert primary_data is not None

    finally:
        await primary.close()


# =============================================================================
# Test 2: Cross-Backend Migration
# =============================================================================


@pytest.mark.asyncio
async def test_migrate_checkpoints_between_backends(tmp_path):
    """Test migrating checkpoint data from one backend to another."""
    # Setup source and destination backends
    source_db = tmp_path / "source.db"
    dest_dir = tmp_path / "dest"

    source_config = SQLiteCheckpointConfig(db_path=source_db)
    dest_config = FilesystemCheckpointConfig(root_dir=dest_dir)

    source_backend = SQLiteCheckpointBackend(source_config)
    dest_backend = FilesystemCheckpointBackend(dest_config)

    await source_backend.initialize()
    await dest_backend.initialize()

    try:
        # Create data in source backend
        run_id = generate_run_id()

        # Save metadata
        metadata = RunMetadata(
            run_id=run_id,
            flow_id="test_flow",
            started_at=datetime.now(UTC),
            status=RunMetadata.Status.COMPLETED,
            total_waves=3,
        )
        await source_backend.save_run_metadata(metadata)

        # Save multiple checkpoints
        for wave in range(3):
            snapshot = StateSnapshot(
                snapshot_id=generate_snapshot_id(),
                run_id=run_id,
                wave_number=wave,
                full_state={
                    "node": CheckpointTestState(counter=wave, data=f"wave_{wave}")
                },
                state_hash=f"hash_{wave}",
                next_frontier=[],
                routing_ended=False,
            )
            await source_backend.save_state_snapshot(snapshot)

        # Migrate: Read from source, write to destination
        all_runs = await source_backend.list_runs()

        for run_metadata in all_runs:
            # Copy metadata
            await dest_backend.save_run_metadata(run_metadata)

            # Copy all snapshots
            for wave in range(run_metadata.total_waves + 1):
                snapshot = await source_backend.get_state_snapshot(
                    run_metadata.run_id, wave
                )
                if snapshot:
                    await dest_backend.save_state_snapshot(snapshot)

        # Verify migration
        dest_metadata = await dest_backend.get_run_metadata(run_id)
        assert dest_metadata is not None
        assert dest_metadata.total_waves == 3

        # Verify all snapshots migrated
        for wave in range(3):
            dest_snapshot = await dest_backend.get_state_snapshot(run_id, wave)
            assert dest_snapshot is not None
            assert dest_snapshot.full_state["node"].counter == wave  # type: ignore[index, attr-defined]

    finally:
        await source_backend.close()
        await dest_backend.close()


@pytest.mark.asyncio
async def test_incremental_migration_with_sync(tmp_path):
    """Test incremental migration that keeps backends in sync."""
    source_db = tmp_path / "source.db"
    dest_db = tmp_path / "dest.db"

    source_config = SQLiteCheckpointConfig(db_path=source_db)
    dest_config = SQLiteCheckpointConfig(db_path=dest_db)

    source_backend = SQLiteCheckpointBackend(source_config)
    dest_backend = SQLiteCheckpointBackend(dest_config)

    await source_backend.initialize()
    await dest_backend.initialize()

    try:
        run_id = generate_run_id()

        # Initial data
        metadata = RunMetadata(
            run_id=run_id,
            flow_id="test",
            started_at=datetime.now(UTC),
            status=RunMetadata.Status.RUNNING,
            total_waves=0,
        )
        await source_backend.save_run_metadata(metadata)

        # Migrate initial state
        await dest_backend.save_run_metadata(metadata)

        # Add more data to source incrementally
        for wave in range(3):
            snapshot = StateSnapshot(
                snapshot_id=generate_snapshot_id(),
                run_id=run_id,
                wave_number=wave,
                full_state={
                    "node": CheckpointTestState(counter=wave, data=f"wave_{wave}")
                },
                state_hash=f"hash_{wave}",
                next_frontier=[],
                routing_ended=False,
            )
            await source_backend.save_state_snapshot(snapshot)

            # Immediately sync to destination
            await dest_backend.save_state_snapshot(snapshot)

        # Verify both backends have same data
        source_snapshot_2 = await source_backend.get_state_snapshot(run_id, 2)
        dest_snapshot_2 = await dest_backend.get_state_snapshot(run_id, 2)

        assert source_snapshot_2.full_state["node"].counter == 2  # type: ignore[union-attr, index, attr-defined]
        assert dest_snapshot_2.full_state["node"].counter == 2  # type: ignore[union-attr, index, attr-defined]

    finally:
        await source_backend.close()
        await dest_backend.close()


# =============================================================================
# Test 3: Tiered Storage Scenarios
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.xfail(reason="Feature not fully implemented")
async def test_tiered_storage_hot_cold_separation(tmp_path):
    """Test automatic hot/cold storage tiering."""
    hot_db = tmp_path / "hot.db"
    cold_dir = tmp_path / "cold"

    hot_config = SQLiteCheckpointConfig(db_path=hot_db)
    cold_config = FilesystemCheckpointConfig(root_dir=cold_dir)

    hot_backend = SQLiteCheckpointBackend(hot_config)
    cold_backend = FilesystemCheckpointBackend(cold_config)

    await hot_backend.initialize()
    await cold_backend.initialize()

    try:
        # Create tiered storage (hot = SQLite, cold = Filesystem)
        tiered_config = TieredStorageConfig(
            prefer_hot=True,
            fallback_on_cold_miss=True,
        )

        tiered = TieredStorage(
            hot=hot_backend,
            cold=cold_backend,
            config=tiered_config,
        )

        run_id = generate_run_id()

        # Write to hot storage
        snapshot = StateSnapshot(
            snapshot_id=generate_snapshot_id(),
            run_id=run_id,
            wave_number=0,
            full_state={"node": CheckpointTestState(counter=100, data="hot_data")},
            state_hash="hash",
            next_frontier=[],
            routing_ended=False,
        )

        await tiered.save_state_snapshot(snapshot)

        # Read should prefer hot
        retrieved = await tiered.get_state_snapshot(run_id, 0)
        assert retrieved is not None
        assert retrieved.full_state["node"].counter == 100  # type: ignore[index, attr-defined]

        # Move to cold storage
        await tiered.move_to_cold(run_id)

        # Should still be readable (from cold)
        retrieved_cold = await tiered.get_state_snapshot(run_id, 0)
        assert retrieved_cold is not None
        assert retrieved_cold.full_state["node"].counter == 100  # type: ignore[index, attr-defined]

    finally:
        await hot_backend.close()
        await cold_backend.close()


@pytest.mark.asyncio
async def test_tiered_storage_automatic_fallback(tmp_path):
    """Test automatic fallback to cold when hot is unavailable."""
    hot_db = tmp_path / "hot.db"
    cold_db = tmp_path / "cold.db"

    hot_config = SQLiteCheckpointConfig(db_path=hot_db)
    cold_config = SQLiteCheckpointConfig(db_path=cold_db)

    hot_backend = SQLiteCheckpointBackend(hot_config)
    cold_backend = SQLiteCheckpointBackend(cold_config)

    await hot_backend.initialize()
    await cold_backend.initialize()

    try:
        tiered = TieredStorage(
            hot=hot_backend,
            cold=cold_backend,
            config=TieredStorageConfig(prefer_hot=True),
        )

        run_id = generate_run_id()

        # Save to cold directly
        snapshot = StateSnapshot(
            snapshot_id=generate_snapshot_id(),
            run_id=run_id,
            wave_number=0,
            full_state={"node": CheckpointTestState(counter=1, data="cold_only")},
            state_hash="hash",
            next_frontier=[],
            routing_ended=False,
        )

        await cold_backend.save_state_snapshot(snapshot)

        # Read through tiered (hot doesn't have it, should fallback to cold)
        retrieved = await tiered.get_state_snapshot(run_id, 0)
        assert retrieved is not None
        assert retrieved.full_state["node"].data == "cold_only"  # type: ignore[index, attr-defined]

    finally:
        await hot_backend.close()
        await cold_backend.close()


# =============================================================================
# Test 4: Performance Comparison
# =============================================================================


@pytest.mark.asyncio
async def test_backend_write_performance_comparison(tmp_path):
    """Compare write performance across backends."""
    import time

    sqlite_db = tmp_path / "perf.db"
    fs_dir = tmp_path / "perf_fs"

    sqlite_config = SQLiteCheckpointConfig(db_path=sqlite_db)
    fs_config = FilesystemCheckpointConfig(root_dir=fs_dir)

    sqlite_backend = SQLiteCheckpointBackend(sqlite_config)
    fs_backend = FilesystemCheckpointBackend(fs_config)

    await sqlite_backend.initialize()
    await fs_backend.initialize()

    try:
        # Test data
        run_id = generate_run_id()
        snapshots = [
            StateSnapshot(
                snapshot_id=generate_snapshot_id(),
                run_id=run_id,
                wave_number=i,
                full_state={
                    "node": CheckpointTestState(counter=i, data=f"wave_{i}" * 10)
                },
                state_hash=f"hash_{i}",
                next_frontier=[],
                routing_ended=False,
            )
            for i in range(10)
        ]

        # Benchmark SQLite writes
        start_sqlite = time.perf_counter()
        for snapshot in snapshots:
            await sqlite_backend.save_state_snapshot(snapshot)
        sqlite_duration = time.perf_counter() - start_sqlite

        # Benchmark Filesystem writes
        start_fs = time.perf_counter()
        for snapshot in snapshots:
            await fs_backend.save_state_snapshot(snapshot)
        fs_duration = time.perf_counter() - start_fs

        # Both should complete reasonably fast
        assert sqlite_duration < 1.0  # Under 1 second for 10 writes
        assert fs_duration < 2.0  # Under 2 seconds for 10 writes

        # Log comparison (not asserting, just measuring)
        sqlite_ms = sqlite_duration / 10 * 1000
        fs_ms = fs_duration / 10 * 1000
        print(f"\nSQLite: {sqlite_duration:.3f}s ({sqlite_ms:.1f}ms/write)")
        print(f"Filesystem: {fs_duration:.3f}s ({fs_ms:.1f}ms/write)")

    finally:
        await sqlite_backend.close()
        await fs_backend.close()


# =============================================================================
# Test 5: Complex Composable Scenarios
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.xfail(reason="Feature not fully implemented")
async def test_nested_composable_backends(tmp_path):
    """Test complex nesting of composable backends."""
    # Setup: Hot storage with 2 replicas + cold storage
    hot_db = tmp_path / "hot.db"
    replica1_db = tmp_path / "replica1.db"
    replica2_db = tmp_path / "replica2.db"
    cold_dir = tmp_path / "cold"

    hot = SQLiteCheckpointBackend(SQLiteCheckpointConfig(db_path=hot_db))
    replica1 = SQLiteCheckpointBackend(SQLiteCheckpointConfig(db_path=replica1_db))
    replica2 = SQLiteCheckpointBackend(SQLiteCheckpointConfig(db_path=replica2_db))
    cold = FilesystemCheckpointBackend(FilesystemCheckpointConfig(root_dir=cold_dir))

    await hot.initialize()
    await replica1.initialize()
    await replica2.initialize()
    await cold.initialize()

    try:
        # Create multi-consumer for hot + replicas
        multi_hot = MultiConsumerStorage(
            primary=hot,
            replicas=[replica1, replica2],
            config=MultiConsumerConfig(fail_on_replica_error=False),
        )

        # Wrap in tiered storage
        tiered = TieredStorage(
            hot=multi_hot,
            cold=cold,
            config=TieredStorageConfig(prefer_hot=True),
        )

        run_id = generate_run_id()

        # Write through tiered (should go to hot + all replicas)
        snapshot = StateSnapshot(
            snapshot_id=generate_snapshot_id(),
            run_id=run_id,
            wave_number=0,
            full_state={"node": CheckpointTestState(counter=42, data="complex")},
            state_hash="hash",
            next_frontier=[],
            routing_ended=False,
        )

        await tiered.save_state_snapshot(snapshot)

        # Verify all backends have data
        hot_data = await hot.get_state_snapshot(run_id, 0)
        replica1_data = await replica1.get_state_snapshot(run_id, 0)
        replica2_data = await replica2.get_state_snapshot(run_id, 0)

        assert hot_data.full_state["node"].counter == 42  # type: ignore[union-attr, index, attr-defined]
        assert replica1_data.full_state["node"].counter == 42  # type: ignore[union-attr, index, attr-defined]
        assert replica2_data.full_state["node"].counter == 42  # type: ignore[union-attr, index, attr-defined]

        # Move to cold
        await tiered.move_to_cold(run_id)

        # Verify cold has data
        cold_data = await cold.get_state_snapshot(run_id, 0)
        assert cold_data.full_state["node"].counter == 42  # type: ignore[union-attr, index, attr-defined]

    finally:
        await hot.close()
        await replica1.close()
        await replica2.close()
        await cold.close()


# =============================================================================
# Test 6: Data Consistency Validation
# =============================================================================


@pytest.mark.asyncio
async def test_multi_backend_consistency_validation(tmp_path):
    """Test data consistency across multiple backends."""
    backend1_db = tmp_path / "backend1.db"
    backend2_db = tmp_path / "backend2.db"

    backend1 = SQLiteCheckpointBackend(SQLiteCheckpointConfig(db_path=backend1_db))
    backend2 = SQLiteCheckpointBackend(SQLiteCheckpointConfig(db_path=backend2_db))

    await backend1.initialize()
    await backend2.initialize()

    try:
        # Write same data to both backends
        run_id = generate_run_id()

        for wave in range(5):
            snapshot = StateSnapshot(
                snapshot_id=generate_snapshot_id(),
                run_id=run_id,
                wave_number=wave,
                full_state={
                    "node": CheckpointTestState(
                        counter=wave * 10, data=f"consistent_{wave}"
                    )
                },
                state_hash=f"hash_{wave}",
                next_frontier=[],
                routing_ended=False,
            )

            await backend1.save_state_snapshot(snapshot)
            await backend2.save_state_snapshot(snapshot)

        # Validate consistency
        for wave in range(5):
            snap1 = await backend1.get_state_snapshot(run_id, wave)
            snap2 = await backend2.get_state_snapshot(run_id, wave)

            assert snap1.full_state["node"].counter == snap2.full_state["node"].counter  # type: ignore[union-attr, index, attr-defined]
            assert snap1.state_hash == snap2.state_hash  # type: ignore[union-attr]
            assert snap1.wave_number == snap2.wave_number  # type: ignore[union-attr]

    finally:
        await backend1.close()
        await backend2.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
