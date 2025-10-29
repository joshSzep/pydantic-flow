"""Tests for FlatFileCheckpointStore implementation."""

from __future__ import annotations

from pathlib import Path

import pytest

from pydantic_flow.checkpoints.flatfile import FlatFileCheckpointStore
from pydantic_flow.checkpoints.flatfile import FlatFileCheckpointStoreConfig
from pydantic_flow.checkpoints.flatfile import PartitioningStrategy
from tests.test_checkpoints_conformance import CheckpointStoreConformanceTests


class TestFlatFileCheckpointStoreConformance(CheckpointStoreConformanceTests):
    """Conformance tests for FlatFileCheckpointStore."""

    @pytest.fixture
    async def store(self, tmp_path: Path) -> FlatFileCheckpointStore:
        """Create a FlatFileCheckpointStore instance for testing."""
        config = FlatFileCheckpointStoreConfig(
            base_path=tmp_path / "checkpoints",
            partitioning=PartitioningStrategy.BY_RUN,
        )
        store = FlatFileCheckpointStore(config)
        await store.healthcheck()
        return store


class TestFlatFileCheckpointStoreSpecific:
    """Tests specific to FlatFileCheckpointStore implementation."""

    @pytest.fixture
    async def store(self, tmp_path: Path) -> FlatFileCheckpointStore:
        """Create a FlatFileCheckpointStore instance for testing."""
        config = FlatFileCheckpointStoreConfig(
            base_path=tmp_path / "checkpoints",
            partitioning=PartitioningStrategy.BY_RUN,
        )
        return FlatFileCheckpointStore(config)

    @pytest.mark.asyncio
    async def test_partitioning_by_run(self, tmp_path: Path, sample_envelope) -> None:
        """Test BY_RUN partitioning strategy creates correct directory structure."""
        config = FlatFileCheckpointStoreConfig(
            base_path=tmp_path / "checkpoints",
            partitioning=PartitioningStrategy.BY_RUN,
        )
        store = FlatFileCheckpointStore(config)

        await store.save(sample_envelope)

        expected_path = (
            tmp_path
            / "checkpoints"
            / "runs"
            / sample_envelope.run_id
            / f"{sample_envelope.id}.json"
        )
        assert expected_path.exists()

        index_path = (
            tmp_path / "checkpoints" / "runs" / sample_envelope.run_id / "index.jsonl"
        )
        assert index_path.exists()

    @pytest.mark.asyncio
    async def test_partitioning_by_date(self, tmp_path: Path, sample_envelope) -> None:
        """Test BY_DATE partitioning strategy creates date-based directories."""
        config = FlatFileCheckpointStoreConfig(
            base_path=tmp_path / "checkpoints",
            partitioning=PartitioningStrategy.BY_DATE,
        )
        store = FlatFileCheckpointStore(config)

        await store.save(sample_envelope)

        date_str = sample_envelope.created_at.strftime("%Y-%m-%d")
        expected_path = (
            tmp_path
            / "checkpoints"
            / "dates"
            / date_str
            / sample_envelope.run_id
            / f"{sample_envelope.id}.json"
        )
        assert expected_path.exists()

    @pytest.mark.asyncio
    async def test_partitioning_none(self, tmp_path: Path, sample_envelope) -> None:
        """Test NONE partitioning strategy stores files flat in base directory."""
        config = FlatFileCheckpointStoreConfig(
            base_path=tmp_path / "checkpoints",
            partitioning=PartitioningStrategy.NONE,
        )
        store = FlatFileCheckpointStore(config)

        await store.save(sample_envelope)

        expected_path = (
            tmp_path
            / "checkpoints"
            / f"{sample_envelope.run_id}_{sample_envelope.id}.json"
        )
        assert expected_path.exists()

    @pytest.mark.asyncio
    async def test_index_maintains_chronological_order(
        self, store: FlatFileCheckpointStore, sample_checkpoint
    ) -> None:
        """Test that index file maintains chronological order of checkpoints."""
        from datetime import UTC
        from datetime import datetime
        import json

        import anyio

        from pydantic_flow.checkpoints.interface import CheckpointEnvelope
        from pydantic_flow.checkpoints.interface import CheckpointId
        from pydantic_flow.checkpoints.interface import RunId

        run_id = RunId("test_run")

        envelopes = [
            CheckpointEnvelope(
                id=CheckpointId(f"ckpt_{i}"),
                run_id=run_id,
                node_id=f"node_{i}",
                created_at=datetime(2024, 1, 1, 12, i, 0, tzinfo=UTC),
                schema_version=1,
                checkpoint=sample_checkpoint,
            )
            for i in range(3)
        ]

        for envelope in envelopes:
            await store.save(envelope)

        index_path = store._get_index_path(run_id)
        assert index_path.exists()

        async with await anyio.open_file(index_path, "r") as f:
            content = await f.read()
            lines = content.strip().split("\n")
            assert len(lines) == 3

            for i, line in enumerate(lines):
                entry = json.loads(line)
                assert entry["checkpoint_id"] == f"ckpt_{i}"

    @pytest.mark.asyncio
    async def test_concurrent_writes_to_same_run(
        self, store: FlatFileCheckpointStore, sample_checkpoint
    ) -> None:
        """Test concurrent writes to the same run are handled correctly."""
        import asyncio
        from datetime import UTC
        from datetime import datetime

        from pydantic_flow.checkpoints.interface import CheckpointEnvelope
        from pydantic_flow.checkpoints.interface import CheckpointId
        from pydantic_flow.checkpoints.interface import CheckpointQuery
        from pydantic_flow.checkpoints.interface import RunId

        run_id = RunId("concurrent_run")

        async def save_checkpoint(checkpoint_num: int) -> None:
            envelope = CheckpointEnvelope(
                id=CheckpointId(f"ckpt_{checkpoint_num}"),
                run_id=run_id,
                node_id=f"node_{checkpoint_num}",
                created_at=datetime.now(UTC),
                schema_version=1,
                checkpoint=sample_checkpoint,
            )
            await store.save(envelope)

        await asyncio.gather(*[save_checkpoint(i) for i in range(5)])

        query = CheckpointQuery(run_id=run_id)
        checkpoints, _ = await store.list(query)
        assert len(checkpoints) == 5

    @pytest.mark.asyncio
    async def test_atomic_writes(
        self, store: FlatFileCheckpointStore, sample_envelope
    ) -> None:
        """Test that writes are atomic (no partial files)."""
        await store.save(sample_envelope)

        checkpoint_path = store._get_checkpoint_path(sample_envelope)
        assert checkpoint_path.exists()

        checkpoint_path_str: str = str(checkpoint_path)
        assert not checkpoint_path_str.endswith(".tmp")
        assert not any(
            (checkpoint_path.parent / f".{checkpoint_path.name}")
            .with_suffix(ext)
            .exists()
            for ext in [".tmp", ".partial", ".writing"]
        )


@pytest.fixture
def anyio():
    """Provide anyio for tests."""
    import anyio

    return anyio
