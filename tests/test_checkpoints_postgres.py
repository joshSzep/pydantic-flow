"""Tests for PostgresCheckpointStore implementation with mocked asyncpg."""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest

from pydantic_flow.hitl.checkpoints.interface import CheckpointQuery
from pydantic_flow.hitl.checkpoints.postgres import PostgresCheckpointStore
from pydantic_flow.hitl.checkpoints.postgres import PostgresCheckpointStoreConfig
from tests.test_checkpoints_conformance import CheckpointStoreConformanceTests


class MockPostgresConnection:
    """Mock asyncpg connection for testing."""

    def __init__(self) -> None:
        """Initialize mock connection with in-memory storage."""
        self._checkpoints: list[dict[str, str | datetime | int | None]] = []

    async def execute(self, query: str, *args: object) -> str:
        """Execute a query (INSERT, UPDATE, DELETE)."""
        if "INSERT" in query or "CREATE" in query:
            if "INSERT" in query and len(args) >= 6:
                checkpoint_id = str(args[1]) if args[1] else None

                if "ON CONFLICT(checkpoint_id) DO UPDATE" in query:
                    for cp in self._checkpoints:
                        if cp["checkpoint_id"] == checkpoint_id:
                            cp["run_id"] = str(args[0]) if args[0] else None
                            cp["node_id"] = (
                                str(args[2]) if args[2] is not None else None
                            )
                            cp["created_at"] = (
                                args[3] if isinstance(args[3], datetime) else None
                            )
                            cp["schema_version"] = (
                                int(args[4]) if args[4] else None  # type: ignore[arg-type]
                            )
                            cp["envelope_json"] = str(args[5]) if args[5] else None
                            cp["content_hash"] = (
                                str(args[6]) if len(args) > 6 and args[6] else None
                            )
                            return "INSERT 0 1"

                self._checkpoints.append({
                    "run_id": str(args[0]) if args[0] else None,
                    "checkpoint_id": checkpoint_id,
                    "node_id": str(args[2]) if args[2] is not None else None,
                    "created_at": (args[3] if isinstance(args[3], datetime) else None),
                    "schema_version": (
                        int(args[4]) if args[4] else None  # type: ignore[arg-type]
                    ),
                    "envelope_json": str(args[5]) if args[5] else None,
                    "content_hash": (
                        str(args[6]) if len(args) > 6 and args[6] else None
                    ),
                })
            return "INSERT 0 1"
        elif "DELETE" in query:
            if "WHERE run_id = $1 AND checkpoint_id = $2" in query:
                run_id = str(args[0]) if args else None
                checkpoint_id = str(args[1]) if len(args) > 1 else None
                initial_count = len(self._checkpoints)
                self._checkpoints = [
                    cp
                    for cp in self._checkpoints
                    if not (
                        cp["run_id"] == run_id and cp["checkpoint_id"] == checkpoint_id
                    )
                ]
                deleted_count = initial_count - len(self._checkpoints)
                return f"DELETE {deleted_count}"
            else:
                run_id = str(args[0]) if args else None
                initial_count = len(self._checkpoints)
                self._checkpoints = [
                    cp for cp in self._checkpoints if cp["run_id"] != run_id
                ]
                deleted_count = initial_count - len(self._checkpoints)
                return f"DELETE {deleted_count}"
        return "OK"

    def _filter_checkpoints(
        self, query: str, args: tuple[object, ...]
    ) -> list[dict[str, str | datetime | int | None]]:
        """Filter checkpoints based on query conditions."""
        filtered = list(self._checkpoints)
        arg_idx = 0

        if "WHERE run_id = $1 AND checkpoint_id = $2" in query:
            run_id = str(args[0]) if len(args) > 0 else None
            checkpoint_id = str(args[1]) if len(args) > 1 else None
            filtered = [
                cp
                for cp in filtered
                if cp["run_id"] == run_id and cp["checkpoint_id"] == checkpoint_id
            ]
        elif "WHERE" in query:
            if "run_id = $" in query:
                run_id = str(args[arg_idx]) if len(args) > arg_idx else None
                filtered = [cp for cp in filtered if cp["run_id"] == run_id]
                arg_idx += 1

            if "node_id = $" in query:
                node_id = str(args[arg_idx]) if len(args) > arg_idx else None
                filtered = [cp for cp in filtered if cp["node_id"] == node_id]
                arg_idx += 1

            if "created_at >= $" in query:
                since = args[arg_idx] if len(args) > arg_idx else None
                if since and isinstance(since, datetime):
                    filtered = [
                        cp
                        for cp in filtered
                        if isinstance(cp["created_at"], datetime)
                        and cp["created_at"] >= since
                    ]
                arg_idx += 1

            if "created_at <= $" in query:
                until = args[arg_idx] if len(args) > arg_idx else None
                if until and isinstance(until, datetime):
                    filtered = [
                        cp
                        for cp in filtered
                        if isinstance(cp["created_at"], datetime)
                        and cp["created_at"] <= until
                    ]
                arg_idx += 1

        return filtered

    def _sort_results(
        self, query: str, results: list[dict[str, object]]
    ) -> list[dict[str, object]]:
        """Sort results based on ORDER BY clause."""
        if "ORDER BY created_at DESC" in query:
            results.sort(
                key=lambda x: x["created_at"].timestamp()  # type: ignore[union-attr]
                if isinstance(x.get("created_at"), datetime)
                else 0,
                reverse=True,
            )
        elif "ORDER BY created_at ASC" in query:
            results.sort(
                key=lambda x: x["created_at"].timestamp()  # type: ignore[union-attr]
                if isinstance(x.get("created_at"), datetime)
                else 0
            )
        return results

    def _limit_results(
        self, query: str, args: tuple[object, ...], results: list[dict[str, object]]
    ) -> list[dict[str, object]]:
        """Apply LIMIT and OFFSET clauses to results."""
        if "LIMIT" not in query:
            return results

        limit = None
        offset = 0

        if "LIMIT $" in query and "OFFSET $" in query and len(args) >= 2:
            limit = int(args[-2]) if args[-2] is not None else None  # type: ignore[arg-type]
            offset = int(args[-1]) if args[-1] is not None else 0  # type: ignore[arg-type]
        elif "LIMIT $" in query and "OFFSET $" not in query and len(args) >= 1:
            limit = int(args[-1]) if args[-1] is not None else None  # type: ignore[arg-type]
        else:
            import re

            limit_match = re.search(r"LIMIT\s+(\d+)", query, re.IGNORECASE)
            if limit_match:
                limit = int(limit_match.group(1))

            offset_match = re.search(r"OFFSET\s+(\d+)", query, re.IGNORECASE)
            if offset_match:
                offset = int(offset_match.group(1))

        if offset > 0:
            results = results[offset:]

        if limit is not None:
            results = results[:limit]

        return results

    async def fetch(self, query: str, *args: object) -> list[dict[str, object]]:
        """Fetch multiple rows."""
        filtered = self._filter_checkpoints(query, args)

        results: list[dict[str, object]] = [
            {
                "checkpoint_id": cp["checkpoint_id"],
                "run_id": cp["run_id"],
                "node_id": cp["node_id"],
                "created_at": cp["created_at"],
                "schema_version": cp["schema_version"],
                "envelope_json": cp["envelope_json"],
                "content_hash": cp["content_hash"],
            }
            for cp in filtered
        ]

        results = self._sort_results(query, results)
        results = self._limit_results(query, args, results)

        if "envelope::text" in query:
            results = [{"envelope": r["envelope_json"]} for r in results]

        return results  # type: ignore[return-value]

    async def fetchrow(self, query: str, *args: object) -> dict[str, object] | None:
        """Fetch a single row."""
        results = await self.fetch(query, *args)
        return results[0] if results else None

    async def fetchval(self, query: str, *args: object) -> object | None:
        """Fetch a single value."""
        if "COUNT" in query:
            run_id = str(args[0]) if args else None
            return sum(1 for cp in self._checkpoints if cp["run_id"] == run_id)
        if "SELECT 1 FROM" in query and "WHERE checkpoint_id" in query:
            checkpoint_id = str(args[0]) if args else None
            for cp in self._checkpoints:
                if cp["checkpoint_id"] == checkpoint_id:
                    return 1
            return None
        return None

    async def close(self) -> None:
        """Close connection."""
        pass

    def transaction(self) -> MockPostgresTransaction:
        """Create a transaction context manager."""
        return MockPostgresTransaction()


class MockPostgresTransaction:
    """Mock asyncpg transaction."""

    async def __aenter__(self) -> MockPostgresTransaction:
        """Enter transaction."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[no-untyped-def]
        """Exit transaction."""
        pass


class MockPostgresPool:
    """Mock asyncpg connection pool."""

    def __init__(self) -> None:
        """Initialize mock pool."""
        self._connection = MockPostgresConnection()

    def acquire(self):
        """Acquire connection from pool (returns async context manager)."""
        return self

    async def __aenter__(self) -> MockPostgresConnection:
        """Enter async context manager."""
        return self._connection

    async def __aexit__(self, *args) -> None:
        """Exit async context manager."""
        pass

    async def release(self, connection: MockPostgresConnection) -> None:
        """Release connection back to pool."""
        pass

    async def close(self) -> None:
        """Close pool."""
        pass


@pytest.fixture
async def mock_pg_pool() -> MockPostgresPool:
    """Create a mock Postgres connection pool."""
    return MockPostgresPool()


@pytest.fixture
async def mock_create_pool(mock_pg_pool: MockPostgresPool) -> AsyncMock:
    """Create async mock that returns our mock pool."""
    return AsyncMock(return_value=mock_pg_pool)


class TestPostgresCheckpointStoreConformance(CheckpointStoreConformanceTests):
    """Conformance tests for PostgresCheckpointStore with mocked asyncpg."""

    @pytest.fixture
    async def store(
        self, mock_pg_pool: MockPostgresPool, mock_create_pool: AsyncMock
    ) -> PostgresCheckpointStore:
        """Create a PostgresCheckpointStore with mocked asyncpg pool."""
        config = PostgresCheckpointStoreConfig(
            dsn="postgresql://user:pass@localhost/testdb",
            schema_name="public",
        )
        store = PostgresCheckpointStore(config)

        with patch("asyncpg.create_pool", mock_create_pool):
            await store.healthcheck()

        store._pool = mock_pg_pool  # type: ignore[assignment]
        return store


class TestPostgresCheckpointStoreSpecific:
    """Tests specific to PostgresCheckpointStore implementation."""

    @pytest.fixture
    async def store(
        self, mock_pg_pool: MockPostgresPool, mock_create_pool: AsyncMock
    ) -> PostgresCheckpointStore:
        """Create a PostgresCheckpointStore with mocked asyncpg pool."""
        config = PostgresCheckpointStoreConfig(
            dsn="postgresql://user:pass@localhost/testdb",
            schema_name="public",
        )
        store = PostgresCheckpointStore(config)

        with patch("asyncpg.create_pool", mock_create_pool):
            await store.healthcheck()

        store._pool = mock_pg_pool  # type: ignore[assignment]
        return store

    @pytest.mark.asyncio
    async def test_schema_name_configuration(self) -> None:
        """Test that schema name can be configured."""
        config = PostgresCheckpointStoreConfig(
            dsn="postgresql://user:pass@localhost/testdb",
            schema_name="custom_schema",
        )
        store = PostgresCheckpointStore(config)

        assert store.config.schema_name == "custom_schema"

    @pytest.mark.asyncio
    async def test_connection_lazy_initialization(self) -> None:
        """Test that connection pool is lazily initialized."""
        config = PostgresCheckpointStoreConfig(
            dsn="postgresql://user:pass@localhost/testdb"
        )
        store = PostgresCheckpointStore(config)

        assert store._pool is None

        config = PostgresCheckpointStoreConfig(
            dsn="postgresql://user:pass@localhost/testdb"
        )
        store = PostgresCheckpointStore(config)

        with patch("asyncpg.create_pool") as mock_create:
            mock_pool = MockPostgresPool()

            # Make create_pool return an awaitable that resolves to the mock pool
            async def async_mock():
                return mock_pool

            mock_create.return_value = async_mock()

            pool = await store._get_pool()
            assert pool is mock_pool
            assert store._pool is mock_pool

    @pytest.mark.asyncio
    async def test_concurrent_operations(
        self, store: PostgresCheckpointStore, sample_checkpoint
    ) -> None:
        """Test concurrent database operations."""
        import asyncio

        from pydantic_flow.hitl.checkpoints.interface import CheckpointEnvelope
        from pydantic_flow.hitl.checkpoints.interface import CheckpointId
        from pydantic_flow.hitl.checkpoints.interface import RunId

        run_id = RunId("concurrent_test")

        async def save_checkpoint(num: int) -> None:
            envelope = CheckpointEnvelope(
                id=CheckpointId(f"ckpt_{num}"),
                run_id=run_id,
                node_id=f"node_{num}",
                created_at=datetime.now(UTC),
                schema_version=1,
                checkpoint=sample_checkpoint,
            )
            await store.save(envelope)

        await asyncio.gather(*[save_checkpoint(i) for i in range(10)])

        query = CheckpointQuery(run_id=run_id)
        checkpoints, _ = await store.list(query)
        assert len(checkpoints) == 10

    @pytest.mark.asyncio
    async def test_schema_initialization(
        self, mock_pg_pool: MockPostgresPool, mock_create_pool: AsyncMock
    ) -> None:
        """Test that database schema is initialized on first access."""
        config = PostgresCheckpointStoreConfig(
            dsn="postgresql://user:pass@localhost/testdb"
        )
        store = PostgresCheckpointStore(config)

        with patch("asyncpg.create_pool", mock_create_pool):
            result = await store.healthcheck()

        assert result is True
