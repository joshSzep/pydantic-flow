"""Tests for RedisCheckpointStore implementation with mocked client."""

from __future__ import annotations

from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest

from pydantic_flow.checkpoints.redis import RedisCheckpointStore
from pydantic_flow.checkpoints.redis import RedisCheckpointStoreConfig
from tests.test_checkpoints_conformance import CheckpointStoreConformanceTests


class MockRedis:
    """Mock Redis client for testing."""

    def __init__(self) -> None:
        """Initialize mock Redis with in-memory storage."""
        self._data: dict[str, bytes] = {}
        self._sorted_sets: dict[str, dict[bytes, float]] = {}

    async def get(self, key: str) -> bytes | None:
        """Get value by key."""
        return self._data.get(key)

    async def set(
        self,
        key: str,
        value: bytes,
        nx: bool = False,
        ex: int | None = None,
    ) -> bool:
        """Set value with optional NX (not exists) flag."""
        if nx and key in self._data:
            return False
        self._data[key] = value
        return True

    async def setnx(self, key: str, value: bytes) -> bool:
        """Set if not exists."""
        if key in self._data:
            return False
        self._data[key] = value
        return True

    async def exists(self, *keys: str) -> int:
        """Check if keys exist."""
        return sum(1 for key in keys if key in self._data)

    async def expire(self, key: str, seconds: int) -> bool:
        """Set expiration (mock - we don't actually expire)."""
        return key in self._data

    async def delete(self, *keys: str) -> int:
        """Delete one or more keys."""
        count = 0
        for key in keys:
            if key in self._data:
                del self._data[key]
                count += 1
            if key in self._sorted_sets:
                del self._sorted_sets[key]
        return count

    async def zadd(
        self,
        key: str,
        mapping: dict[bytes, float],
        nx: bool = False,
    ) -> int:
        """Add members to sorted set."""
        if key not in self._sorted_sets:
            self._sorted_sets[key] = {}

        count = 0
        for member, score in mapping.items():
            if nx and member in self._sorted_sets[key]:
                continue
            # Ensure member is bytes and score is float
            member_bytes = member if isinstance(member, bytes) else str(member).encode()
            self._sorted_sets[key][member_bytes] = float(score)
            count += 1
        return count

    async def zrange(
        self,
        key: str,
        start: int,
        stop: int,
        desc: bool = False,
        withscores: bool = False,
    ) -> list[bytes] | list[tuple[bytes, float]]:
        """Get range from sorted set."""
        if key not in self._sorted_sets:
            return []

        items = sorted(
            self._sorted_sets[key].items(),
            key=lambda x: x[1],
            reverse=desc,
        )

        if start < 0:
            start = len(items) + start
        if stop < 0:
            stop = len(items) + stop

        items = items[start : stop + 1]

        if withscores:
            return items
        return [member for member, _ in items]

    async def zrevrange(
        self,
        key: str,
        start: int,
        stop: int,
        withscores: bool = False,
    ) -> list[bytes] | list[tuple[bytes, float]]:
        """Get range from sorted set in reverse order (highest to lowest score)."""
        return await self.zrange(key, start, stop, desc=True, withscores=withscores)

    async def zrangebyscore(  # noqa: PLR0913
        self,
        key: str,
        min_score: float,
        max_score: float,
        start: int | None = None,
        num: int | None = None,
        withscores: bool = False,
    ) -> list[bytes] | list[tuple[bytes, float]]:
        """Get range by score from sorted set."""
        if key not in self._sorted_sets:
            return []

        # Ensure scores are compared as floats
        items = [
            (member, float(score))
            for member, score in self._sorted_sets[key].items()
            if float(min_score) <= float(score) <= float(max_score)
        ]
        items.sort(key=lambda x: x[1])

        if start is not None and num is not None:
            items = items[start : start + num]

        if withscores:
            return items
        return [member for member, _ in items]

    async def zrevrangebyscore(  # noqa: PLR0913
        self,
        key: str,
        max_score: float,
        min_score: float,
        start: int | None = None,
        num: int | None = None,
        withscores: bool = False,
    ) -> list[bytes] | list[tuple[bytes, float]]:
        """Get range by score (descending) from sorted set."""
        if key not in self._sorted_sets:
            return []

        # Ensure scores are compared as floats
        items = [
            (member, float(score))
            for member, score in self._sorted_sets[key].items()
            if float(min_score) <= float(score) <= float(max_score)
        ]
        items.sort(key=lambda x: x[1], reverse=True)

        if start is not None and num is not None:
            items = items[start : start + num]

        if withscores:
            return items
        return [member for member, _ in items]

    async def zrem(self, key: str, *members: bytes) -> int:
        """Remove members from sorted set."""
        if key not in self._sorted_sets:
            return 0

        count = 0
        for member in members:
            if member in self._sorted_sets[key]:
                del self._sorted_sets[key][member]
                count += 1
        return count

    async def keys(self, pattern: str) -> list[str]:
        """Get keys matching pattern."""
        import re

        regex_pattern = pattern.replace("*", ".*").replace("?", ".")
        regex = re.compile(regex_pattern)
        return [key for key in self._data if regex.match(key)]

    async def scan(
        self, cursor: int, match: str | None = None, count: int | None = None
    ) -> tuple[int, list[str]]:
        """Scan keys matching pattern (simplified mock - returns all at once)."""
        if cursor != 0:
            # Simplified: return empty on subsequent calls
            return 0, []

        import re

        if match:
            regex_pattern = match.replace("*", ".*").replace("?", ".")
            regex = re.compile(regex_pattern)
            keys = [key for key in self._data if regex.match(key)]
        else:
            keys = list(self._data.keys())

        # Return all keys with cursor=0 to indicate completion
        return 0, keys

    def ping(self) -> bool:
        """Ping command (synchronous to avoid coroutine warnings)."""
        return True


@pytest.fixture
def mock_redis() -> MockRedis:
    """Create a mock Redis client."""
    return MockRedis()


@pytest.fixture
def mock_redis_connection(mock_redis: MockRedis) -> AsyncMock:
    """Create async mock that returns our mock Redis."""
    mock = AsyncMock(return_value=mock_redis)
    return mock


class TestRedisCheckpointStoreConformance(CheckpointStoreConformanceTests):
    """Conformance tests for RedisCheckpointStore with mocked Redis."""

    @pytest.fixture
    async def store(
        self, mock_redis: MockRedis, mock_redis_connection: AsyncMock
    ) -> RedisCheckpointStore:
        """Create a RedisCheckpointStore with mocked Redis client."""
        config = RedisCheckpointStoreConfig(
            redis_url="redis://localhost:6379/0",
            key_prefix="test:ckpt",
        )
        store = RedisCheckpointStore(config)

        with patch("pydantic_flow.checkpoints.redis.from_url", mock_redis_connection):
            await store.healthcheck()

        store._redis = mock_redis  # type: ignore[assignment]
        return store


class TestRedisCheckpointStoreSpecific:
    """Tests specific to RedisCheckpointStore implementation."""

    @pytest.fixture
    async def store(
        self, mock_redis: MockRedis, mock_redis_connection: AsyncMock
    ) -> RedisCheckpointStore:
        """Create a RedisCheckpointStore with mocked Redis client."""
        config = RedisCheckpointStoreConfig(
            redis_url="redis://localhost:6379/0",
            key_prefix="test:ckpt",
        )
        store = RedisCheckpointStore(config)

        with patch("pydantic_flow.checkpoints.redis.from_url", mock_redis_connection):
            await store.healthcheck()

        store._redis = mock_redis  # type: ignore[assignment]
        return store

    @pytest.mark.asyncio
    async def test_key_prefix_applied(
        self, store: RedisCheckpointStore, sample_envelope
    ) -> None:
        """Test that key prefix is correctly applied."""
        await store.save(sample_envelope)

        checkpoint_key = store._checkpoint_key(
            sample_envelope.run_id, sample_envelope.id
        )
        assert checkpoint_key.startswith("test:ckpt:")

    @pytest.mark.asyncio
    async def test_ttl_configuration(self, mock_redis: MockRedis) -> None:
        """Test that TTL can be configured."""
        config = RedisCheckpointStoreConfig(
            redis_url="redis://localhost:6379/0",
            ttl_seconds=3600,
        )
        store = RedisCheckpointStore(config)
        store._redis = mock_redis  # type: ignore[assignment]

        assert store.config.ttl_seconds == 3600

    @pytest.mark.asyncio
    async def test_sorted_set_indexing(
        self, store: RedisCheckpointStore, sample_envelope
    ) -> None:
        """Test that sorted sets are used for indexing."""
        await store.save(sample_envelope)

        mock_redis: MockRedis = store._redis  # type: ignore[assignment]
        index_key = store._index_key(sample_envelope.run_id)

        assert index_key in mock_redis._sorted_sets
        assert len(mock_redis._sorted_sets[index_key]) == 1

    @pytest.mark.asyncio
    async def test_node_specific_index(
        self, store: RedisCheckpointStore, sample_checkpoint
    ) -> None:
        """Test that node-specific indexes are created."""
        from datetime import UTC
        from datetime import datetime

        from pydantic_flow.checkpoints.interface import CheckpointEnvelope
        from pydantic_flow.checkpoints.interface import CheckpointId
        from pydantic_flow.checkpoints.interface import RunId

        envelope = CheckpointEnvelope(
            id=CheckpointId("test_ckpt"),
            run_id=RunId("test_run"),
            node_id="specific_node",
            created_at=datetime.now(UTC),
            schema_version=1,
            checkpoint=sample_checkpoint,
        )

        await store.save(envelope)

        mock_redis: MockRedis = store._redis  # type: ignore[assignment]
        node_index_key = store._index_key(envelope.run_id, envelope.node_id)

        assert node_index_key in mock_redis._sorted_sets

    @pytest.mark.asyncio
    async def test_connection_lazy_initialization(self) -> None:
        """Test that Redis connection is lazily initialized."""
        config = RedisCheckpointStoreConfig(redis_url="redis://localhost:6379/0")
        store = RedisCheckpointStore(config)

        assert store._redis is None

        with patch("pydantic_flow.checkpoints.redis.from_url") as mock_from_url:
            mock_client = MockRedis()

            # Make from_url return an awaitable that resolves to the mock client
            async def async_mock():
                return mock_client

            mock_from_url.return_value = async_mock()

            redis_client = await store._get_redis()
            assert redis_client is mock_client
            assert store._redis is mock_client
