"""Redis checkpoint store implementation.

Stores checkpoints as JSON blobs with sorted sets for indexing.
"""

from __future__ import annotations

from pydantic import BaseModel
from redis.asyncio import Redis
from redis.asyncio import from_url

from pydantic_flow.hitl.checkpoints.base import BaseCheckpointStore
from pydantic_flow.hitl.checkpoints.interface import CheckpointConflict
from pydantic_flow.hitl.checkpoints.interface import CheckpointEnvelope
from pydantic_flow.hitl.checkpoints.interface import CheckpointId
from pydantic_flow.hitl.checkpoints.interface import CheckpointQuery
from pydantic_flow.hitl.checkpoints.interface import RunId
from pydantic_flow.hitl.checkpoints.interface import SortOrder
from pydantic_flow.hitl.checkpoints.serde import deserialize_checkpoint
from pydantic_flow.hitl.checkpoints.serde import serialize_checkpoint


class RedisCheckpointStoreConfig(BaseModel):
    """Configuration for Redis checkpoint store.

    Attributes:
        redis_url: Redis connection URL.
        key_prefix: Prefix for all Redis keys.
        ttl_seconds: Optional TTL for checkpoint keys.

    """

    redis_url: str = "redis://localhost:6379/0"
    key_prefix: str = "pf:ckpt"
    ttl_seconds: int | None = None


class RedisCheckpointStore(BaseCheckpointStore):
    """Redis-based checkpoint store with sorted set indexing."""

    def __init__(self, config: RedisCheckpointStoreConfig) -> None:
        """Initialize the Redis store.

        Args:
            config: Store configuration.

        """
        self.config = config
        self._redis: Redis | None = None  # type: ignore[type-arg]

    async def _get_redis(self) -> Redis:  # type: ignore[type-arg]
        """Get or create Redis connection."""
        if self._redis is None:
            self._redis = await from_url(
                self.config.redis_url,
                encoding="utf-8",
                decode_responses=False,
            )
        return self._redis

    def _checkpoint_key(self, run_id: RunId, checkpoint_id: CheckpointId) -> str:
        """Generate Redis key for checkpoint blob."""
        return f"{self.config.key_prefix}:{run_id}:{checkpoint_id}"

    def _index_key(self, run_id: RunId, node_id: str | None = None) -> str:
        """Generate Redis key for index sorted set."""
        if node_id is not None:
            return f"{self.config.key_prefix}:idx:{run_id}:{node_id}"
        return f"{self.config.key_prefix}:idx:{run_id}"

    async def _do_save(
        self, envelope: CheckpointEnvelope, overwrite: bool
    ) -> CheckpointEnvelope:
        """Save checkpoint to Redis.

        Args:
            envelope: The prepared checkpoint envelope with computed hash.
            overwrite: If False, raise CheckpointConflict if key exists.

        Returns:
            The saved envelope.

        Raises:
            CheckpointConflict: If checkpoint exists and overwrite=False.

        """
        redis = await self._get_redis()

        json_str = serialize_checkpoint(envelope)
        checkpoint_key = self._checkpoint_key(envelope.run_id, envelope.id)

        if not overwrite:
            exists = await redis.exists(checkpoint_key)
            if exists:
                msg = (
                    f"Checkpoint {envelope.id} already exists for run {envelope.run_id}"
                )
                raise CheckpointConflict(msg)

            set_result = await redis.setnx(checkpoint_key, json_str.encode("utf-8"))
            if not set_result:
                msg = (
                    f"Checkpoint {envelope.id} already exists for run {envelope.run_id}"
                )
                raise CheckpointConflict(msg)
        else:
            await redis.set(checkpoint_key, json_str.encode("utf-8"))

        if self.config.ttl_seconds is not None:
            await redis.expire(checkpoint_key, self.config.ttl_seconds)

        score = envelope.created_at.timestamp()
        run_index = self._index_key(envelope.run_id)
        await redis.zadd(run_index, {envelope.id: score})

        node_index = None
        if envelope.node_id is not None:
            node_index = self._index_key(envelope.run_id, envelope.node_id)
            await redis.zadd(node_index, {envelope.id: score})

        if self.config.ttl_seconds is not None:
            await redis.expire(run_index, self.config.ttl_seconds)
            if node_index is not None:
                await redis.expire(node_index, self.config.ttl_seconds)

        return envelope

    async def _do_latest(
        self, run_id: RunId, node_id: str | None = None
    ) -> CheckpointEnvelope | None:
        """Get the most recent checkpoint from Redis.

        Args:
            run_id: The run to query.
            node_id: Optional node filter.

        Returns:
            The latest checkpoint envelope, or None if not found.

        """
        redis = await self._get_redis()
        index_key = self._index_key(run_id, node_id)

        result = await redis.zrevrange(index_key, 0, 0)
        if not result:
            return None

        checkpoint_id = CheckpointId(result[0].decode("utf-8"))
        checkpoint_key = self._checkpoint_key(run_id, checkpoint_id)

        json_bytes = await redis.get(checkpoint_key)
        if json_bytes is None:
            return None

        return deserialize_checkpoint(json_bytes.decode("utf-8"))

    async def _do_get(
        self, run_id: RunId, checkpoint_id: CheckpointId
    ) -> CheckpointEnvelope | None:
        """Get a specific checkpoint by ID from Redis.

        Args:
            run_id: The run identifier.
            checkpoint_id: The checkpoint identifier.

        Returns:
            The checkpoint envelope, or None if not found.

        """
        redis = await self._get_redis()
        checkpoint_key = self._checkpoint_key(run_id, checkpoint_id)

        json_bytes = await redis.get(checkpoint_key)
        if json_bytes is None:
            return None

        return deserialize_checkpoint(json_bytes.decode("utf-8"))

    async def _do_list(
        self, query: CheckpointQuery
    ) -> tuple[list[CheckpointEnvelope], str | None]:
        """List checkpoints matching query criteria from Redis.

        Args:
            query: Query parameters for filtering and pagination.

        Returns:
            Tuple of (list of checkpoint envelopes, next cursor or None).

        """
        if query.run_id is None:
            return [], None

        redis = await self._get_redis()
        index_key = self._index_key(query.run_id, query.node_id)

        min_score = query.since.timestamp() if query.since is not None else "-inf"
        max_score = query.until.timestamp() if query.until is not None else "+inf"

        cursor_offset = 0
        if query.cursor is not None:
            try:
                cursor_offset = int(query.cursor)
            except ValueError:
                cursor_offset = 0

        if query.sort_order == SortOrder.DESC:
            all_ids = await redis.zrevrangebyscore(
                index_key,
                max_score,
                min_score,
                start=cursor_offset,
                num=query.limit + 1,
            )
        else:
            all_ids = await redis.zrangebyscore(
                index_key,
                min_score,
                max_score,
                start=cursor_offset,
                num=query.limit + 1,
            )

        checkpoint_ids = [CheckpointId(cid.decode("utf-8")) for cid in all_ids]

        envelopes: list[CheckpointEnvelope] = []
        for checkpoint_id in checkpoint_ids[: query.limit]:
            checkpoint_key = self._checkpoint_key(query.run_id, checkpoint_id)
            json_bytes = await redis.get(checkpoint_key)
            if json_bytes is not None:
                envelopes.append(deserialize_checkpoint(json_bytes.decode("utf-8")))

        next_cursor = None
        if len(checkpoint_ids) > query.limit:
            next_cursor = str(cursor_offset + query.limit)

        return envelopes, next_cursor

    async def _do_delete(self, run_id: RunId, checkpoint_id: CheckpointId) -> bool:
        """Delete a specific checkpoint from Redis.

        Args:
            run_id: The run identifier.
            checkpoint_id: The checkpoint identifier.

        Returns:
            True if checkpoint was deleted, False if it didn't exist.

        """
        redis = await self._get_redis()
        checkpoint_key = self._checkpoint_key(run_id, checkpoint_id)

        envelope_json = await redis.get(checkpoint_key)
        if envelope_json is None:
            return False

        envelope = deserialize_checkpoint(envelope_json.decode("utf-8"))

        await redis.delete(checkpoint_key)

        run_index = self._index_key(run_id)
        await redis.zrem(run_index, checkpoint_id)

        if envelope.node_id is not None:
            node_index = self._index_key(run_id, envelope.node_id)
            await redis.zrem(node_index, checkpoint_id)

        return True

    async def _do_purge(self, run_id: RunId) -> int:
        """Delete all checkpoints for a run from Redis.

        Args:
            run_id: The run identifier.

        Returns:
            Number of checkpoints deleted.

        """
        redis = await self._get_redis()
        run_index = self._index_key(run_id)

        all_ids = await redis.zrange(run_index, 0, -1)
        checkpoint_ids = [CheckpointId(cid.decode("utf-8")) for cid in all_ids]

        deleted_count = 0
        for checkpoint_id in checkpoint_ids:
            checkpoint_key = self._checkpoint_key(run_id, checkpoint_id)
            result = await redis.delete(checkpoint_key)
            deleted_count += result

        await redis.delete(run_index)

        pattern = self._index_key(run_id, "*")
        cursor = 0
        while True:
            cursor, keys = await redis.scan(cursor, match=pattern, count=100)
            if keys:
                await redis.delete(*keys)
            if cursor == 0:
                break

        return deleted_count

    async def _do_healthcheck(self) -> bool:
        """Verify Redis connectivity.

        Returns:
            True if Redis is healthy.

        Raises:
            Exception: If Redis ping fails.

        """
        redis = await self._get_redis()
        result = redis.ping()
        if not result:
            msg = "Redis ping returned False"
            raise RuntimeError(msg)
        return True

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None

    def __repr__(self) -> str:
        """Return a string representation of the store."""
        return (
            f"RedisCheckpointStore(redis_url=<redacted>, "
            f"key_prefix={self.config.key_prefix})"
        )
