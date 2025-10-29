"""In-memory checkpoint store implementation.

Thread-safe implementation for testing and development.
Not recommended for production use as data is lost on process exit.
"""

from __future__ import annotations

import asyncio

from pydantic_flow.checkpoints.interface import CheckpointConflict
from pydantic_flow.checkpoints.interface import CheckpointEnvelope
from pydantic_flow.checkpoints.interface import CheckpointId
from pydantic_flow.checkpoints.interface import CheckpointQuery
from pydantic_flow.checkpoints.interface import RunId
from pydantic_flow.checkpoints.interface import SortOrder
from pydantic_flow.checkpoints.serde import compute_content_hash


class InMemoryCheckpointStore:
    """In-memory checkpoint store for testing and development.

    All data is stored in memory and lost when the process exits.
    Thread-safe through asyncio locks.
    """

    def __init__(self) -> None:
        """Initialize the in-memory store."""
        self._checkpoints: dict[tuple[RunId, CheckpointId], CheckpointEnvelope] = {}
        self._lock = asyncio.Lock()

    async def save(
        self, envelope: CheckpointEnvelope, *, overwrite: bool = False
    ) -> CheckpointEnvelope:
        """Save a checkpoint to memory.

        Args:
            envelope: The checkpoint envelope to save.
            overwrite: If False, raise CheckpointConflict if ID exists.

        Returns:
            The saved envelope with computed content hash.

        Raises:
            CheckpointConflict: If checkpoint ID exists and overwrite=False.

        """
        key = (envelope.run_id, envelope.id)

        async with self._lock:
            if not overwrite and key in self._checkpoints:
                msg = (
                    f"Checkpoint {envelope.id} already exists for run {envelope.run_id}"
                )
                raise CheckpointConflict(msg)

            envelope_copy = envelope.model_copy(deep=True)
            if envelope_copy.content_hash is None:
                envelope_copy.content_hash = compute_content_hash(envelope_copy)

            self._checkpoints[key] = envelope_copy
            return envelope_copy

    async def latest(
        self, run_id: RunId, node_id: str | None = None
    ) -> CheckpointEnvelope | None:
        """Get the most recent checkpoint for a run.

        Args:
            run_id: The run to query.
            node_id: Optional node filter.

        Returns:
            The latest checkpoint envelope, or None if not found.

        """
        async with self._lock:
            candidates = [
                env
                for (r_id, _), env in self._checkpoints.items()
                if r_id == run_id and (node_id is None or env.node_id == node_id)
            ]

            if not candidates:
                return None

            return max(candidates, key=lambda e: e.created_at)

    async def get(
        self, run_id: RunId, checkpoint_id: CheckpointId
    ) -> CheckpointEnvelope | None:
        """Get a specific checkpoint by ID.

        Args:
            run_id: The run identifier.
            checkpoint_id: The checkpoint identifier.

        Returns:
            The checkpoint envelope, or None if not found.

        """
        key = (run_id, checkpoint_id)
        async with self._lock:
            return self._checkpoints.get(key)

    async def list(
        self, query: CheckpointQuery
    ) -> tuple[list[CheckpointEnvelope], str | None]:
        """List checkpoints matching query criteria.

        Args:
            query: Query parameters for filtering and pagination.

        Returns:
            Tuple of (list of checkpoint envelopes, next cursor or None).

        """
        async with self._lock:
            results = list(self._checkpoints.values())

            if query.run_id is not None:
                results = [e for e in results if e.run_id == query.run_id]

            if query.node_id is not None:
                results = [e for e in results if e.node_id == query.node_id]

            if query.since is not None:
                results = [e for e in results if e.created_at >= query.since]

            if query.until is not None:
                results = [e for e in results if e.created_at <= query.until]

            reverse = query.sort_order == SortOrder.DESC
            results.sort(key=lambda e: e.created_at, reverse=reverse)

            cursor_idx = 0
            if query.cursor is not None:
                try:
                    cursor_idx = int(query.cursor)
                except ValueError:
                    cursor_idx = 0

            page = results[cursor_idx : cursor_idx + query.limit]

            next_cursor = None
            if cursor_idx + query.limit < len(results):
                next_cursor = str(cursor_idx + query.limit)

            return page, next_cursor

    async def delete(self, run_id: RunId, checkpoint_id: CheckpointId) -> bool:
        """Delete a specific checkpoint.

        Args:
            run_id: The run identifier.
            checkpoint_id: The checkpoint identifier.

        Returns:
            True if checkpoint was deleted, False if it didn't exist.

        """
        key = (run_id, checkpoint_id)
        async with self._lock:
            if key in self._checkpoints:
                del self._checkpoints[key]
                return True
            return False

    async def purge(self, run_id: RunId) -> int:
        """Delete all checkpoints for a run.

        Args:
            run_id: The run identifier.

        Returns:
            Number of checkpoints deleted.

        """
        async with self._lock:
            keys_to_delete = [
                key for key, env in self._checkpoints.items() if env.run_id == run_id
            ]
            for key in keys_to_delete:
                del self._checkpoints[key]
            return len(keys_to_delete)

    async def healthcheck(self) -> bool:
        """Verify store connectivity.

        Always succeeds for in-memory store.
        """
        return True

    def __repr__(self) -> str:
        """Return a string representation of the store."""
        return f"InMemoryCheckpointStore(checkpoints={len(self._checkpoints)})"
