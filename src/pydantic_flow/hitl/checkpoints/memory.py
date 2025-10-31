"""In-memory checkpoint store implementation.

Thread-safe implementation for testing and development.
Not recommended for production use as data is lost on process exit.
"""

from __future__ import annotations

import asyncio

from pydantic_flow.hitl.checkpoints.base import BaseCheckpointStore
from pydantic_flow.hitl.checkpoints.interface import CheckpointConflict
from pydantic_flow.hitl.checkpoints.interface import CheckpointEnvelope
from pydantic_flow.hitl.checkpoints.interface import CheckpointId
from pydantic_flow.hitl.checkpoints.interface import CheckpointQuery
from pydantic_flow.hitl.checkpoints.interface import RunId
from pydantic_flow.hitl.checkpoints.interface import SortOrder


class InMemoryCheckpointStore(BaseCheckpointStore):
    """In-memory checkpoint store for testing and development.

    All data is stored in memory and lost when the process exits.
    Thread-safe through asyncio locks.
    """

    def __init__(self) -> None:
        """Initialize the in-memory store."""
        self._checkpoints: dict[tuple[RunId, CheckpointId], CheckpointEnvelope] = {}
        self._lock = asyncio.Lock()

    async def _do_save(
        self, envelope: CheckpointEnvelope, overwrite: bool
    ) -> CheckpointEnvelope:
        """Save checkpoint to memory.

        Args:
            envelope: The prepared checkpoint envelope with computed hash.
            overwrite: If False, raise CheckpointConflict if ID exists.

        Returns:
            The saved envelope.

        Raises:
            CheckpointConflict: If checkpoint exists and overwrite=False.

        """
        key = (envelope.run_id, envelope.id)

        async with self._lock:
            if not overwrite and key in self._checkpoints:
                msg = (
                    f"Checkpoint {envelope.id} already exists for run {envelope.run_id}"
                )
                raise CheckpointConflict(msg)

            self._checkpoints[key] = envelope
            return envelope

    async def _do_latest(
        self, run_id: RunId, node_id: str | None = None
    ) -> CheckpointEnvelope | None:
        """Get the most recent checkpoint from memory.

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

    async def _do_get(
        self, run_id: RunId, checkpoint_id: CheckpointId
    ) -> CheckpointEnvelope | None:
        """Get a specific checkpoint from memory.

        Args:
            run_id: The run identifier.
            checkpoint_id: The checkpoint identifier.

        Returns:
            The checkpoint envelope, or None if not found.

        """
        key = (run_id, checkpoint_id)
        async with self._lock:
            return self._checkpoints.get(key)

    async def _do_list(
        self, query: CheckpointQuery
    ) -> tuple[list[CheckpointEnvelope], str | None]:
        """List checkpoints from memory.

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

    async def _do_delete(self, run_id: RunId, checkpoint_id: CheckpointId) -> bool:
        """Delete a checkpoint from memory.

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

    async def _do_purge(self, run_id: RunId) -> int:
        """Delete all checkpoints for a run from memory.

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

    async def _do_healthcheck(self) -> bool:
        """Verify store health.

        Always succeeds for in-memory store.

        Returns:
            True.

        """
        return True

    async def _do_count_checkpoints(self, run_id: RunId) -> int:
        """Count checkpoints for a run in memory.

        Args:
            run_id: The run identifier.

        Returns:
            Number of checkpoints for the run.

        """
        async with self._lock:
            return sum(1 for env in self._checkpoints.values() if env.run_id == run_id)

    async def _do_get_checkpoint_history(
        self, run_id: RunId, limit: int
    ) -> list[CheckpointEnvelope]:
        """Get checkpoint history from memory, newest first.

        Args:
            run_id: The run identifier.
            limit: Maximum number of checkpoints to return.

        Returns:
            List of checkpoint envelopes, sorted by creation time (newest first).

        """
        async with self._lock:
            candidates = [
                env for env in self._checkpoints.values() if env.run_id == run_id
            ]
            candidates.sort(key=lambda e: e.created_at, reverse=True)
            return candidates[:limit]

    def __repr__(self) -> str:
        """Return a string representation of the store."""
        return f"InMemoryCheckpointStore(checkpoints={len(self._checkpoints)})"
