"""Checkpoint inspection - read-only queries and traversal.

This module provides the data access layer for checkpoint debugging.
Pure read operations with no side effects.
"""

from typing import Any

from pydantic_flow.checkpoints.interface import CheckpointStorageBackend
from pydantic_flow.checkpoints.reconstructor import StateReconstructor
from pydantic_flow.checkpoints.types import ExecutionTrace
from pydantic_flow.checkpoints.types import RunId
from pydantic_flow.checkpoints.types import RunMetadata
from pydantic_flow.checkpoints.types import StateSnapshot


class CheckpointInspector:
    """Read-only checkpoint data access layer.

    Provides methods to query and traverse checkpoint history without
    modifying any state. All methods are idempotent.
    """

    def __init__(self, backend: CheckpointStorageBackend) -> None:
        """Initialize inspector with storage backend.

        Args:
            backend: Storage backend for checkpoint data

        """
        self.backend = backend
        self.reconstructor = StateReconstructor(backend)

    async def list_runs(
        self,
        status: RunMetadata.Status | None = None,
        limit: int = 50,
    ) -> list[RunMetadata]:
        """List execution runs.

        Args:
            status: Optional status filter
            limit: Maximum number of runs to return

        Returns:
            List of run metadata sorted by start time (most recent first)

        """
        return await self.backend.list_runs(limit=limit)

    async def get_run(self, run_id: RunId) -> RunMetadata | None:
        """Get metadata for a specific run.

        Args:
            run_id: Run identifier

        Returns:
            Run metadata if found, None otherwise

        """
        return await self.backend.get_run_metadata(run_id)

    async def get_wave_timeline(
        self, run_id: RunId, start_wave: int = 0, end_wave: int | None = None
    ) -> list[StateSnapshot]:
        """Get timeline of waves for a run.

        Args:
            run_id: Run identifier
            start_wave: Starting wave number (default: 0)
            end_wave: Ending wave number (default: all remaining waves)

        Returns:
            List of state snapshots in chronological order

        """
        if end_wave is None:
            end_wave = 999999  # Fetch all

        snapshots = await self.backend.get_snapshots_range(
            run_id=run_id,
            start_wave=start_wave,
            end_wave=end_wave,
        )
        return snapshots

    async def get_wave_snapshot(self, run_id: RunId, wave: int) -> StateSnapshot | None:
        """Get snapshot for a specific wave.

        Args:
            run_id: Run identifier
            wave: Wave number

        Returns:
            State snapshot if found, None otherwise

        """
        snapshots = await self.backend.get_snapshots_range(
            run_id=run_id,
            start_wave=wave,
            end_wave=wave,
        )
        return snapshots[0] if snapshots else None

    async def get_wave_trace(self, run_id: RunId, wave: int) -> ExecutionTrace | None:
        """Get execution trace for a specific wave.

        Args:
            run_id: Run identifier
            wave: Wave number

        Returns:
            Execution trace if found, None otherwise

        """
        return await self.backend.get_trace(run_id=run_id, wave_number=wave)

    async def reconstruct_state(self, run_id: RunId, wave: int) -> dict[str, Any]:
        """Reconstruct full state at a specific wave.

        Applies deltas forward from last full snapshot to reconstruct complete state.

        Args:
            run_id: Run identifier
            wave: Target wave number

        Returns:
            Complete state dictionary at target wave

        Raises:
            ValueError: If wave not found or reconstruction fails

        """
        # First try to get the snapshot
        snapshot = await self.get_wave_snapshot(run_id, wave)
        if snapshot is None:
            msg = f"No snapshot found for {run_id}/wave={wave}"
            raise ValueError(msg)

        # If it has full state, return directly
        if snapshot.full_state is not None:
            return snapshot.full_state

        # Otherwise reconstruct from deltas
        return await self.reconstructor.reconstruct_state_at(run_id, wave)

    async def get_latest_wave(self, run_id: RunId) -> int | None:
        """Get the latest wave number for a run.

        Args:
            run_id: Run identifier

        Returns:
            Latest wave number, or None if run has no waves

        """
        metadata = await self.get_run(run_id)
        return (
            metadata.total_waves - 1 if metadata and metadata.total_waves > 0 else None
        )
