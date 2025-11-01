"""State reconstruction with bounded complexity.

This module provides utilities for reconstructing state from checkpoints
with O(10) bounded complexity for forward time-travel and efficient
backward time-travel.
"""

from __future__ import annotations

from pydantic import BaseModel

from pydantic_flow.checkpoints.delta import DeltaComputer
from pydantic_flow.checkpoints.interface import CheckpointStorageBackend
from pydantic_flow.checkpoints.types import RunId


class StateReconstructor:
    """Reconstruct state with bounded complexity.

    Provides efficient state reconstruction for time-travel debugging with:
    - Forward reconstruction: O(10) bounded (uses delta chain from base)
    - Backward reconstruction: O(10) bounded (uses reverse deltas)
    - Batch fetching: Single query for snapshot range

    The design stores full state every 10 waves, with deltas in between,
    enabling fast state reconstruction at any wave number.
    """

    def __init__(self, backend: CheckpointStorageBackend):
        """Initialize state reconstructor.

        Args:
            backend: Storage backend for fetching snapshots.

        """
        self.backend = backend

    async def reconstruct_state_at(
        self,
        run_id: RunId,
        wave_number: int,
    ) -> dict[str, BaseModel]:
        """Reconstruct state at specific wave (forward time-travel).

        Uses forward deltas from the most recent full snapshot (stored every
        10 waves). Complexity is O(10) bounded since we apply at most 10 deltas.

        Args:
            run_id: Flow execution run identifier.
            wave_number: Target wave number to reconstruct.

        Returns:
            Complete reconstructed state dictionary.

        Raises:
            ValueError: If no snapshots found for run.
            RuntimeError: If state reconstruction fails.

        Example:
            >>> reconstructor = StateReconstructor(backend)
            >>> state = await reconstructor.reconstruct_state_at("run123", 47)
            >>> # Reconstructs from wave 40 (full) + deltas 41-47

        """
        # Find base wave (every 10th wave has full state)
        base_wave = (wave_number // 10) * 10

        # Batch fetch all snapshots from base to target
        snapshots = await self.backend.get_snapshots_range(
            run_id, base_wave, wave_number, order="ASC"
        )

        if not snapshots:
            msg = f"No snapshots found for run {run_id}"
            raise ValueError(msg)

        # Start with full state from base
        first_snapshot = snapshots[0]
        if first_snapshot.full_state is None:
            msg = f"Base snapshot at wave {base_wave} missing full state"
            raise RuntimeError(msg)

        state = dict(first_snapshot.full_state)

        # Apply forward deltas sequentially
        for snapshot in snapshots[1:]:
            if snapshot.forward_delta:
                state = DeltaComputer.apply_forward_delta(state, snapshot.forward_delta)

        return state

    async def rewind_state_to(
        self,
        run_id: RunId,
        from_wave: int,
        to_wave: int,
    ) -> dict[str, BaseModel]:
        """Rewind state backwards (backward time-travel).

        Uses reverse deltas to efficiently move backward in time. Complexity
        is O(from_wave - to_wave) bounded by number of steps.

        Args:
            run_id: Flow execution run identifier.
            from_wave: Starting wave number (current state).
            to_wave: Target wave number to rewind to (must be < from_wave).

        Returns:
            Complete reconstructed state at target wave.

        Raises:
            ValueError: If to_wave >= from_wave or snapshots not found.
            RuntimeError: If state reconstruction fails.

        Example:
            >>> state = await reconstructor.rewind_state_to(
            ...     "run123", from_wave=47, to_wave=42
            ... )
            >>> # Rewinds using reverse deltas 47->46->...->42

        """
        if to_wave >= from_wave:
            msg = f"Cannot rewind forward: to_wave={to_wave} >= from_wave={from_wave}"
            raise ValueError(msg)

        # Batch fetch snapshots in reverse order
        snapshots = await self.backend.get_snapshots_range(
            run_id, to_wave, from_wave, order="DESC"
        )

        if not snapshots:
            msg = (
                f"No snapshots found for run {run_id} in range [{to_wave}, {from_wave}]"
            )
            raise ValueError(msg)

        # Start with full state or reconstruct from most recent full state
        start_snapshot = snapshots[0]  # This is from_wave (DESC order)

        if start_snapshot.full_state:
            # If starting point has full state, use it
            state = dict(start_snapshot.full_state)
        else:
            # Otherwise reconstruct forward from base
            state = await self.reconstruct_state_at(run_id, from_wave)

        # Apply reverse deltas backwards
        for snapshot in snapshots:
            if snapshot.wave_number == from_wave:
                continue  # Skip starting snapshot

            if snapshot.reverse_delta:
                state = DeltaComputer.apply_reverse_delta(state, snapshot.reverse_delta)

            if snapshot.wave_number == to_wave:
                break

        return state

    async def get_state_hash_at_wave(
        self,
        run_id: RunId,
        wave_number: int,
    ) -> str:
        """Get state hash at specific wave without reconstruction.

        Efficient way to check state integrity or compare states without
        reconstructing the full state.

        Args:
            run_id: Flow execution run identifier.
            wave_number: Wave number to query.

        Returns:
            SHA-256 hash of state at that wave.

        Raises:
            ValueError: If snapshot not found.

        """
        snapshot = await self.backend.get_state_snapshot(run_id, wave_number)

        if not snapshot:
            msg = f"No snapshot found for run {run_id} at wave {wave_number}"
            raise ValueError(msg)

        return snapshot.state_hash

    async def validate_state_chain(
        self,
        run_id: RunId,
        start_wave: int,
        end_wave: int,
    ) -> bool:
        """Validate state chain integrity.

        Verifies that forward and reverse deltas are consistent by:
        1. Reconstructing state forward using deltas
        2. Computing hash and comparing with stored hash

        Args:
            run_id: Flow execution run identifier.
            start_wave: Starting wave number.
            end_wave: Ending wave number.

        Returns:
            True if chain is valid, False if corruption detected.

        """
        snapshots = await self.backend.get_snapshots_range(
            run_id, start_wave, end_wave, order="ASC"
        )

        if not snapshots:
            return False

        # Start with full state
        first = snapshots[0]
        if not first.full_state:
            return False

        state = dict(first.full_state)
        computed_hash = first.compute_state_hash(state)

        if computed_hash != first.state_hash:
            return False

        # Validate each subsequent snapshot
        for snapshot in snapshots[1:]:
            if snapshot.forward_delta:
                state = DeltaComputer.apply_forward_delta(state, snapshot.forward_delta)

            computed_hash = snapshot.compute_state_hash(state)
            if computed_hash != snapshot.state_hash:
                return False

        return True
