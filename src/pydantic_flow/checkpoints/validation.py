"""Validation utilities for checkpoint-trace bidirectional references.

This module provides functions to validate and maintain integrity between
checkpoints and traces, ensuring that bidirectional references are correct.
"""

from __future__ import annotations

from pydantic_flow.checkpoints.interface import CheckpointStorageBackend
from pydantic_flow.checkpoints.types import ExecutionTrace
from pydantic_flow.checkpoints.types import RunId


class CheckpointIntegrityError(Exception):
    """Raised when checkpoint-trace integrity is violated."""

    pass


async def validate_and_save_trace(
    backend: CheckpointStorageBackend,
    trace: ExecutionTrace,
    *,
    update_checkpoint: bool = True,
) -> None:
    """Validate trace references checkpoint and optionally update bidirectional link.

    Ensures that:
    1. The trace's checkpoint_snapshot_id references a valid checkpoint
    2. Optionally updates the checkpoint's trace_id to point back to this trace

    Args:
        backend: Storage backend for validation and updates.
        trace: Execution trace to validate and save.
        update_checkpoint: If True, update checkpoint with trace_id.

    Raises:
        CheckpointIntegrityError: If checkpoint reference is invalid.
        Exception: If storage operations fail.

    Example:
        >>> await validate_and_save_trace(
        ...     backend,
        ...     trace,
        ...     update_checkpoint=True
        ... )

    """
    # Validate checkpoint exists
    checkpoint = await backend.get_state_snapshot(trace.run_id, trace.wave_number)

    if not checkpoint:
        msg = (
            f"Trace references non-existent checkpoint: "
            f"run_id={trace.run_id}, wave={trace.wave_number}"
        )
        raise CheckpointIntegrityError(msg)

    if checkpoint.snapshot_id != trace.checkpoint_snapshot_id:
        msg = (
            f"Trace checkpoint_snapshot_id mismatch: "
            f"expected {checkpoint.snapshot_id}, "
            f"got {trace.checkpoint_snapshot_id}"
        )
        raise CheckpointIntegrityError(msg)

    # Save trace (backend enforces foreign key constraint)
    await backend.save_trace(trace)

    # Update checkpoint with trace reference
    if update_checkpoint:
        checkpoint.trace_id = trace.trace_id
        await backend.update_state_snapshot(checkpoint)


async def validate_checkpoint_integrity(
    backend: CheckpointStorageBackend,
    run_id: RunId,
    wave_number: int,
) -> bool:
    """Validate checkpoint-trace bidirectional integrity.

    Checks that:
    1. Checkpoint exists
    2. If checkpoint has trace_id, the trace exists and references back
    3. If trace exists, it references this checkpoint

    Args:
        backend: Storage backend for validation.
        run_id: Run identifier.
        wave_number: Wave number to validate.

    Returns:
        True if integrity is valid, False otherwise.

    Example:
        >>> is_valid = await validate_checkpoint_integrity(
        ...     backend, "run123", 5
        ... )

    """
    # Get checkpoint
    checkpoint = await backend.get_state_snapshot(run_id, wave_number)
    if not checkpoint:
        return False

    # Get trace (if exists)
    trace = await backend.get_trace(run_id, wave_number)

    # Check consistency
    has_checkpoint_trace = checkpoint.trace_id is not None
    has_trace = trace is not None

    # Dangling reference: checkpoint has trace_id but trace doesn't exist
    if has_checkpoint_trace and not has_trace:
        return False

    # Both exist: check bidirectional references
    if (
        has_checkpoint_trace
        and has_trace
        and trace is not None
        and (
            trace.trace_id != checkpoint.trace_id
            or trace.checkpoint_snapshot_id != checkpoint.snapshot_id
        )
    ):
        return False

    # Trace exists without checkpoint reference - should point to this checkpoint
    return not (
        has_trace
        and trace is not None
        and not has_checkpoint_trace
        and trace.checkpoint_snapshot_id != checkpoint.snapshot_id
    )


async def repair_bidirectional_references(
    backend: CheckpointStorageBackend,
    run_id: RunId,
    *,
    start_wave: int = 0,
    end_wave: int | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    """Repair broken bidirectional references in a run.

    Scans checkpoints and traces to fix bidirectional references:
    - Updates checkpoint.trace_id to match existing traces
    - Reports orphaned traces (no checkpoint)
    - Reports orphaned checkpoints (trace_id but no trace)

    Args:
        backend: Storage backend for repairs.
        run_id: Run identifier to repair.
        start_wave: Starting wave number.
        end_wave: Ending wave number (None for all).
        dry_run: If True, report issues without fixing.

    Returns:
        Dictionary with repair statistics:
            - fixed_checkpoints: Number of checkpoints updated
            - orphaned_traces: Number of traces with invalid checkpoint refs
            - dangling_checkpoint_refs: Number of checkpoints with invalid trace_id

    Example:
        >>> stats = await repair_bidirectional_references(
        ...     backend, "run123", dry_run=True
        ... )
        >>> print(f"Would fix {stats['fixed_checkpoints']} checkpoints")

    """
    stats = {
        "fixed_checkpoints": 0,
        "orphaned_traces": 0,
        "dangling_checkpoint_refs": 0,
    }

    # Get run metadata to determine range
    metadata = await backend.get_run_metadata(run_id)
    if not metadata:
        return stats

    if end_wave is None:
        end_wave = metadata.total_waves

    # Scan each wave
    for wave in range(start_wave, end_wave + 1):
        checkpoint = await backend.get_state_snapshot(run_id, wave)
        trace = await backend.get_trace(run_id, wave)

        if checkpoint and trace:
            # Both exist - ensure bidirectional link
            if checkpoint.trace_id != trace.trace_id:
                if not dry_run:
                    checkpoint.trace_id = trace.trace_id
                    await backend.update_state_snapshot(checkpoint)
                stats["fixed_checkpoints"] += 1

        elif checkpoint and checkpoint.trace_id and not trace:
            # Dangling checkpoint reference
            stats["dangling_checkpoint_refs"] += 1
            if not dry_run:
                checkpoint.trace_id = None
                await backend.update_state_snapshot(checkpoint)

        elif trace and not checkpoint:
            # Orphaned trace
            stats["orphaned_traces"] += 1

    return stats
