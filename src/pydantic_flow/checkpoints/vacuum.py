"""Vacuum manager for checkpoint lifecycle management.

This module provides data lifecycle management for checkpoints, including:
- Time-based cleanup of old traces
- Run deletion with configurable retention
- Policy-based vacuum operations
"""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
from datetime import timedelta
from typing import Any

from pydantic import BaseModel
from pydantic import Field


class VacuumPolicy(BaseModel):
    """Policy for vacuum operations.

    Attributes:
        trace_retention_days: Delete traces older than this (None = never).
        completed_run_retention_days: Delete completed runs older than this.
        failed_run_retention_days: Delete failed runs older than this.
        keep_checkpoints: Keep state snapshots when deleting runs.
        dry_run: If True, report what would be deleted without deleting.

    """

    trace_retention_days: int | None = Field(
        default=30, description="Trace retention in days (None = keep forever)"
    )
    completed_run_retention_days: int | None = Field(
        default=90,
        description="Completed run retention in days (None = keep forever)",
    )
    failed_run_retention_days: int | None = Field(
        default=180,
        description="Failed run retention in days (None = keep forever)",
    )
    keep_checkpoints: bool = Field(
        default=True, description="Keep state snapshots when deleting runs"
    )
    dry_run: bool = Field(
        default=False,
        description="Report what would be deleted without actually deleting",
    )


class VacuumReport(BaseModel):
    """Report of vacuum operation results.

    Attributes:
        traces_deleted: Number of traces deleted.
        runs_deleted: Number of runs deleted.
        bytes_freed: Estimated bytes freed (if supported).
        dry_run: Whether this was a dry run.

    """

    traces_deleted: int = 0
    runs_deleted: int = 0
    bytes_freed: int | None = None
    dry_run: bool = False


class VacuumManager:
    """Manager for checkpoint data lifecycle operations.

    Provides time-based and policy-based cleanup of checkpoint data.

    Usage:
        backend = SQLiteCheckpointBackend(...)
        await backend.initialize()

        manager = VacuumManager(backend)

        # Delete old traces
        report = await manager.vacuum_traces_before(
            before=datetime.now() - timedelta(days=30)
        )

        # Delete old runs by policy
        policy = VacuumPolicy(
            trace_retention_days=30,
            completed_run_retention_days=90
        )
        report = await manager.vacuum_by_policy(policy)

    Args:
        backend: Checkpoint storage backend to manage.

    """

    def __init__(self, backend: Any):
        """Initialize vacuum manager.

        Args:
            backend: Backend to perform vacuum operations on.

        """
        self.backend = backend

    async def vacuum_traces_before(
        self, before: datetime, *, dry_run: bool = False
    ) -> VacuumReport:
        """Delete execution traces older than the specified datetime.

        Args:
            before: Delete traces with completion_time < this datetime
            dry_run: If True, report what would be deleted without deleting

        Returns:
            VacuumReport with deletion statistics

        """
        report = VacuumReport(dry_run=dry_run)

        runs = await self.backend.list_runs()
        for run in runs:
            metadata = await self.backend.get_run_metadata(run.run_id)
            if metadata and metadata.completed_at:
                # Make sure we compare UTC-aware datetimes
                completed_at = (
                    metadata.completed_at.replace(tzinfo=UTC)
                    if metadata.completed_at.tzinfo is None
                    else metadata.completed_at
                )
                before_utc = (
                    before.replace(tzinfo=UTC) if before.tzinfo is None else before
                )

                if completed_at < before_utc:
                    # Delete all traces for this run
                    for wave in range(metadata.total_waves):
                        if not dry_run:
                            await self.backend.delete_trace(run.run_id, wave)
                        report.traces_deleted += 1

        return report

    async def vacuum_run(
        self,
        run_id: str,
        *,
        keep_checkpoints: bool = False,
        dry_run: bool = False,
    ) -> VacuumReport:
        """Delete all data for a specific run.

        Args:
            run_id: Run ID to delete.
            keep_checkpoints: Keep state snapshots.
            dry_run: If True, report without deleting.

        Returns:
            Vacuum report with deletion statistics.

        """
        report = VacuumReport(dry_run=dry_run)

        metadata = await self.backend.get_run_metadata(run_id)

        if metadata:
            if not dry_run:
                await self.backend.delete_run(run_id, keep_checkpoints=keep_checkpoints)

            report.runs_deleted = 1
            report.traces_deleted = metadata.total_waves

        return report

    async def vacuum_by_policy(self, policy: VacuumPolicy) -> VacuumReport:
        """Vacuum checkpoint data based on policy.

        Args:
            policy: Vacuum policy defining retention rules.

        Returns:
            Vacuum report with deletion statistics.

        """
        report = VacuumReport(dry_run=policy.dry_run)

        now = datetime.now(UTC)

        # Vacuum old traces
        if policy.trace_retention_days is not None:
            trace_cutoff = now - timedelta(days=policy.trace_retention_days)
            trace_report = await self.vacuum_traces_before(
                trace_cutoff, dry_run=policy.dry_run
            )
            report.traces_deleted += trace_report.traces_deleted

        # Vacuum old runs
        runs = await self.backend.list_runs()

        for run in runs:
            should_delete = False

            if (
                run.status.value == "completed"
                and policy.completed_run_retention_days is not None
            ):
                cutoff = now - timedelta(days=policy.completed_run_retention_days)
                if run.started_at < cutoff:
                    should_delete = True

            elif (
                run.status.value == "failed"
                and policy.failed_run_retention_days is not None
            ):
                cutoff = now - timedelta(days=policy.failed_run_retention_days)
                if run.started_at < cutoff:
                    should_delete = True

            if should_delete:
                run_report = await self.vacuum_run(
                    run.run_id,
                    keep_checkpoints=policy.keep_checkpoints,
                    dry_run=policy.dry_run,
                )
                report.runs_deleted += run_report.runs_deleted

        return report
