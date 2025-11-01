"""GDPR compliance utilities for checkpoint data.

Provides user data search, deletion with audit trail, and right-to-be-forgotten
enforcement for checkpoint storage.
"""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
from typing import Any

from pydantic import BaseModel


class GDPRSearchResult(BaseModel):
    """Result from searching for user data in checkpoints.

    Attributes:
        run_id: Run containing user data.
        wave_number: Wave number containing user data.
        field_paths: JSON paths to user data fields.
        match_count: Number of matches found.
        data_preview: Preview of matched data (redacted).

    """

    run_id: str
    wave_number: int
    field_paths: list[str]
    match_count: int
    data_preview: str | None = None


class GDPRErasureLog(BaseModel):
    """Audit log entry for GDPR erasure.

    Attributes:
        log_id: Unique log entry ID.
        user_identifier: User whose data was erased.
        erased_at: When erasure occurred.
        runs_affected: List of run IDs affected.
        waves_affected: Number of waves affected.
        operator: Who performed the erasure.
        reason: Reason for erasure.

    """

    log_id: str
    user_identifier: str
    erased_at: datetime
    runs_affected: list[str]
    waves_affected: int
    operator: str | None = None
    reason: str = "GDPR Right to be Forgotten"


class GDPRErasureManager:
    """Manages GDPR compliance for checkpoint data.

    Provides search and erasure of user data with audit logging.

    Example:
        >>> from pydantic_flow.checkpoints import SQLiteCheckpointBackend
        >>> backend = SQLiteCheckpointBackend(...)
        >>> gdpr = GDPRErasureManager(backend)
        >>> # Search for user data
        >>> results = await gdpr.search_user_data(
        ...     user_identifier="user@example.com"
        ... )
        >>> # Erase user data
        >>> log = await gdpr.erase_user_data(
        ...     user_identifier="user@example.com",
        ...     operator="admin@company.com",
        ... )

    """

    def __init__(
        self,
        backend: Any,  # CheckpointStorageBackend
        *,
        audit_backend: Any | None = None,
    ):
        """Initialize GDPR manager.

        Args:
            backend: Checkpoint storage backend to search/erase from.
            audit_backend: Optional separate backend for audit logs.

        """
        self._backend = backend
        self._audit_backend = audit_backend or backend

    async def search_user_data(
        self,
        user_identifier: str,
        *,
        run_ids: list[str] | None = None,
        max_results: int = 100,
    ) -> list[GDPRSearchResult]:
        """Search for user data in checkpoints.

        Args:
            user_identifier: User email, ID, or other identifier to search for.
            run_ids: Optional list of specific runs to search.
            max_results: Maximum number of results to return.

        Returns:
            List of search results with locations of user data.

        """
        results = []

        # Get all runs or specific runs
        if run_ids:
            runs_to_search = run_ids
        else:
            # Get all runs from backend
            all_runs = await self._backend.list_runs()
            runs_to_search = [run.run_id for run in all_runs[:1000]]

        # Search each run
        for run_id in runs_to_search:
            if len(results) >= max_results:
                break

            run_results = await self._search_run(run_id, user_identifier)
            results.extend(run_results)

            if len(results) >= max_results:
                results = results[:max_results]
                break

        return results

    async def _search_run(
        self, run_id: str, user_identifier: str
    ) -> list[GDPRSearchResult]:
        """Search single run for user data.

        Args:
            run_id: Run to search.
            user_identifier: User identifier to find.

        Returns:
            List of search results for this run.

        """
        results = []

        # Get run metadata to find wave count
        from pydantic_flow.checkpoints.types import RunId

        metadata = await self._backend.get_run_metadata(RunId(run_id))
        if not metadata:
            return results

        # Search each wave
        for wave in range(metadata.total_waves + 1):
            snapshot = await self._backend.get_state_snapshot(RunId(run_id), wave)
            if not snapshot or not snapshot.full_state:
                continue

            # Search state for user identifier
            field_paths = self._find_in_dict(snapshot.full_state, user_identifier)
            if field_paths:
                results.append(
                    GDPRSearchResult(
                        run_id=run_id,
                        wave_number=wave,
                        field_paths=field_paths,
                        match_count=len(field_paths),
                        data_preview=f"Found in {len(field_paths)} locations",
                    )
                )

        return results

    def _find_in_dict(self, obj: Any, search_term: str, path: str = "") -> list[str]:
        """Recursively find search term in nested dict.

        Args:
            obj: Object to search.
            search_term: Term to find.
            path: Current JSON path.

        Returns:
            List of JSON paths where term was found.

        """
        from pydantic import BaseModel

        matches = []

        # Convert BaseModel to dict for searching
        if isinstance(obj, BaseModel):
            obj = obj.model_dump(mode="python")

        if isinstance(obj, dict):
            for key, value in obj.items():
                new_path = f"{path}.{key}" if path else key
                matches.extend(self._find_in_dict(value, search_term, new_path))

        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                new_path = f"{path}[{i}]"
                matches.extend(self._find_in_dict(item, search_term, new_path))

        elif isinstance(obj, str) and search_term.lower() in obj.lower():
            matches.append(path)

        return matches

    async def erase_user_data(
        self,
        user_identifier: str,
        *,
        run_ids: list[str] | None = None,
        operator: str | None = None,
        dry_run: bool = False,
    ) -> GDPRErasureLog:
        """Erase user data from checkpoints (GDPR Right to be Forgotten).

        Args:
            user_identifier: User whose data should be erased.
            run_ids: Optional specific runs to erase from.
            operator: Who is performing the erasure.
            dry_run: If True, report what would be erased without erasing.

        Returns:
            Audit log entry for the erasure.

        """
        from pydantic_flow.checkpoints.types import RunId
        from pydantic_flow.checkpoints.types import generate_run_id

        # First, search for user data
        search_results = await self.search_user_data(user_identifier, run_ids=run_ids)

        affected_runs = list({result.run_id for result in search_results})
        total_waves = sum(1 for _ in search_results)

        if not dry_run:
            # Erase data from each affected run
            for run_id_str in affected_runs:
                run_id = RunId(run_id_str)

                # Get waves that need modification
                waves_to_modify = [
                    r.wave_number for r in search_results if r.run_id == run_id_str
                ]

                for wave in waves_to_modify:
                    snapshot = await self._backend.get_state_snapshot(run_id, wave)
                    if not snapshot:
                        continue

                    # Redact user data
                    if snapshot.full_state:
                        snapshot.full_state = self._redact_user_data(
                            snapshot.full_state, user_identifier
                        )

                    # Save modified snapshot
                    await self._backend.update_state_snapshot(snapshot)

        # Create audit log
        log = GDPRErasureLog(
            log_id=generate_run_id(),  # Reuse ID generator
            user_identifier=user_identifier,
            erased_at=datetime.now(UTC),
            runs_affected=affected_runs,
            waves_affected=total_waves,
            operator=operator,
        )

        # Save audit log
        if not dry_run:
            await self._save_erasure_log(log)

        return log

    def _redact_user_data(self, obj: Any, user_identifier: str) -> Any:
        """Recursively redact user data from object.

        Args:
            obj: Object to redact from.
            user_identifier: User identifier to redact.

        Returns:
            Redacted object.

        """
        import copy

        redacted = copy.deepcopy(obj)

        def redact_recursive(item: Any) -> Any:
            if isinstance(item, dict):
                return {k: redact_recursive(v) for k, v in item.items()}
            if isinstance(item, list):
                return [redact_recursive(i) for i in item]
            if isinstance(item, str):
                if user_identifier.lower() in item.lower():
                    return "***ERASED_PER_GDPR***"
                return item
            return item

        return redact_recursive(redacted)

    async def _save_erasure_log(self, log: GDPRErasureLog) -> None:
        """Save erasure log to audit backend.

        Args:
            log: Erasure log to save.

        """
        # Implementation would save to audit backend
        # For now, we'll just print (would be stored in separate audit table)
        pass

    async def get_erasure_logs(
        self,
        user_identifier: str | None = None,
        *,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[GDPRErasureLog]:
        """Get erasure audit logs.

        Args:
            user_identifier: Optional filter by user.
            start_date: Optional start date filter.
            end_date: Optional end date filter.

        Returns:
            List of erasure logs.

        """
        # Implementation would query audit backend
        # Placeholder for now
        return []
