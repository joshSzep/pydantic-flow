"""Access control for checkpoint operations.

Provides permission-based access control for checkpoint read/write/delete operations.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel
from pydantic import Field


class Permission(str, Enum):
    """Checkpoint operation permissions.

    Attributes:
        READ_CHECKPOINT: Can read checkpoint snapshots.
        WRITE_CHECKPOINT: Can write checkpoint snapshots.
        DELETE_CHECKPOINT: Can delete checkpoints.
        READ_TRACE: Can read execution traces.
        WRITE_TRACE: Can write execution traces.
        DELETE_TRACE: Can delete traces.
        ADMIN: Full administrative access.

    """

    READ_CHECKPOINT = "read_checkpoint"
    WRITE_CHECKPOINT = "write_checkpoint"
    DELETE_CHECKPOINT = "delete_checkpoint"
    READ_TRACE = "read_trace"
    WRITE_TRACE = "write_trace"
    DELETE_TRACE = "delete_trace"
    ADMIN = "admin"


class AccessDeniedError(Exception):
    """Raised when access is denied to a checkpoint operation."""

    pass


class AccessPolicy(BaseModel):
    """Access control policy.

    Attributes:
        user_id: User this policy applies to.
        permissions: Granted permissions.
        allowed_run_ids: Optional list of specific runs user can access.
        denied_run_ids: Optional list of runs user is explicitly denied.

    """

    user_id: str
    permissions: list[Permission] = Field(default_factory=list)
    allowed_run_ids: list[str] | None = None
    denied_run_ids: list[str] = Field(default_factory=list)


class CheckpointAccessControl:
    """Access control for checkpoint operations.

    Enforces permission-based access control on checkpoint storage operations.

    Example:
        >>> from pydantic_flow.checkpoints import SQLiteCheckpointBackend
        >>> base_backend = SQLiteCheckpointBackend(...)
        >>> # Define policies
        >>> policies = {
        ...     "user1": AccessPolicy(
        ...         user_id="user1",
        ...         permissions=[Permission.READ_CHECKPOINT],
        ...     ),
        ...     "admin": AccessPolicy(
        ...         user_id="admin",
        ...         permissions=[Permission.ADMIN],
        ...     ),
        ... }
        >>> # Wrap backend with access control
        >>> protected_backend = CheckpointAccessControl(
        ...     backend=base_backend,
        ...     policies=policies,
        ...     current_user="user1",
        ... )

    """

    def __init__(
        self,
        backend: Any,  # CheckpointStorageBackend
        policies: dict[str, AccessPolicy],
        current_user: str,
    ):
        """Initialize access control.

        Args:
            backend: Underlying checkpoint storage backend.
            policies: Mapping of user_id to AccessPolicy.
            current_user: Current user performing operations.

        """
        self._backend = backend
        self._policies = policies
        self._current_user = current_user

    def _check_permission(
        self, permission: Permission, run_id: str | None = None
    ) -> None:
        """Check if current user has permission.

        Args:
            permission: Permission to check.
            run_id: Optional run ID for run-specific checks.

        Raises:
            AccessDeniedError: If permission is denied.

        """
        policy = self._policies.get(self._current_user)
        if not policy:
            msg = f"No policy found for user: {self._current_user}"
            raise AccessDeniedError(msg)

        # Admin has all permissions
        if Permission.ADMIN in policy.permissions:
            return

        # Check specific permission
        if permission not in policy.permissions:
            msg = (
                f"User {self._current_user} does not have "
                f"permission: {permission.value}"
            )
            raise AccessDeniedError(msg)

        # Check run-specific access
        if run_id:
            # If denied list exists and run is denied
            if run_id in policy.denied_run_ids:
                msg = f"Access denied to run: {run_id}"
                raise AccessDeniedError(msg)

            # If allowed list exists and run is not in it
            if (
                policy.allowed_run_ids is not None
                and run_id not in policy.allowed_run_ids
            ):
                msg = f"User not authorized for run: {run_id}"
                raise AccessDeniedError(msg)

    async def get_state_snapshot(self, run_id: Any, wave_number: int) -> Any | None:
        """Get state snapshot with permission check.

        Args:
            run_id: Run identifier.
            wave_number: Wave number.

        Returns:
            StateSnapshot or None.

        Raises:
            AccessDeniedError: If user lacks READ_CHECKPOINT permission.

        """
        self._check_permission(Permission.READ_CHECKPOINT, str(run_id))
        return await self._backend.get_state_snapshot(run_id, wave_number)

    async def save_state_snapshot(self, snapshot: Any) -> None:
        """Save state snapshot with permission check.

        Args:
            snapshot: StateSnapshot to save.

        Raises:
            AccessDeniedError: If user lacks WRITE_CHECKPOINT permission.

        """
        self._check_permission(Permission.WRITE_CHECKPOINT, str(snapshot.run_id))
        await self._backend.save_state_snapshot(snapshot)

    async def delete_run(self, run_id: Any) -> None:
        """Delete run with permission check.

        Args:
            run_id: Run to delete.

        Raises:
            AccessDeniedError: If user lacks DELETE_CHECKPOINT permission.

        """
        self._check_permission(Permission.DELETE_CHECKPOINT, str(run_id))
        await self._backend.delete_run(run_id)

    async def get_trace(self, run_id: Any, wave_number: int) -> Any | None:
        """Get trace with permission check.

        Args:
            run_id: Run identifier.
            wave_number: Wave number.

        Returns:
            ExecutionTrace or None.

        Raises:
            AccessDeniedError: If user lacks READ_TRACE permission.

        """
        self._check_permission(Permission.READ_TRACE, str(run_id))
        return await self._backend.get_trace(run_id, wave_number)

    async def save_trace(self, trace: Any) -> None:
        """Save trace with permission check.

        Args:
            trace: ExecutionTrace to save.

        Raises:
            AccessDeniedError: If user lacks WRITE_TRACE permission.

        """
        self._check_permission(Permission.WRITE_TRACE, str(trace.run_id))
        await self._backend.save_trace(trace)

    async def list_runs(self, limit: int | None = None) -> list[Any]:
        """List runs with permission filtering.

        Only returns runs the user has access to.

        Args:
            limit: Optional limit on number of results.

        Returns:
            List of RunMetadata user can access.

        """
        # Get all runs from backend
        all_runs = await self._backend.list_runs(limit=limit)

        # Filter based on user's access policy
        policy = self._policies.get(self._current_user)
        if not policy:
            return []

        # Admin sees everything
        if Permission.ADMIN in policy.permissions:
            return all_runs

        # Filter runs
        filtered_runs = []
        for run in all_runs:
            run_id_str = str(run.run_id)

            # Check denied list
            if run_id_str in policy.denied_run_ids:
                continue

            # Check allowed list (if it exists)
            if (
                policy.allowed_run_ids is not None
                and run_id_str not in policy.allowed_run_ids
            ):
                continue

            filtered_runs.append(run)

        return filtered_runs

    # Delegate other methods to underlying backend
    def __getattr__(self, name: str) -> Any:
        """Delegate unknown methods to underlying backend."""
        return getattr(self._backend, name)


def create_readonly_backend(backend: Any, user_id: str) -> CheckpointAccessControl:
    """Create readonly access to checkpoint backend.

    Convenience function for common use case.

    Args:
        backend: Base checkpoint backend.
        user_id: User identifier.

    Returns:
        CheckpointAccessControl with readonly access.

    Example:
        >>> readonly_backend = create_readonly_backend(
        ...     base_backend, "analyst@company.com"
        ... )

    """
    policy = AccessPolicy(
        user_id=user_id,
        permissions=[Permission.READ_CHECKPOINT, Permission.READ_TRACE],
    )
    return CheckpointAccessControl(
        backend=backend,
        policies={user_id: policy},
        current_user=user_id,
    )


def create_admin_backend(backend: Any, user_id: str) -> CheckpointAccessControl:
    """Create admin access to checkpoint backend.

    Args:
        backend: Base checkpoint backend.
        user_id: Admin user identifier.

    Returns:
        CheckpointAccessControl with full admin access.

    """
    policy = AccessPolicy(
        user_id=user_id,
        permissions=[Permission.ADMIN],
    )
    return CheckpointAccessControl(
        backend=backend,
        policies={user_id: policy},
        current_user=user_id,
    )
