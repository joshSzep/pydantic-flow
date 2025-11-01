"""Delta computation for state snapshots.

This module provides utilities for computing forward and reverse deltas
between state dictionaries to enable efficient storage and time-travel.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel

from pydantic_flow.checkpoints.types import DELETED_KEY
from pydantic_flow.checkpoints.types import DeletedKey


class DeltaComputer:
    """Compute forward and reverse deltas between states."""

    @staticmethod
    def compute_forward_delta(
        prev_state: Mapping[str, BaseModel],
        current_state: Mapping[str, BaseModel],
    ) -> dict[str, BaseModel]:
        """Compute forward delta from previous to current state.

        Args:
            prev_state: Previous state dictionary.
            current_state: Current state dictionary.

        Returns:
            Dictionary containing only changed or new keys.

        """
        delta: dict[str, BaseModel] = {}

        for key, value in current_state.items():
            if key not in prev_state or prev_state[key] != value:
                delta[key] = value

        return delta

    @staticmethod
    def compute_reverse_delta(
        prev_state: Mapping[str, BaseModel],
        current_state: Mapping[str, BaseModel],
    ) -> dict[str, BaseModel | DeletedKey]:
        """Compute reverse delta from current to previous state.

        Args:
            prev_state: Previous state dictionary.
            current_state: Current state dictionary.

        Returns:
            Dictionary containing reverted or deleted keys.

        """
        delta: dict[str, BaseModel | DeletedKey] = {}

        for key, value in current_state.items():
            if key in prev_state:
                if prev_state[key] != value:
                    delta[key] = prev_state[key]
            else:
                delta[key] = DELETED_KEY

        return delta

    @staticmethod
    def apply_forward_delta(
        base_state: Mapping[str, BaseModel],
        delta: Mapping[str, BaseModel],
    ) -> dict[str, BaseModel]:
        """Apply forward delta to base state.

        Args:
            base_state: Base state to apply delta to.
            delta: Forward delta to apply.

        Returns:
            New state with delta applied.

        """
        result = dict(base_state)
        result.update(delta)
        return result

    @staticmethod
    def apply_reverse_delta(
        current_state: Mapping[str, BaseModel],
        delta: Mapping[str, BaseModel | DeletedKey],
    ) -> dict[str, BaseModel]:
        """Apply reverse delta to current state.

        Args:
            current_state: Current state to reverse.
            delta: Reverse delta to apply.

        Returns:
            Previous state with delta applied.

        """
        result = dict(current_state)

        for key, value in delta.items():
            if isinstance(value, DeletedKey):
                result.pop(key, None)
            else:
                result[key] = value

        return result
