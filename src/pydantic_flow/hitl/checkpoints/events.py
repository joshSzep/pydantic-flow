"""Checkpoint-related streaming events.

This module defines events related to checkpoint persistence and recovery.
"""

from __future__ import annotations

from pydantic_flow.streaming.base import ProgressItem
from pydantic_flow.streaming.base import ProgressType


class CheckpointSaved(ProgressItem):
    """Checkpoint was persisted to storage.

    Attributes:
        node_id: Node where checkpoint was created.
        checkpoint_id: Unique identifier for the saved checkpoint.
        run_id: Run identifier for checkpoint correlation.
        store_type: Type of checkpoint store used.

    """

    type: ProgressType = ProgressType.CHECKPOINT_SAVED
    checkpoint_id: str
    store_type: str
