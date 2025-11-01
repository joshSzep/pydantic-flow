"""Checkpoint configuration for flow execution.

This module provides configuration options for checkpoint behavior during
flow execution, including trace sampling and storage backend setup.
"""

from __future__ import annotations

import random
from typing import Any

from pydantic import BaseModel
from pydantic import Field


class CheckpointConfig(BaseModel):
    """Configuration for checkpoint capture during execution.

    Attributes:
        enabled: Whether checkpointing is enabled.
        storage_backend: Storage backend for checkpoints and traces.
        trace_sample_rate: Probability of capturing full traces (0.0-1.0).
        save_full_snapshot_every: Save full state every N waves.
        enable_delta_compression: Use delta compression for intermediate snapshots.

    """

    enabled: bool = True
    storage_backend: Any = Field(default=None)
    trace_sample_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    save_full_snapshot_every: int = Field(default=10, ge=1)
    enable_delta_compression: bool = True

    model_config = {"arbitrary_types_allowed": True}

    def should_sample_trace(self) -> bool:
        """Determine if current execution should capture full trace.

        Returns:
            True if trace should be captured.

        """
        if not self.enabled:
            return False
        if self.trace_sample_rate >= 1.0:
            return True
        if self.trace_sample_rate <= 0.0:
            return False
        return random.random() < self.trace_sample_rate

    def is_full_snapshot_wave(self, wave_number: int) -> bool:
        """Determine if given wave should save full snapshot.

        Args:
            wave_number: Wave/step number.

        Returns:
            True if full snapshot should be saved.

        """
        return wave_number % self.save_full_snapshot_every == 0
