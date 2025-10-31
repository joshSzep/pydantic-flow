"""Durability modes for checkpoint persistence.

This module provides durability configuration for controlling when and how
checkpoints are created during flow execution.
"""

from __future__ import annotations

from enum import StrEnum


class DurabilityMode(StrEnum):
    """Controls checkpoint frequency and performance tradeoffs.

    Different modes provide different balances between execution performance
    and data durability. Choose based on your application's requirements.

    Note: HITL (Human-in-the-Loop) checkpoints are created independently when
    InterruptionRequested is raised. Durability modes control automatic crash
    recovery checkpoints, not human intervention checkpoints.

    Attributes:
        ASYNC: Checkpoint in background while next node executes (DEFAULT).
            Balanced durability and performance. Recommended for most production
            workloads. Small risk that checkpoint may not complete if process crashes.
        SYNC: Checkpoint after every node, before next node starts.
            Highest durability, lowest performance. Use for critical workflows
            where data loss is unacceptable.
        EXIT: Checkpoint only on flow completion or error.
            Highest performance, lowest durability. Use for fast batch processing
            where intermediate state recovery is not needed.

    Examples:
        >>> from pydantic_flow import RunConfig, DurabilityMode
        >>> # Default: automatic background checkpoints
        >>> config = RunConfig()  # Uses ASYNC by default
        >>>
        >>> # For critical workflows requiring maximum durability
        >>> config = RunConfig(durability_mode=DurabilityMode.SYNC)
        >>>
        >>> # For fast batch jobs with minimal checkpointing
        >>> config = RunConfig(durability_mode=DurabilityMode.EXIT)

    """

    ASYNC = "async"
    SYNC = "sync"
    EXIT = "exit"
