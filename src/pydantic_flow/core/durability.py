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
        SYNC: Checkpoint after every node completes, flushed synchronously.
            Highest durability, ensures every node completion is persisted
            before next node starts. Use for critical workflows where data
            loss is unacceptable.
        ASYNC: Checkpoint after every node completes, flushed asynchronously (DEFAULT).
            Balanced durability and performance. Checkpoints are saved in background
            while execution continues. Recommended for most production workloads.
        EXIT: No automatic checkpointing. Checkpoints only created for HITL
            interrupts or explicit flow termination. Highest performance, use
            for fast batch processing where intermediate state recovery is not needed.

    Examples:
        >>> from pydantic_flow import RunConfig, DurabilityMode
        >>> # Default: automatic async checkpoints
        >>> config = RunConfig()  # Uses ASYNC by default
        >>>
        >>> # For critical workflows requiring maximum durability
        >>> config = RunConfig(durability_mode=DurabilityMode.SYNC)
        >>>
        >>> # For fast batch jobs with minimal checkpointing
        >>> config = RunConfig(durability_mode=DurabilityMode.EXIT)

    """

    SYNC = "sync"
    ASYNC = "async"
    EXIT = "exit"
