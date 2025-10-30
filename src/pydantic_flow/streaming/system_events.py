"""System-level streaming events.

This module defines events related to system operations,
errors, and health monitoring.
"""

from __future__ import annotations

from pydantic_flow.streaming.base import ProgressItem
from pydantic_flow.streaming.base import ProgressType


class NonFatalError(ProgressItem):
    """A non-fatal error or warning during execution.

    Attributes:
        message: Error description.
        recoverable: Whether execution can continue.

    """

    type: ProgressType = ProgressType.ERROR
    message: str = ""
    recoverable: bool = True


class Heartbeat(ProgressItem):
    """Liveness signal during long-running operations.

    Attributes:
        message: Optional status message.

    """

    type: ProgressType = ProgressType.HEARTBEAT
    message: str = ""
