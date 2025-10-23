"""Custom exceptions for pydantic-flow.

This module provides specialized exception types for flow execution errors.
"""


class FlowError(Exception):
    """Base exception for flow-related errors."""


class RecursionLimitError(FlowError):
    """Raised when the maximum number of execution steps is exceeded.

    This prevents infinite loops in flows with conditional routing.
    The error message includes the step count and recent trace information.
    """


class RoutingError(FlowError):
    """Raised when a routing function returns an invalid target.

    This occurs when:
    - A router returns a node name that doesn't exist in the flow
    - A router returns an empty list of targets without ending
    - A router's output cannot be mapped via the provided mapping dict
    """


class FlowTimeoutError(FlowError):
    """Raised when flow execution exceeds the configured timeout.

    This prevents flows from running indefinitely when a time limit is set.
    """
