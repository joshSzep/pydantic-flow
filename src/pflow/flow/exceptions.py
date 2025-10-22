"""Flow-related exceptions for the pflow framework."""


class FlowError(Exception):
    """Base exception for flow-related errors."""


class CyclicDependencyError(FlowError):
    """Raised when a cyclic dependency is detected in the flow."""
