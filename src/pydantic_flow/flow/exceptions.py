"""Flow-related exceptions for the pydantic-flow framework."""

from pydantic_flow.core.errors import FlowError


class CyclicDependencyError(FlowError):
    """Raised when a cyclic dependency is detected in the flow."""
