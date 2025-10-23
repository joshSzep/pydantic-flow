"""Type protocols and guards for node type checking.

This module provides runtime type checking protocols and type guards
for different node patterns, reducing reliance on getattr() and improving
type safety throughout the codebase.
"""

from typing import Any
from typing import Protocol
from typing import TypeGuard
from typing import runtime_checkable

from pydantic_flow.nodes.base import NodeOutput


@runtime_checkable
class NodeWithInput(Protocol):
    """Protocol for nodes with explicit single input dependency.

    Nodes implementing this protocol have a single explicit input
    from another node's output, rather than taking flow-level inputs.
    """

    input: NodeOutput[Any]
    name: str


@runtime_checkable
class NodeWithInputs(Protocol):
    """Protocol for nodes with multiple input dependencies.

    Nodes implementing this protocol (like MergeNode) have multiple
    explicit inputs from other nodes' outputs.
    """

    inputs: tuple[NodeOutput[Any], ...]
    name: str


def has_input_dependency(node: Any) -> TypeGuard[NodeWithInput]:
    """Type guard for nodes with single input dependency.

    Args:
        node: Node to check.

    Returns:
        True if node has a single input dependency.

    """
    return hasattr(node, "input") and node.input is not None  # type: ignore[attr-defined]


def has_multiple_inputs(node: Any) -> TypeGuard[NodeWithInputs]:
    """Type guard for nodes with multiple input dependencies.

    Args:
        node: Node to check.

    Returns:
        True if node has multiple input dependencies.

    """
    return (
        hasattr(node, "inputs")
        and node.inputs is not None  # type: ignore[attr-defined]
        and isinstance(node.inputs, tuple)  # type: ignore[attr-defined]
    )
