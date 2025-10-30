"""Routing types and enums for conditional flow control.

This module provides the routing primitives for conditional edges that enable
loops and dynamic control flow in workflows.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING
from typing import Any
from typing import Protocol
from typing import TypeVar
from typing import runtime_checkable

from pydantic import BaseModel

if TYPE_CHECKING:
    pass


class Route(StrEnum):
    """Special routing outcomes for conditional flow control.

    Attributes:
        END: Sentinel value to terminate flow execution.

    """

    END = "END"


if TYPE_CHECKING:
    T_Route = Route | str | "BaseNode"
else:
    T_Route = Route | str

"""Type alias for routing outcomes.

A router function can return:
- Route.END: to terminate the flow
- str: the name of the target node to route to
- BaseNode: direct reference to the target node
- list of the above: multiple target nodes for fan-out
"""


StateT_contra = TypeVar("StateT_contra", bound=BaseModel, contravariant=True)


@runtime_checkable
class RouterFunction(Protocol[StateT_contra]):
    """Protocol for router functions with type-safe state access.

    This protocol enables type-safe router definitions where the state
    parameter is properly typed, providing IDE support and type checking.

    Example:
        ```python
        from pydantic_flow.core.routing import Route

        def router(state: OutputState) -> Route | BaseNode:
            if state.tick.n >= 5:  # IDE knows about .tick attribute
                return Route.END
            return tick_node  # Return node reference

        flow.add_conditional_edges(tick_node, router)  # Type checked!
        ```

    """

    def __call__(self, state: StateT_contra) -> Any:
        """Route based on typed state.

        Args:
            state: The typed state object with known fields.

        Returns:
            Routing outcome(s) - Route.END, node reference, or list.

        """
        ...
