"""Routing types and enums for conditional flow control.

This module provides the routing primitives for conditional edges that enable
loops and dynamic control flow in workflows.
"""

from enum import StrEnum
from typing import Protocol
from typing import TypeVar
from typing import runtime_checkable

from pydantic import BaseModel


class Route(StrEnum):
    """Special routing outcomes for conditional flow control.

    Attributes:
        END: Sentinel value to terminate flow execution.

    """

    END = "END"


T_Route = Route | str
"""Type alias for routing outcomes.

A router function can return:
- Route.END: to terminate the flow
- str: the name of the target node to route to
- list[str]: multiple target nodes for fan-out
"""


StateT_contra = TypeVar("StateT_contra", bound=BaseModel, contravariant=True)


@runtime_checkable
class RouterFunction(Protocol[StateT_contra]):
    """Protocol for router functions with type-safe state access.

    This protocol enables type-safe router definitions where the state
    parameter is properly typed, providing IDE support and type checking.

    Example:
        ```python
        from pydantic_flow.core.routing import T_Route

        def router(state: OutputState) -> T_Route:
            if state.tick.n >= 5:  # IDE knows about .tick attribute
                return Route.END
            return "tick"

        flow.add_conditional_edges("tick", router)  # Type checked!
        ```

    """

    def __call__(self, state: StateT_contra) -> T_Route | list[T_Route]:
        """Route based on typed state.

        Args:
            state: The typed state object with known fields.

        Returns:
            Routing outcome(s) - either Route.END, node name(s), or list of node names.

        """
        ...
