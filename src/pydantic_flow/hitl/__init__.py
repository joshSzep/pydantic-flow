"""Human-in-the-Loop (HITL) support for pydantic-flow.

This package provides comprehensive HITL capabilities including:
- Interrupt handling and decision making
- Human input nodes (HumanNode, ApprovalNode)
- Checkpoint persistence for resuming interrupted flows
- Priority-based interrupt callbacks

Example:
    ```python
    from pydantic_flow.hitl import HumanNode, ApprovalNode, InterruptionRequested
    from pydantic_flow.hitl import HandlerPriority, InterruptDecision

    # Create a human approval node
    approval = ApprovalNode(
        prompt="Review this action",
        input=action_node.output,
    )

    # Add interrupt handler to flow
    flow.add_interrupt_handler(
        callback=my_handler,
        priority=HandlerPriority.HIGH
    )
    ```

"""

from pydantic_flow.hitl.decisions import InterruptCallback
from pydantic_flow.hitl.decisions import InterruptDecision
from pydantic_flow.hitl.interrupts import FlowCheckpoint
from pydantic_flow.hitl.interrupts import HandlerPriority
from pydantic_flow.hitl.interrupts import InterruptHandlerRegistration
from pydantic_flow.hitl.interrupts import InterruptionRequested
from pydantic_flow.hitl.nodes import ApprovalNode
from pydantic_flow.hitl.nodes import HumanInputRequest
from pydantic_flow.hitl.nodes import HumanNode
from pydantic_flow.hitl.nodes import HumanResponse

__all__ = [
    "ApprovalNode",
    "FlowCheckpoint",
    "HandlerPriority",
    "HumanInputRequest",
    "HumanNode",
    "HumanResponse",
    "InterruptCallback",
    "InterruptDecision",
    "InterruptHandlerRegistration",
    "InterruptionRequested",
]
