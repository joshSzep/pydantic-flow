"""HITL nodes for Human-in-the-Loop interactions.

This module provides specialized nodes that interrupt flow execution
to request human input or approval before continuing.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel
from pydantic import Field

from pydantic_flow.checkpoints.types import SnapshotReason
from pydantic_flow.checkpoints.types import StateSnapshot
from pydantic_flow.checkpoints.types import generate_run_id
from pydantic_flow.checkpoints.types import generate_snapshot_id
from pydantic_flow.hitl.decisions import InterruptDecision
from pydantic_flow.hitl.interrupts import InterruptionRequested
from pydantic_flow.nodes.base import NodeOutput
from pydantic_flow.nodes.base import NodeWithInput
from pydantic_flow.streaming.base import ProgressItem
from pydantic_flow.streaming.core_events import StreamStart


class HumanInputRequest(BaseModel):
    """Request for human input during flow execution.

    Attributes:
        prompt: Message or question to present to the human.
        context: Additional context about why input is needed.
        input_type: Expected type of input (text, approval, choice, etc.).
        options: Optional list of choices for selection-type inputs.
        metadata: Additional metadata about the request.

    """

    prompt: str
    context: dict[str, Any] = Field(default_factory=dict)
    input_type: str = "text"
    options: list[str] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class HumanResponse(BaseModel):
    """Response from human input.

    Attributes:
        value: The actual response value.
        approved: Whether the action was approved (for approval-type requests).
        metadata: Additional metadata about the response.

    """

    value: Any
    approved: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class HumanNode[InputModel: BaseModel, OutputModel: BaseModel](
    NodeWithInput[InputModel, OutputModel]
):
    """A node that always interrupts execution for human input.

    This node is designed for HITL patterns where human intervention
    is required. It always raises InterruptionRequested, forcing the
    flow to pause and wait for human input before resuming.

    Example:
        ```python
        # Create a human approval node
        approval_node = HumanNode(
            prompt="Review the generated content",
            response_parser=lambda resp: ApprovalModel(approved=resp.approved),
            input=content_generator.output,
        )

        # In the flow, this will interrupt and wait for human input
        flow.add_nodes(content_generator, approval_node)
        ```

    """

    def __init__(  # noqa: PLR0913
        self,
        prompt: str | Callable[[InputModel], str],
        *,
        response_parser: Callable[[HumanResponse], OutputModel] | None = None,
        input_type: str = "text",
        options: list[str] | None = None,
        input: NodeOutput[InputModel] | None = None,
        name: str | None = None,
        run_id: str | None = None,
    ) -> None:
        """Initialize a HumanNode.

        Args:
            prompt: Static prompt or function that takes input and returns prompt.
            response_parser: Function to parse HumanResponse into OutputModel.
                If None, expects HumanResponse as the output type.
            input_type: Type of input expected (text, approval, choice, etc.).
            options: Optional list of choices for selection-type inputs.
            input: Optional input from another node's output.
            name: Optional unique identifier for this node.
            run_id: Optional run identifier for tracking execution.

        """
        super().__init__(input, name, run_id)
        self._prompt = prompt
        self._response_parser = response_parser
        self._human_input_type = input_type
        self._options = options

    def _format_prompt(self, input_data: InputModel) -> str:
        """Format the prompt based on input data.

        Args:
            input_data: The input model instance.

        Returns:
            Formatted prompt string.

        """
        if callable(self._prompt):
            return self._prompt(input_data)
        return self._prompt

    def _create_input_request(self, input_data: InputModel) -> HumanInputRequest:
        """Create the input request with context.

        Args:
            input_data: The input data for context.

        Returns:
            HumanInputRequest with formatted prompt and context.

        """
        prompt = self._format_prompt(input_data)
        context = {}
        if hasattr(input_data, "model_dump"):
            context = input_data.model_dump()

        return HumanInputRequest(
            prompt=prompt,
            context=context,
            input_type=self._human_input_type,
            options=self._options,
            metadata={"node_id": self.name, "run_id": self.run_id or ""},
        )

    async def astream(self, input_data: InputModel) -> AsyncIterator[ProgressItem]:
        """Stream progress items and then immediately interrupt for human input.

        This method always raises InterruptionRequested after emitting
        a StreamStart event. The flow must be resumed with human input
        to continue.

        Args:
            input_data: Input data for this node.

        Yields:
            StreamStart event.

        Raises:
            InterruptionRequested: Always raised to request human input.

        """
        actual_run_id = self.run_id or ""

        start_item = StreamStart(
            run_id=actual_run_id,
            node_id=self.name,
            input_preview={"prompt": self._format_prompt(input_data)[:100]},
        )
        yield start_item

        input_request = self._create_input_request(input_data)

        checkpoint = StateSnapshot(
            snapshot_id=generate_snapshot_id(),
            run_id=generate_run_id(),
            wave_number=0,
            state_hash="",
            next_frontier=[],
            routing_ended=False,
            reason=SnapshotReason.HITL_INTERRUPT,
            interrupted_node_id=self.name,
            metadata=input_request.model_dump(),
        )

        decision = InterruptDecision.interrupt(
            reason=f"Human input required: {input_request.prompt}",
            replacement_value=input_request,
            metadata={"request": input_request.model_dump()},
        )

        raise InterruptionRequested(snapshot=checkpoint, decision=decision)

    def parse_response(self, response: HumanResponse) -> OutputModel:
        """Parse human response into the output model.

        Args:
            response: The human response.

        Returns:
            Parsed output model. If no response_parser is configured,
            returns the HumanResponse as-is (assumes OutputModel = HumanResponse).

        """
        if self._response_parser is not None:
            return self._response_parser(response)

        return response  # type: ignore


class ApprovalNode[InputModel: BaseModel](NodeWithInput[InputModel, HumanResponse]):
    """Specialized HumanNode for approval workflows.

    This is a convenience node for simple approve/reject patterns.
    It returns a HumanResponse with the approval decision.

    Example:
        ```python
        approval = ApprovalNode(
            prompt="Approve this action?",
            input=action_node.output,
        )
        ```

    """

    def __init__(
        self,
        prompt: str | Callable[[InputModel], str],
        *,
        input: NodeOutput[InputModel] | None = None,
        name: str | None = None,
        run_id: str | None = None,
    ) -> None:
        """Initialize an ApprovalNode.

        Args:
            prompt: Static prompt or function that takes input and returns prompt.
            input: Optional input from another node's output.
            name: Optional unique identifier for this node.
            run_id: Optional run identifier for tracking execution.

        """
        self._prompt = prompt
        super().__init__(input, name, run_id)

    def _format_prompt(self, input_data: InputModel) -> str:
        """Format the prompt based on input data."""
        if callable(self._prompt):
            return self._prompt(input_data)
        return self._prompt

    async def astream(self, input_data: InputModel) -> AsyncIterator[ProgressItem]:
        """Stream and interrupt for approval.

        Yields:
            StreamStart event.

        Raises:
            InterruptionRequested: Always raised to request approval.

        """
        actual_run_id = self.run_id or ""

        start_item = StreamStart(
            run_id=actual_run_id,
            node_id=self.name,
            input_preview={"prompt": self._format_prompt(input_data)[:100]},
        )
        yield start_item

        prompt = self._format_prompt(input_data)
        context = input_data.model_dump() if hasattr(input_data, "model_dump") else {}
        input_request = HumanInputRequest(
            prompt=prompt,
            context=context,
            input_type="approval",
            options=["approve", "reject"],
            metadata={"node_id": self.name, "run_id": actual_run_id},
        )

        checkpoint = StateSnapshot(
            snapshot_id=generate_snapshot_id(),
            run_id=generate_run_id(),
            wave_number=0,
            state_hash="",
            next_frontier=[],
            routing_ended=False,
            reason=SnapshotReason.HITL_INTERRUPT,
            interrupted_node_id=self.name,
            metadata=input_request.model_dump(),
        )

        decision = InterruptDecision.interrupt(
            reason=f"Approval required: {prompt}",
            replacement_value=input_request,
            metadata={"request": input_request.model_dump()},
        )

        raise InterruptionRequested(snapshot=checkpoint, decision=decision)
