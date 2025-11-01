"""Checkpoint manager for stepper engine integration.

This module provides the CheckpointManager class that handles checkpoint
and trace creation during flow execution.
"""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from pydantic_flow.checkpoints.config import CheckpointConfig
from pydantic_flow.checkpoints.delta import DeltaComputer
from pydantic_flow.checkpoints.event_log import StreamingEventLog
from pydantic_flow.checkpoints.interface import CheckpointStorageBackend
from pydantic_flow.checkpoints.types import ConversationMessageId
from pydantic_flow.checkpoints.types import ExecutionTrace
from pydantic_flow.checkpoints.types import NodeExecutionTrace
from pydantic_flow.checkpoints.types import RunId
from pydantic_flow.checkpoints.types import RunMetadata
from pydantic_flow.checkpoints.types import SnapshotId
from pydantic_flow.checkpoints.types import SnapshotReason
from pydantic_flow.checkpoints.types import StateRef
from pydantic_flow.checkpoints.types import StateSnapshot
from pydantic_flow.checkpoints.types import generate_run_id
from pydantic_flow.checkpoints.types import generate_snapshot_id


class CheckpointManager:
    """Manages checkpoint and trace creation during execution.

    This class encapsulates the logic for:
    - Saving state snapshots with delta compression
    - Creating and managing execution traces
    - Coordinating event capture with streaming

    Attributes:
        config: Checkpoint configuration.
        storage: Storage backend for persistence.
        run_id: Current run identifier.
        flow_id: Flow identifier.
        wave_number: Current wave number.
        previous_state: Previous wave's state for delta computation.
        current_snapshot_id: Current snapshot ID.
        conversation_head_id: Most recent conversation message ID (linked list head).
        previous_message_id: Previous message ID for building linked list.

    """

    def __init__(
        self,
        config: CheckpointConfig,
        storage: CheckpointStorageBackend,
        flow_id: str,
        run_id: RunId | None = None,
    ) -> None:
        """Initialize checkpoint manager.

        Args:
            config: Checkpoint configuration.
            storage: Storage backend.
            flow_id: Flow identifier.
            run_id: Optional run ID (generates new one if not provided).

        """
        self.config = config
        self.storage = storage
        self.flow_id = flow_id
        self.run_id = run_id or generate_run_id()
        self.wave_number = 0
        self.previous_state: dict[str, BaseModel] | None = None
        self.current_snapshot_id = generate_snapshot_id()
        self.conversation_head_id: ConversationMessageId | None = None
        self.previous_message_id: ConversationMessageId | None = None

    async def initialize_run(self) -> None:
        """Initialize run metadata in storage."""
        metadata = RunMetadata(
            run_id=self.run_id,
            flow_id=self.flow_id,
            started_at=datetime.now(UTC),
            status=RunMetadata.Status.RUNNING,
            total_waves=0,
        )
        await self.storage.save_run_metadata(metadata)

    async def save_wave_checkpoint(
        self,
        current_state: dict[str, BaseModel],
        next_frontier: list[str],
        routing_ended: bool,
    ) -> StateSnapshot:
        """Save checkpoint after wave execution.

        Args:
            current_state: Current execution state.
            next_frontier: Next nodes to execute.
            routing_ended: Whether routing ended with Route.END.

        Returns:
            Saved state snapshot.

        """
        snapshot_id = generate_snapshot_id()
        is_full_snapshot = self.config.is_full_snapshot_wave(self.wave_number)

        # Compute state hash
        state_hash = StateSnapshot(
            version=2,
            snapshot_id=snapshot_id,
            run_id=self.run_id,
            wave_number=self.wave_number,
            forward_delta=None,
            reverse_delta=None,
            full_state=current_state if is_full_snapshot else None,
            state_hash="",
            next_frontier=next_frontier,
            routing_ended=routing_ended,
            trace_id=None,
        ).compute_state_hash(current_state)

        # Create snapshot with deltas or full state
        if is_full_snapshot or self.previous_state is None:
            snapshot = StateSnapshot(
                version=2,
                snapshot_id=snapshot_id,
                run_id=self.run_id,
                wave_number=self.wave_number,
                forward_delta=None,
                reverse_delta=None,
                full_state=current_state,
                state_hash=state_hash,
                next_frontier=next_frontier,
                routing_ended=routing_ended,
                trace_id=None,
                conversation_head_id=self.conversation_head_id,
            )
        else:
            # Compute deltas
            forward_delta = DeltaComputer.compute_forward_delta(
                self.previous_state, current_state
            )
            reverse_delta = DeltaComputer.compute_reverse_delta(
                self.previous_state, current_state
            )

            snapshot = StateSnapshot(
                version=2,
                snapshot_id=snapshot_id,
                run_id=self.run_id,
                wave_number=self.wave_number,
                forward_delta=forward_delta,
                reverse_delta=reverse_delta,
                full_state=None,
                state_hash=state_hash,
                next_frontier=next_frontier,
                routing_ended=routing_ended,
                trace_id=None,
                conversation_head_id=self.conversation_head_id,
            )

        # Save snapshot
        await self.storage.save_state_snapshot(snapshot)

        # Update tracking
        self.previous_state = dict(current_state)
        self.current_snapshot_id = snapshot_id
        self.wave_number += 1

        return snapshot

    async def create_trace(
        self,
        node_traces: list[NodeExecutionTrace],
        checkpoint_snapshot_id: SnapshotId,
    ) -> ExecutionTrace:
        """Create and save execution trace.

        Args:
            node_traces: List of node execution traces.
            checkpoint_snapshot_id: Associated checkpoint snapshot ID.

        Returns:
            Created execution trace.

        """
        from pydantic_flow.checkpoints.types import generate_checkpoint_id

        trace = ExecutionTrace(
            trace_id=generate_checkpoint_id(),
            run_id=self.run_id,
            wave_number=self.wave_number - 1,  # Previous wave
            checkpoint_snapshot_id=SnapshotId(checkpoint_snapshot_id),
            node_traces=node_traces,
            parallel_batch_id=generate_checkpoint_id(),
            started_at=min(nt.started_at for nt in node_traces)
            if node_traces
            else datetime.now(UTC),
            completed_at=max(nt.completed_at for nt in node_traces)
            if node_traces
            else datetime.now(UTC),
        )

        await self.storage.save_trace(trace)
        return trace

    async def create_hitl_interrupt_snapshot(
        self,
        current_state: dict[str, BaseModel],
        interrupted_node_id: str,
        next_frontier: list[str],
        conversation_head_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StateSnapshot:
        """Create snapshot for HITL interrupt.

        Creates a snapshot with reason=HITL_INTERRUPT, enabling unified
        resume and time-travel for interrupted flows.

        Args:
            current_state: Current execution state at interrupt.
            interrupted_node_id: Node where interrupt occurred.
            next_frontier: Next nodes to execute after resume.
            conversation_head_id: Optional reference to conversation chain.
            metadata: Additional interrupt context.

        Returns:
            Created HITL interrupt snapshot.

        """
        from pydantic_flow.checkpoints.types import ConversationMessageId

        snapshot_id = generate_snapshot_id()

        # Always save full state for HITL interrupts (easier recovery)
        state_hash = StateSnapshot(
            version=2,
            snapshot_id=snapshot_id,
            run_id=self.run_id,
            wave_number=self.wave_number,
            forward_delta=None,
            reverse_delta=None,
            full_state=current_state,
            state_hash="",
            next_frontier=next_frontier,
            routing_ended=False,
            trace_id=None,
            reason=SnapshotReason.HITL_INTERRUPT,
            interrupted_node_id=interrupted_node_id,
        ).compute_state_hash(current_state)

        snapshot = StateSnapshot(
            version=2,
            snapshot_id=snapshot_id,
            run_id=self.run_id,
            wave_number=self.wave_number,
            forward_delta=None,
            reverse_delta=None,
            full_state=current_state,
            state_hash=state_hash,
            next_frontier=next_frontier,
            routing_ended=False,
            trace_id=None,
            reason=SnapshotReason.HITL_INTERRUPT,
            interrupted_node_id=interrupted_node_id,
            conversation_head_id=ConversationMessageId(conversation_head_id)
            if conversation_head_id
            else None,
            metadata=metadata or {},
        )

        await self.storage.save_state_snapshot(snapshot)

        # Update run metadata to mark as interrupted
        existing = await self.storage.get_run_metadata(self.run_id)
        if existing:
            existing.status = RunMetadata.Status.INTERRUPTED
            existing.interrupted_at_wave = self.wave_number
            existing.interrupt_snapshot_id = snapshot_id
            existing.awaiting_human_decision = True
            await self.storage.save_run_metadata(existing)

        # Track for next wave (don't increment wave_number yet)
        self.current_snapshot_id = snapshot_id

        return snapshot

    def create_event_log(
        self,
        node_id: str,
    ) -> StreamingEventLog | None:
        """Create event log for node execution if sampling.

        Args:
            node_id: Node identifier.

        Returns:
            StreamingEventLog if trace should be captured, None otherwise.

        """
        if not self.config.should_sample_trace():
            return None

        return StreamingEventLog(
            store=self.storage,
            run_id=self.run_id,
            node_id=node_id,
            wave_number=self.wave_number,
            snapshot_id=self.current_snapshot_id,
        )

    def create_state_ref(self, state_key: str) -> StateRef:
        """Create reference to state in current snapshot.

        Args:
            state_key: Key within the snapshot.

        Returns:
            State reference.

        """
        return StateRef(
            snapshot_id=self.current_snapshot_id,
            state_key=state_key,
        )

    async def save_conversation_message(
        self, message: Any, metadata: dict[str, Any] | None = None
    ) -> str:
        """Save a conversation message to the linked list.

        Creates a ConversationMessage with proper linking to previous message,
        saves it to storage, and updates the conversation head pointer.

        Args:
            message: The message content (typically ModelMessage from pydantic-ai).
            metadata: Optional metadata about the message.

        Returns:
            The message ID of the saved message.

        """
        from pydantic_flow.checkpoints.conversation import ConversationMessage
        from pydantic_flow.checkpoints.types import ConversationMessageId
        from pydantic_flow.checkpoints.types import generate_message_id

        message_id = generate_message_id()

        conversation_message = ConversationMessage(
            message_id=message_id,
            run_id=self.run_id,
            previous_message_id=ConversationMessageId(self.previous_message_id)
            if self.previous_message_id
            else None,
            message=message,
            metadata=metadata or {},
        )

        await self.storage.save_conversation_message(conversation_message)

        # Update linked list tracking
        self.previous_message_id = message_id
        self.conversation_head_id = message_id

        return message_id

    async def get_conversation_history(self, limit: int | None = None) -> list[Any]:
        """Retrieve conversation history by walking linked list.

        Args:
            limit: Maximum number of messages to retrieve.

        Returns:
            List of messages in reverse chronological order (newest first).

        """
        if not self.conversation_head_id:
            return []

        messages = await self.storage.get_conversation_history(
            head_id=self.conversation_head_id, limit=limit
        )
        return [msg.message for msg in messages]

    async def finalize_run(
        self,
        status: RunMetadata.Status,
        error: dict[str, Any] | None = None,
    ) -> None:
        """Finalize run metadata.

        Args:
            status: Final run status.
            error: Optional error details.

        """
        from pydantic_flow.checkpoints.types import ExecutionError

        # Get existing metadata for started_at timestamp
        existing = await self.storage.get_run_metadata(self.run_id)
        started_at = existing.started_at if existing else datetime.now(UTC)

        # Mark as awaiting human decision if interrupted
        awaiting_human_decision = status == RunMetadata.Status.INTERRUPTED

        metadata = RunMetadata(
            run_id=self.run_id,
            flow_id=self.flow_id,
            started_at=started_at,
            completed_at=datetime.now(UTC),
            status=status,
            total_waves=self.wave_number,
            error=ExecutionError(**error) if error else None,
            awaiting_human_decision=awaiting_human_decision,
        )
        await self.storage.save_run_metadata(metadata)
