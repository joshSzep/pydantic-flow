"""Conversation message types for linked list conversation memory."""

from datetime import UTC
from datetime import datetime
from typing import Any

from pydantic import BaseModel
from pydantic import Field

from pydantic_flow.checkpoints.serialization import TypedSerializer
from pydantic_flow.checkpoints.serialization import compress
from pydantic_flow.checkpoints.serialization import decompress
from pydantic_flow.checkpoints.types import ConversationMessageId
from pydantic_flow.checkpoints.types import RunId
from pydantic_flow.checkpoints.types import generate_message_id


class ConversationMessage(BaseModel):
    """Single message in conversation linked list.

    Each message references its predecessor, forming an append-only
    linked list structure. Messages are stored once and referenced
    by snapshot conversation_head_id.

    Attributes:
        message_id: Unique identifier for this message.
        run_id: Flow execution run identifier.
        previous_message_id: Reference to previous message (None if first).
        created_at: Timestamp of message creation.
        message: The actual message content.
        metadata: Additional message metadata.

    """

    message_id: ConversationMessageId = Field(default_factory=generate_message_id)
    run_id: RunId
    previous_message_id: ConversationMessageId | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    message: Any
    metadata: dict[str, Any] = Field(default_factory=dict)

    def serialize(self) -> bytes:
        """Serialize with type preservation and compression.

        Returns:
            Compressed serialized bytes.

        """
        data = TypedSerializer.serialize(self)
        return compress(data, level=6)

    @classmethod
    def deserialize(cls, data: bytes) -> ConversationMessage:
        """Deserialize with type reconstruction.

        Args:
            data: Compressed serialized bytes.

        Returns:
            Reconstructed ConversationMessage.

        """
        decompressed = decompress(data)
        return TypedSerializer.deserialize(decompressed)
