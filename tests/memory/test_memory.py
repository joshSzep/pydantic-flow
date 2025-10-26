"""Tests for ConversationMemory class.

Note: These tests use mock ModelMessage objects since pydantic-ai's
message types require proper initialization that we'll test in integration tests.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from pydantic_flow.memory import ConversationMemory


def test_conversation_memory_initialization_empty():
    """Test ConversationMemory initializes empty."""
    memory = ConversationMemory()
    assert len(memory) == 0
    assert memory.get() == []
    assert repr(memory) == "ConversationMemory(0 messages)"


def test_conversation_memory_initialization_with_messages():
    """Test ConversationMemory initializes with initial messages."""
    mock_msg1 = MagicMock()
    mock_msg2 = MagicMock()
    initial_messages = [mock_msg1, mock_msg2]

    memory = ConversationMemory(initial_messages=initial_messages)
    assert len(memory) == 2
    assert memory.get() == initial_messages
    assert repr(memory) == "ConversationMemory(2 messages)"


def test_conversation_memory_append():
    """Test appending messages to conversation memory."""
    memory = ConversationMemory()

    msg1 = MagicMock()
    msg2 = MagicMock()

    memory.append(msg1)
    assert len(memory) == 1

    memory.append(msg2)
    assert len(memory) == 2

    messages = memory.get()
    assert messages[0] == msg1
    assert messages[1] == msg2


def test_conversation_memory_extend():
    """Test extending conversation memory with multiple messages."""
    memory = ConversationMemory()

    messages = [MagicMock(), MagicMock(), MagicMock()]

    memory.extend(messages)
    assert len(memory) == 3
    assert memory.get() == messages


def test_conversation_memory_get_returns_copy():
    """Test that get() returns a copy, not the original list."""
    memory = ConversationMemory()
    msg = MagicMock()
    memory.append(msg)

    messages1 = memory.get()
    messages2 = memory.get()

    # They should be different list objects
    assert messages1 is not messages2
    # But contain the same messages
    assert messages1 == messages2


def test_conversation_memory_clear():
    """Test clearing all messages from conversation memory."""
    memory = ConversationMemory()
    memory.append(MagicMock())
    memory.append(MagicMock())

    assert len(memory) == 2

    memory.clear()
    assert len(memory) == 0
    assert memory.get() == []


def test_conversation_memory_copy():
    """Test creating a deep copy of conversation memory."""
    memory1 = ConversationMemory()
    memory1.append(MagicMock())
    memory1.append(MagicMock())

    memory2 = memory1.copy()

    # Should have same content
    assert len(memory2) == 2
    assert memory2.get() == memory1.get()

    # But be independent objects
    memory1.append(MagicMock())
    assert len(memory1) == 3
    assert len(memory2) == 2


def test_conversation_memory_len():
    """Test __len__ returns correct message count."""
    memory = ConversationMemory()
    assert len(memory) == 0

    memory.append(MagicMock())
    assert len(memory) == 1

    memory.append(MagicMock())
    assert len(memory) == 2

    memory.clear()
    assert len(memory) == 0
