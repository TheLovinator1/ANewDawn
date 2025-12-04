from __future__ import annotations

import pytest

from main import (
    add_message_to_memory,
    last_trigger_time,
    recent_messages,
    reset_memory,
    reset_snapshots,
    undo_reset,
    update_trigger_time,
)


@pytest.fixture(autouse=True)
def clear_state() -> None:
    """Clear all state before each test."""
    recent_messages.clear()
    last_trigger_time.clear()
    reset_snapshots.clear()


class TestResetMemory:
    """Tests for the reset_memory function."""

    def test_reset_memory_clears_messages(self) -> None:
        """Test that reset_memory clears messages for the channel."""
        channel_id = "test_channel_123"
        add_message_to_memory(channel_id, "user1", "Hello")
        add_message_to_memory(channel_id, "user2", "World")

        assert channel_id in recent_messages
        assert len(recent_messages[channel_id]) == 2

        reset_memory(channel_id)

        assert channel_id not in recent_messages

    def test_reset_memory_clears_trigger_times(self) -> None:
        """Test that reset_memory clears trigger times for the channel."""
        channel_id = "test_channel_123"
        update_trigger_time(channel_id, "user1")

        assert channel_id in last_trigger_time

        reset_memory(channel_id)

        assert channel_id not in last_trigger_time

    def test_reset_memory_creates_snapshot(self) -> None:
        """Test that reset_memory creates a snapshot for undo."""
        channel_id = "test_channel_123"
        add_message_to_memory(channel_id, "user1", "Test message")
        update_trigger_time(channel_id, "user1")

        reset_memory(channel_id)

        assert channel_id in reset_snapshots
        messages_snapshot, trigger_snapshot = reset_snapshots[channel_id]
        assert len(messages_snapshot) == 1
        assert "user1" in trigger_snapshot

    def test_reset_memory_no_snapshot_for_empty_channel(self) -> None:
        """Test that reset_memory doesn't create snapshot for empty channel."""
        channel_id = "empty_channel"

        reset_memory(channel_id)

        assert channel_id not in reset_snapshots


class TestUndoReset:
    """Tests for the undo_reset function."""

    def test_undo_reset_restores_messages(self) -> None:
        """Test that undo_reset restores messages."""
        channel_id = "test_channel_123"
        add_message_to_memory(channel_id, "user1", "Hello")
        add_message_to_memory(channel_id, "user2", "World")

        reset_memory(channel_id)
        assert channel_id not in recent_messages

        result = undo_reset(channel_id)

        assert result is True
        assert channel_id in recent_messages
        assert len(recent_messages[channel_id]) == 2

    def test_undo_reset_restores_trigger_times(self) -> None:
        """Test that undo_reset restores trigger times."""
        channel_id = "test_channel_123"
        update_trigger_time(channel_id, "user1")
        original_time = last_trigger_time[channel_id]["user1"]

        reset_memory(channel_id)
        assert channel_id not in last_trigger_time

        result = undo_reset(channel_id)

        assert result is True
        assert channel_id in last_trigger_time
        assert last_trigger_time[channel_id]["user1"] == original_time

    def test_undo_reset_removes_snapshot(self) -> None:
        """Test that undo_reset removes the snapshot after restoring."""
        channel_id = "test_channel_123"
        add_message_to_memory(channel_id, "user1", "Hello")

        reset_memory(channel_id)
        assert channel_id in reset_snapshots

        undo_reset(channel_id)

        assert channel_id not in reset_snapshots

    def test_undo_reset_returns_false_when_no_snapshot(self) -> None:
        """Test that undo_reset returns False when no snapshot exists."""
        channel_id = "nonexistent_channel"

        result = undo_reset(channel_id)

        assert result is False

    def test_undo_reset_only_works_once(self) -> None:
        """Test that undo_reset only works once (snapshot is removed after undo)."""
        channel_id = "test_channel_123"
        add_message_to_memory(channel_id, "user1", "Hello")

        reset_memory(channel_id)
        first_undo = undo_reset(channel_id)
        second_undo = undo_reset(channel_id)

        assert first_undo is True
        assert second_undo is False


class TestResetUndoIntegration:
    """Integration tests for reset and undo functionality."""

    def test_reset_then_undo_preserves_content(self) -> None:
        """Test that reset followed by undo preserves original content."""
        channel_id = "test_channel_123"
        add_message_to_memory(channel_id, "user1", "Message 1")
        add_message_to_memory(channel_id, "user2", "Message 2")
        add_message_to_memory(channel_id, "user3", "Message 3")
        update_trigger_time(channel_id, "user1")
        update_trigger_time(channel_id, "user2")

        # Capture original state
        original_messages = list(recent_messages[channel_id])
        original_trigger_users = set(last_trigger_time[channel_id].keys())

        reset_memory(channel_id)
        undo_reset(channel_id)

        # Verify restored state matches original
        restored_messages = list(recent_messages[channel_id])
        restored_trigger_users = set(last_trigger_time[channel_id].keys())

        assert len(restored_messages) == len(original_messages)
        assert restored_trigger_users == original_trigger_users

    def test_multiple_resets_overwrite_snapshot(self) -> None:
        """Test that multiple resets overwrite the previous snapshot."""
        channel_id = "test_channel_123"

        # First set of messages
        add_message_to_memory(channel_id, "user1", "First message")
        reset_memory(channel_id)

        # Second set of messages
        add_message_to_memory(channel_id, "user1", "Second message")
        add_message_to_memory(channel_id, "user1", "Third message")
        reset_memory(channel_id)

        # Undo should restore the second set, not the first
        undo_reset(channel_id)

        assert channel_id in recent_messages
        assert len(recent_messages[channel_id]) == 2

    def test_different_channels_independent_undo(self) -> None:
        """Test that different channels have independent undo functionality."""
        channel_1 = "channel_1"
        channel_2 = "channel_2"

        add_message_to_memory(channel_1, "user1", "Channel 1 message")
        add_message_to_memory(channel_2, "user2", "Channel 2 message")

        reset_memory(channel_1)
        reset_memory(channel_2)

        # Undo only channel 1
        undo_reset(channel_1)

        assert channel_1 in recent_messages
        assert channel_2 not in recent_messages
        assert channel_1 not in reset_snapshots
        assert channel_2 in reset_snapshots
