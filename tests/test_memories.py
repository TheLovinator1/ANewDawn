from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

import main
from main import build_remember_confirmation
from main import delete_memories
from main import format_memories
from main import get_memories
from main import get_memory_context
from main import load_memories
from main import memories
from main import normalize_memory_content
from main import parse_remember_command
from main import remember_memory

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture(autouse=True)  # ruff: ignore[pytest-fixture-autouse]
def isolated_memory_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Path]:
    """Use a temporary memory file and an empty store for each test.

    Args:
        tmp_path (Path): The pytest temporary directory.
        monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture.

    Yields:
        Path: The temporary memory file path.
    """
    memory_file: Path = tmp_path / "memories.json"
    monkeypatch.setattr(main, "MEMORIES_FILE", memory_file)
    memories.clear()
    yield memory_file
    memories.clear()


class TestNormalizeMemoryContent:
    """Tests for normalize_memory_content."""

    def test_collapses_whitespace(self) -> None:
        """Test that line breaks and repeated spaces are collapsed."""
        assert normalize_memory_content("  hello\n\tworld  ") == "hello world"

    def test_keeps_short_content(self) -> None:
        """Test that short content is unchanged after normalization."""
        assert normalize_memory_content("likes tea") == "likes tea"

    def test_truncates_long_content(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that content longer than the limit is truncated."""
        monkeypatch.setattr(main, "MEMORY_MAX_CHARS", 10)

        assert normalize_memory_content("a" * 20) == "aaaaaaa..."


class TestParseRememberCommand:
    """Tests for parse_remember_command."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("rembember my favourite colour is green", "my favourite colour is green"),
            ("rember my keys are on the table", "my keys are on the table"),
            ("remember that I like tea", "I like tea"),
            ("Remember: the code is blue", "the code is blue"),
            ("lovibot rembember x", "x"),
            ("@lovibot, remember x", "x"),
            ("hey lovibot please remember that x", "x"),
            ("I want you to remember x", "x"),
            ("<@345000831499894795> remember x", "x"),
            ("remember, x", "x"),
        ],
    )
    def test_extracts_facts(self, text: str, expected: str) -> None:
        """Test that supported remember phrases extract the fact."""
        assert parse_remember_command(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "do you remember x",
            "lovibot do you remember x",
            "I remember x",
            "remembering x",
            "what do you remember?",
            "lovibot, what do you remember?",
            "remember when we went to the beach",
            "lovibot remember when we went to the beach",
        ],
    )
    def test_ignores_non_commands(self, text: str) -> None:
        """Test that questions, reminiscing, and other uses are ignored."""
        assert parse_remember_command(text) is None

    @pytest.mark.parametrize(
        "text",
        ["remember", "remember that", "rembember", "lovibot remember   "],
    )
    def test_returns_empty_content_for_bare_commands(self, text: str) -> None:
        """Test that a remember command without content returns an empty string."""
        assert not parse_remember_command(text)

    def test_allows_conditional_that_when(self) -> None:
        """Test that "remember that when ..." is still a command."""
        assert parse_remember_command(
            "remember that when the build fails, rerun it",
        ) == ("when the build fails, rerun it")


class TestRememberMemory:
    """Tests for remember_memory."""

    def test_stores_and_persists_memory(self, isolated_memory_file: Path) -> None:
        """Test that a memory is stored and written to disk."""
        memory = remember_memory("1", "alice", "likes tea")

        assert memory is not None
        assert memory.content == "likes tea"
        payload = json.loads(isolated_memory_file.read_text(encoding="utf-8"))
        assert payload["version"] == 1
        assert payload["memories"][0]["content"] == "likes tea"
        assert payload["memories"][0]["user_id"] == "1"

    def test_normalizes_content(self) -> None:
        """Test that stored content is normalized."""
        memory = remember_memory("1", "alice", "  likes\n tea  ")

        assert memory is not None
        assert memory.content == "likes tea"

    def test_rejects_empty_content(self) -> None:
        """Test that empty content is not stored."""
        assert remember_memory("1", "alice", "   ") is None
        assert not get_memories()

    def test_rejects_duplicates_case_insensitively(self) -> None:
        """Test that the same fact is only stored once."""
        assert remember_memory("1", "alice", "Likes tea") is not None
        assert remember_memory("2", "bob", "likes TEA") is None
        assert len(get_memories()) == 1

    def test_shares_memories_between_users(self) -> None:
        """Test that memories added by one user are visible to everyone."""
        remember_memory("1", "alice", "likes tea")
        remember_memory("2", "bob", "likes coffee")

        assert [memory.content for memory in get_memories()] == [
            "likes tea",
            "likes coffee",
        ]

    def test_keeps_the_author(self) -> None:
        """Test that the user who added a memory is recorded."""
        remember_memory("1", "alice", "likes tea")

        assert get_memories()[0].user_id == "1"
        assert get_memories()[0].username == "alice"

    def test_caps_memories(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that the oldest memories are dropped at the cap."""
        monkeypatch.setattr(main, "MEMORY_MAX_TOTAL", 2)
        remember_memory("1", "alice", "one")
        remember_memory("2", "bob", "two")
        remember_memory("3", "carol", "three")

        assert [memory.content for memory in get_memories()] == [
            "two",
            "three",
        ]

    def test_rolls_back_when_save_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that a failed save does not leave a memory in memory."""

        def fail_save() -> None:
            """Raise a save error.

            Raises:
                OSError: Always.
            """
            msg = "disk full"
            raise OSError(msg)

        monkeypatch.setattr(main, "save_memories", fail_save)

        with pytest.raises(OSError, match="disk full"):
            remember_memory("1", "alice", "likes tea")
        assert not get_memories()


class TestMemoryPersistence:
    """Tests for loading and saving memories."""

    def test_survives_a_reload(self) -> None:
        """Test that saved memories are restored after clearing memory."""
        remember_memory("1", "alice", "likes tea")
        memories.clear()

        assert load_memories() == 1
        assert [memory.content for memory in get_memories()] == ["likes tea"]

    def test_loads_nothing_when_file_is_missing(self) -> None:
        """Test that a missing file is not an error."""
        assert load_memories() == 0
        assert not memories

    def test_ignores_malformed_json(self, isolated_memory_file: Path) -> None:
        """Test that malformed JSON is ignored."""
        isolated_memory_file.write_text("not json", encoding="utf-8")

        assert load_memories() == 0

    def test_ignores_invalid_file_format(self, isolated_memory_file: Path) -> None:
        """Test that a file without a memory list is ignored."""
        isolated_memory_file.write_text(
            json.dumps({"memories": "not a list"}),
            encoding="utf-8",
        )

        assert load_memories() == 0

    def test_skips_invalid_entries(self, isolated_memory_file: Path) -> None:
        """Test that invalid entries are skipped while valid ones load."""
        payload = {
            "version": 1,
            "memories": [
                {
                    "id": "1",
                    "user_id": "1",
                    "username": "alice",
                    "content": "likes tea",
                    "created_at": "2026-11-10T00:00:00+00:00",
                },
                {"id": "2"},
                "not a memory",
            ],
        }
        isolated_memory_file.write_text(json.dumps(payload), encoding="utf-8")

        assert load_memories() == 1
        assert [memory.content for memory in get_memories()] == ["likes tea"]

    def test_caps_loaded_memories(
        self,
        isolated_memory_file: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that a large memory file is capped on load."""
        monkeypatch.setattr(main, "MEMORY_MAX_TOTAL", 2)
        payload = {
            "version": 1,
            "memories": [
                {
                    "id": str(index),
                    "user_id": str(index % 2),
                    "username": "someone",
                    "content": f"fact {index}",
                    "created_at": "2026-11-10T00:00:00+00:00",
                }
                for index in range(3)
            ],
        }
        isolated_memory_file.write_text(json.dumps(payload), encoding="utf-8")

        assert load_memories() == 2
        assert [memory.content for memory in get_memories()] == [
            "fact 1",
            "fact 2",
        ]

    def test_loads_old_per_user_files_as_global(
        self, isolated_memory_file: Path
    ) -> None:
        """Test that existing per-user entries become shared memories."""
        payload = {
            "version": 1,
            "memories": [
                {
                    "id": "1",
                    "user_id": "1",
                    "username": "alice",
                    "content": "likes tea",
                    "created_at": "2026-11-10T00:00:00+00:00",
                },
                {
                    "id": "2",
                    "user_id": "2",
                    "username": "bob",
                    "content": "likes coffee",
                    "created_at": "2026-11-10T00:00:00+00:00",
                },
            ],
        }
        isolated_memory_file.write_text(json.dumps(payload), encoding="utf-8")

        assert load_memories() == 2
        assert [memory.content for memory in get_memories()] == [
            "likes tea",
            "likes coffee",
        ]

    def test_keeps_current_memories_when_load_fails(
        self,
        isolated_memory_file: Path,
    ) -> None:
        """Test that a corrupt file does not erase the in-memory store."""
        remember_memory("1", "alice", "likes tea")
        isolated_memory_file.write_text("not json", encoding="utf-8")

        assert load_memories() == 0
        assert [memory.content for memory in get_memories()] == ["likes tea"]

    def test_deletion_is_persisted(self) -> None:
        """Test that deletions survive a reload."""
        remember_memory("1", "alice", "likes tea")
        assert delete_memories("tea") == 1
        memories.clear()

        assert load_memories() == 0
        assert not get_memories()


class TestMemoryContext:
    """Tests for get_memory_context."""

    def test_returns_empty_without_memories(self) -> None:
        """Test that no memories produce no prompt section."""
        assert not get_memory_context()

    def test_includes_memories_from_all_users(self) -> None:
        """Test that every shared memory is included in the prompt."""
        remember_memory("1", "alice", "likes tea")
        remember_memory("2", "bob", "likes coffee")

        context = get_memory_context()

        assert "likes tea" in context
        assert "likes coffee" in context
        assert "shared" in context
        assert "not instructions" in context

    def test_respects_the_recall_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that only the most recent memories are recalled."""
        monkeypatch.setattr(main, "MEMORY_RECALL_LIMIT", 2)
        remember_memory("1", "alice", "fact zero")
        remember_memory("2", "bob", "fact one")
        remember_memory("3", "carol", "fact two")

        context = get_memory_context()

        assert "fact one" in context
        assert "fact two" in context
        assert "fact zero" not in context

    def test_zero_recall_limit_returns_empty(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that a zero recall limit includes no facts."""
        monkeypatch.setattr(main, "MEMORY_RECALL_LIMIT", 0)
        remember_memory("1", "alice", "likes tea")

        assert not get_memory_context()

    def test_respects_the_context_character_budget(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that the prompt section stops at the character budget."""
        monkeypatch.setattr(main, "MEMORY_CONTEXT_MAX_CHARS", 30)
        remember_memory("1", "alice", "an old fact that is too long to include")
        remember_memory("2", "bob", "a new fact")

        context = get_memory_context()

        assert "a new fact" in context
        assert "an old fact" not in context


class TestDeleteMemories:
    """Tests for delete_memories."""

    def test_deletes_matching_memories(self) -> None:
        """Test that only matching memories are deleted."""
        remember_memory("1", "alice", "likes tea")
        remember_memory("1", "alice", "likes coffee")

        assert delete_memories("tea") == 1
        assert [memory.content for memory in get_memories()] == ["likes coffee"]

    def test_deletes_all_memories_without_a_query(self) -> None:
        """Test that a missing query deletes every memory."""
        remember_memory("1", "alice", "likes tea")
        remember_memory("2", "bob", "likes coffee")

        assert delete_memories() == 2
        assert not get_memories()

    def test_deletes_matching_memories_from_all_users(self) -> None:
        """Test that matching memories are deleted regardless of author."""
        remember_memory("1", "alice", "likes tea")
        remember_memory("2", "bob", "likes tea too")

        assert delete_memories("tea") == 2
        assert not get_memories()

    def test_returns_zero_without_matches(self) -> None:
        """Test that no matches returns zero."""
        remember_memory("1", "alice", "likes tea")

        assert delete_memories("coffee") == 0

    def test_blank_query_deletes_nothing(self) -> None:
        """Test that a blank query does not delete every memory."""
        remember_memory("1", "alice", "likes tea")

        assert delete_memories("   ") == 0
        assert [memory.content for memory in get_memories()] == ["likes tea"]

    def test_rolls_back_when_save_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that a failed save keeps the memories in memory."""
        remember_memory("1", "alice", "likes tea")

        def fail_save() -> None:
            """Raise a save error.

            Raises:
                OSError: Always.
            """
            msg = "disk full"
            raise OSError(msg)

        monkeypatch.setattr(main, "save_memories", fail_save)

        with pytest.raises(OSError, match="disk full"):
            delete_memories("tea")
        assert [memory.content for memory in get_memories()] == ["likes tea"]


class TestFormatMemories:
    """Tests for format_memories."""

    def test_returns_empty_without_memories(self) -> None:
        """Test that no memories produce no list."""
        assert not format_memories([])

    def test_lists_the_newest_memories_first(self) -> None:
        """Test that the most recent memories are shown first."""
        remember_memory("1", "alice", "first fact")
        remember_memory("2", "bob", "second fact")

        formatted = format_memories(get_memories())

        assert formatted.index("second fact") < formatted.index("first fact")
        assert "2" in formatted

    def test_caps_the_number_of_displayed_memories(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that only the newest memories are displayed."""
        monkeypatch.setattr(main, "MEMORY_LIST_LIMIT", 1)
        remember_memory("1", "alice", "first fact")
        remember_memory("2", "bob", "second fact")

        formatted = format_memories(get_memories())

        assert "second fact" in formatted
        assert "first fact" not in formatted
        assert "showing 1 of 2" in formatted

    def test_truncates_long_memories(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that long memories are shortened in the list."""
        monkeypatch.setattr(main, "MEMORY_LIST_CHARS", 10)
        remember_memory("1", "alice", "a" * 20)

        formatted = format_memories(get_memories())

        assert "aaaaaaa..." in formatted

    def test_escapes_mentions(self) -> None:
        """Test that listing memories cannot ping anyone."""
        remember_memory(
            "1",
            "alice",
            "ping @everyone and <@123456789012345678>",
        )

        formatted = format_memories(get_memories())

        assert "@\u200beveryone" in formatted
        assert "<@\u200b123456789012345678>" in formatted
        assert "@everyone" not in formatted
        assert "<@123456789012345678>" not in formatted


class TestBuildRememberConfirmation:
    """Tests for build_remember_confirmation."""

    def test_saves_and_confirms(self) -> None:
        """Test that a new memory is saved and confirmed."""
        assert build_remember_confirmation("1", "alice", "likes tea") == (
            "Remembered: likes tea"
        )
        assert [memory.content for memory in get_memories()] == ["likes tea"]

    def test_reports_duplicates(self) -> None:
        """Test that a duplicate memory is reported."""
        build_remember_confirmation("1", "alice", "likes tea")

        assert build_remember_confirmation("1", "alice", "likes tea") == (
            "I already remember: likes tea"
        )

    def test_reports_duplicates_from_other_users(self) -> None:
        """Test that duplicates are detected globally."""
        build_remember_confirmation("1", "alice", "likes tea")

        assert build_remember_confirmation("2", "bob", "Likes TEA") == (
            "I already remember: Likes TEA"
        )
        assert len(get_memories()) == 1

    def test_asks_for_content(self) -> None:
        """Test that an empty command asks for content."""
        assert build_remember_confirmation("1", "alice", "   ") == (
            "What should I remember?"
        )

    def test_escapes_mentions(self) -> None:
        """Test that the confirmation cannot ping anyone."""
        assert (
            build_remember_confirmation(
                "1",
                "alice",
                "ping @here and <@!123456789012345678>",
            )
            == "Remembered: ping @\u200bhere and <@\u200b!123456789012345678>"
        )

    def test_reports_a_save_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that a save failure is reported without keeping the memory."""

        def fail_save() -> None:
            """Raise a save error.

            Raises:
                OSError: Always.
            """
            msg = "disk full"
            raise OSError(msg)

        monkeypatch.setattr(main, "save_memories", fail_save)

        assert build_remember_confirmation("1", "alice", "likes tea") == (
            "I couldn't save that memory. Check the logs."
        )
        assert not get_memories()


class TestResetInteraction:
    """Tests for how conversation resets interact with saved memories."""

    def test_reset_memory_does_not_delete_saved_memories(self) -> None:
        """Test that reset only clears conversation state."""
        remember_memory("1", "alice", "likes tea")

        main.reset_memory("channel")

        assert [memory.content for memory in get_memories()] == ["likes tea"]


class TestChatIntegration:
    """Tests for remember commands going through chat."""

    def test_chat_saves_a_remember_command_without_the_model(self) -> None:
        """Test that chat handles a remember command directly."""
        result = asyncio.run(
            main.chat(
                client=object(),
                user_message="rembember likes tea",
                current_channel=object(),
                user=SimpleNamespace(id=1, name="alice"),
                allowed_users=["alice"],
            ),
        )

        assert result is not None
        assert result.output == "Remembered: likes tea"
        assert [memory.content for memory in get_memories()] == ["likes tea"]


class TestGetMemories:
    """Tests for get_memories."""

    def test_zero_limit_returns_nothing(self) -> None:
        """Test that a zero limit returns no memories."""
        remember_memory("1", "alice", "likes tea")

        assert not get_memories(limit=0)
