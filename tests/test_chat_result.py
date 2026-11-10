from __future__ import annotations

import json

from pydantic_ai.messages import ModelRequest
from pydantic_ai.messages import ModelResponse
from pydantic_ai.messages import TextPart
from pydantic_ai.messages import ThinkingPart
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.messages import ToolReturnPart

from main import SOURCES_MAX
from main import THINKING_MAX_CHARS
from main import ChatResult
from main import append_sources
from main import build_responses
from main import compose_response
from main import format_thinking
from main import get_sources
from main import get_thinking


def _search_return(urls: list[str]) -> ModelRequest:
    """Build a model request containing a web_search tool return.

    Args:
        urls (list[str]): The URLs to include in the search results.

    Returns:
        ModelRequest: The model request with the tool return.
    """
    return ModelRequest([
        ToolReturnPart(
            tool_name="web_search",
            content=json.dumps([{"url": url} for url in urls]),
        ),
    ])


def _fetch_call(url: str) -> ModelResponse:
    """Build a model response containing a fetch_web_page tool call.

    Args:
        url (str): The URL to fetch.

    Returns:
        ModelResponse: The model response with the tool call.
    """
    return ModelResponse([
        ToolCallPart(tool_name="fetch_web_page", args={"url": url}),
    ])


class TestGetThinking:
    """Tests for the get_thinking function."""

    def test_extracts_thinking_parts(self) -> None:
        """Test that reasoning parts are joined in order."""
        messages = [
            ModelResponse([
                ThinkingPart(content="First thought."),
                TextPart(content="Answer."),
            ]),
            ModelRequest([]),
            ModelResponse([ThinkingPart(content="Second thought.")]),
        ]

        assert get_thinking(messages) == "First thought.\n\nSecond thought."

    def test_returns_empty_without_thinking(self) -> None:
        """Test that a response without reasoning returns an empty string."""
        assert not get_thinking([ModelResponse([TextPart(content="Answer.")])])

    def test_ignores_blank_thinking(self) -> None:
        """Test that whitespace-only reasoning is ignored."""
        assert not get_thinking([ModelResponse([ThinkingPart(content="  ")])])


class TestGetSources:
    """Tests for the get_sources function."""

    def test_extracts_search_and_fetch_urls(self) -> None:
        """Test that URLs from search results and fetches are combined."""
        messages = [
            _search_return(["https://example.com/a", "https://example.com/b"]),
            _fetch_call("https://example.com/c"),
        ]

        assert get_sources(messages) == [
            "https://example.com/a",
            "https://example.com/b",
            "https://example.com/c",
        ]

    def test_deduplicates_urls(self) -> None:
        """Test that repeated URLs are only returned once."""
        messages = [
            _search_return(["https://example.com/a"]),
            _fetch_call("https://example.com/a"),
        ]

        assert get_sources(messages) == ["https://example.com/a"]

    def test_ignores_malformed_search_results(self) -> None:
        """Test that malformed search tool returns are ignored."""
        messages = [
            ModelRequest([
                ToolReturnPart(tool_name="web_search", content="not json"),
                ToolReturnPart(
                    tool_name="web_search",
                    content=json.dumps([{"url": "ftp://example.com/a"}]),
                ),
            ]),
        ]

        assert get_sources(messages) == []

    def test_ignores_other_tools(self) -> None:
        """Test that unrelated tool calls and returns are ignored."""
        messages = [
            ModelResponse([
                ToolCallPart(tool_name="other_tool", args={"url": "https://a.example"}),
            ]),
            ModelRequest([
                ToolReturnPart(
                    tool_name="other_tool",
                    content=json.dumps([{"url": "https://b.example"}]),
                ),
            ]),
        ]

        assert get_sources(messages) == []

    def test_caps_number_of_sources(self) -> None:
        """Test that the source list is capped."""
        urls = [f"https://example.com/{i}" for i in range(SOURCES_MAX + 5)]

        assert len(get_sources([_search_return(urls)])) == SOURCES_MAX


class TestAppendSources:
    """Tests for the append_sources function."""

    def test_appends_missing_sources(self) -> None:
        """Test that a source section is added to the answer."""
        assert append_sources("Answer", ["https://example.com/a"]) == (
            "Answer\n\n**Sources:**\n- <https://example.com/a>"
        )

    def test_skips_sources_already_in_the_answer(self) -> None:
        """Test that URLs already in the answer are not repeated."""
        assert append_sources(
            "See https://example.com/a",
            ["https://example.com/a", "https://example.com/b"],
        ) == ("See https://example.com/a\n\n**Sources:**\n- <https://example.com/b>")

    def test_leaves_answer_unchanged_without_missing_sources(self) -> None:
        """Test that an answer with all sources stays untouched."""
        answer: str = "See https://example.com/a"
        assert append_sources(answer, ["https://example.com/a"]) == answer


class TestFormatThinking:
    """Tests for the format_thinking function."""

    def test_returns_empty_for_blank_thinking(self) -> None:
        """Test that blank reasoning produces no message."""
        assert not format_thinking("   ")

    def test_formats_thinking(self) -> None:
        """Test that reasoning is formatted as a code block."""
        assert format_thinking("Step one.") == "```\nStep one.\n```"

    def test_uses_longer_fence_for_backticks(self) -> None:
        """Test that code fences in the reasoning do not close the block."""
        assert format_thinking("Use ```python\nprint(1)\n```") == (
            "````\nUse ```python\nprint(1)\n```\n````"
        )

    def test_truncates_long_thinking(self) -> None:
        """Test that long reasoning is truncated."""
        result = format_thinking("a" * (THINKING_MAX_CHARS + 50))

        assert result == f"```\n{'a' * (THINKING_MAX_CHARS - 3)}...\n```"


class TestComposeResponse:
    """Tests for the compose_response function."""

    def test_puts_thinking_before_the_answer(self) -> None:
        """Test that reasoning is placed in a code block before the answer."""
        assert compose_response("Answer.", "Step one.") == (
            "```\nStep one.\n```\nAnswer."
        )

    def test_returns_only_the_answer_without_thinking(self) -> None:
        """Test that no code block is added when there is no reasoning."""
        assert compose_response("Answer.", "") == "Answer."


class TestBuildResponses:
    """Tests for the build_responses function."""

    def test_builds_memory_and_display_answers(self) -> None:
        """Test that sources are in both versions and thinking only in display."""
        chat_result = ChatResult(
            output="Answer.",
            thinking="Step one.",
            sources=["https://example.com/a"],
        )
        memory_response, display_response = build_responses(chat_result)

        assert memory_response == ("Answer.\n\n**Sources:**\n- <https://example.com/a>")
        assert display_response == (
            "```\nStep one.\n```\nAnswer.\n\n**Sources:**\n- <https://example.com/a>"
        )

    def test_returns_fallback_without_a_result(self) -> None:
        """Test that a missing result returns the fallback for both versions."""
        assert build_responses(None) == (
            "I forgor how to think 💀",
            "I forgor how to think 💀",
        )

    def test_returns_fallback_without_model_output(self) -> None:
        """Test that an empty output returns the fallback for both versions."""
        assert build_responses(ChatResult(output="", thinking="Step one.")) == (
            "I forgor how to think 💀",
            "I forgor how to think 💀",
        )
