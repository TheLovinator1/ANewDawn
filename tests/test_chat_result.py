from __future__ import annotations

import json

from pydantic_ai.messages import ModelRequest
from pydantic_ai.messages import ModelResponse
from pydantic_ai.messages import TextPart
from pydantic_ai.messages import ThinkingPart
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.messages import ToolReturnPart

from main import SOURCES_MAX
from main import ChatResult
from main import append_sources
from main import build_responses
from main import build_thinking_file
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
            "Answer\n\n**Sources:** [1](<https://example.com/a>)"
        )

    def test_numbers_multiple_sources_on_one_line(self) -> None:
        """Test that multiple sources are numbered on a single line."""
        assert append_sources(
            "Answer",
            ["https://example.com/a", "https://example.com/b"],
        ) == (
            "Answer\n\n**Sources:** "
            "[1](<https://example.com/a>) [2](<https://example.com/b>)"
        )

    def test_skips_sources_already_in_the_answer(self) -> None:
        """Test that URLs already in the answer are not repeated."""
        assert append_sources(
            "See https://example.com/a",
            ["https://example.com/a", "https://example.com/b"],
        ) == ("See https://example.com/a\n\n**Sources:** [1](<https://example.com/b>)")

    def test_leaves_answer_unchanged_without_missing_sources(self) -> None:
        """Test that an answer with all sources stays untouched."""
        answer: str = "See https://example.com/a"
        assert append_sources(answer, ["https://example.com/a"]) == answer


class TestBuildThinkingFile:
    """Tests for the build_thinking_file function."""

    def test_returns_none_without_reasoning(self) -> None:
        """Test that no file is created without reasoning or an answer."""
        assert build_thinking_file(None) is None
        assert build_thinking_file(ChatResult(output="Answer.")) is None
        assert (
            build_thinking_file(
                ChatResult(output="Answer.", thinking="   "),
            )
            is None
        )
        assert (
            build_thinking_file(
                ChatResult(output="", thinking="Step one."),
            )
            is None
        )

    def test_puts_reasoning_in_a_text_file(self) -> None:
        """Test that reasoning is written to a text attachment."""
        thinking_file = build_thinking_file(
            ChatResult(output="Answer.", thinking="Step one."),
        )

        assert thinking_file is not None
        assert thinking_file.filename == "thinking.txt"
        assert thinking_file.fp.read() == b"Step one."
        thinking_file.close()


class TestBuildResponses:
    """Tests for the build_responses function."""

    def test_builds_memory_and_display_answers(self) -> None:
        """Test that sources are in both versions and thinking is not."""
        chat_result = ChatResult(
            output="Answer.",
            thinking="Step one.",
            sources=["https://example.com/a"],
        )
        memory_response, display_response = build_responses(chat_result)

        expected: str = "Answer.\n\n**Sources:** [1](<https://example.com/a>)"
        assert memory_response == expected
        assert display_response == expected

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
