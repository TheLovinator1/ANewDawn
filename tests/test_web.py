from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

import httpx
import pytest

from main import DEEPSEEK_MAX_PAGE_CHARS
from main import DeepSeekSearchError
from main import deepseek_fetch
from main import deepseek_search
from main import fetch_web_page
from main import web_search

if TYPE_CHECKING:
    from collections.abc import Callable


def _mock_http(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx.Request], httpx.Response],
) -> None:
    """Patch the HTTP client with a mock transport."""
    monkeypatch.setattr("main.deepseek_api_key", "test-key")

    real_async_client: type[httpx.AsyncClient] = httpx.AsyncClient

    def make_client(*, timeout: float) -> httpx.AsyncClient:
        """Create a client that serves responses through the mock.

        Returns:
            httpx.AsyncClient: A client backed by the mock transport.
        """
        return real_async_client(
            transport=httpx.MockTransport(handler),
            timeout=timeout,
        )

    monkeypatch.setattr("main.httpx.AsyncClient", make_client)


class TestDeepSeekSearch:
    """Tests for the deepseek_search function."""

    def test_requires_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that a missing API key raises an error."""
        monkeypatch.setattr("main.deepseek_api_key", "")

        with pytest.raises(DeepSeekSearchError, match="DEEPSEEK_API_KEY"):
            asyncio.run(deepseek_search("DeepSeek"))

    def test_sends_expected_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that the request uses the search endpoint and bearer auth."""
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json={"results": []})

        _mock_http(monkeypatch, handler)

        assert asyncio.run(deepseek_search("DeepSeek V4")) == []

        request = requests[0]
        assert str(request.url) == "https://api.deepseek.com/v1/tools/search"
        assert request.headers["Authorization"] == "Bearer test-key"
        assert request.headers["Content-Type"] == "application/json"
        assert json.loads(request.content) == {
            "query": "DeepSeek V4",
            "count": 5,
            "need_content": False,
        }

    @pytest.mark.parametrize(("count", "expected"), [(0, 1), (100, 50), (10, 10)])
    def test_clamps_count(
        self,
        monkeypatch: pytest.MonkeyPatch,
        count: int,
        expected: int,
    ) -> None:
        """Test that the result count is clamped to the supported range."""
        payloads: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            payloads.append(json.loads(request.content))
            return httpx.Response(200, json={"results": []})

        _mock_http(monkeypatch, handler)

        asyncio.run(deepseek_search("DeepSeek", count=count))

        assert payloads[0]["count"] == expected

    def test_returns_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that search results are returned."""
        result = {
            "title": "DeepSeek V4",
            "url": "https://example.com/deepseek-v4",
            "summary": "DeepSeek V4 was released.",
            "date": "2026-06-01T00:00:00+00:00",
            "site_name": "Example",
        }
        _mock_http(
            monkeypatch,
            lambda request: httpx.Response(200, json={"results": [result]}),
        )

        assert asyncio.run(deepseek_search("DeepSeek V4")) == [result]

    def test_returns_empty_for_malformed_results(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that a response without a results list returns an empty list."""
        _mock_http(
            monkeypatch,
            lambda request: httpx.Response(200, json={"results": "nope"}),
        )

        assert asyncio.run(deepseek_search("DeepSeek")) == []

    def test_raises_api_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that API errors include the message from the response."""
        _mock_http(
            monkeypatch,
            lambda request: httpx.Response(
                401,
                json={
                    "error": {
                        "message": "Invalid API key",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "invalid_request_error",
                    },
                },
            ),
        )

        with pytest.raises(DeepSeekSearchError, match="HTTP 401: Invalid API key"):
            asyncio.run(deepseek_search("DeepSeek"))

    def test_raises_on_network_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that network errors are wrapped in a search error."""

        def handler(request: httpx.Request) -> httpx.Response:
            msg = "boom"
            raise httpx.ConnectError(msg, request=request)

        _mock_http(monkeypatch, handler)

        with pytest.raises(DeepSeekSearchError, match="Could not reach"):
            asyncio.run(deepseek_search("DeepSeek"))

    def test_raises_on_invalid_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that a successful response without JSON raises an error."""
        _mock_http(
            monkeypatch,
            lambda request: httpx.Response(200, text="not json"),
        )

        with pytest.raises(DeepSeekSearchError, match="invalid JSON"):
            asyncio.run(deepseek_search("DeepSeek"))


class TestDeepSeekFetch:
    """Tests for the deepseek_fetch function."""

    def test_sends_expected_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that fetch sends the URL to the fetch endpoint."""
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                200,
                json={"title": "Example", "content": "# Example\n\nHello."},
            )

        _mock_http(monkeypatch, handler)

        page = asyncio.run(deepseek_fetch("https://example.com/article"))

        assert page == {"title": "Example", "content": "# Example\n\nHello."}
        assert str(requests[0].url) == "https://api.deepseek.com/v1/tools/fetch"
        assert json.loads(requests[0].content) == {
            "url": "https://example.com/article",
        }

    def test_raises_api_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that a failed fetch raises a search error."""
        _mock_http(
            monkeypatch,
            lambda request: httpx.Response(
                404,
                json={"error": {"message": "No content found"}},
            ),
        )

        with pytest.raises(DeepSeekSearchError, match="HTTP 404: No content found"):
            asyncio.run(deepseek_fetch("https://example.com/missing"))


class TestWebSearchTool:
    """Tests for the agent's web_search tool."""

    def test_returns_json_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that the tool returns compact JSON with the useful fields."""
        _mock_http(
            monkeypatch,
            lambda request: httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "DeepSeek V4",
                            "url": "https://example.com/deepseek-v4",
                            "summary": "DeepSeek V4 was released.",
                            "snippet": "Short snippet.",
                            "content": "Full content.",
                            "date": "2026-06-01T00:00:00+00:00",
                            "site_name": "Example",
                            "site_icon": "https://example.com/favicon.ico",
                            "authority": 3,
                            "rerank_score": 2.4,
                        },
                    ],
                },
            ),
        )

        results = json.loads(asyncio.run(web_search("DeepSeek V4")))

        assert results == [
            {
                "title": "DeepSeek V4",
                "url": "https://example.com/deepseek-v4",
                "summary": "DeepSeek V4 was released.",
                "date": "2026-06-01T00:00:00+00:00",
                "site_name": "Example",
            },
        ]

    def test_returns_message_for_no_results(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that an empty result set is reported to the model."""
        _mock_http(
            monkeypatch,
            lambda request: httpx.Response(200, json={"results": []}),
        )

        assert asyncio.run(web_search("nonexistent")) == (
            "No web search results found for 'nonexistent'."
        )

    def test_returns_error_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that API errors are returned instead of raised."""
        _mock_http(
            monkeypatch,
            lambda request: httpx.Response(
                402,
                json={"error": {"message": "Insufficient balance"}},
            ),
        )

        assert asyncio.run(web_search("DeepSeek")) == (
            "Web search failed: DeepSeek Search API returned HTTP 402: "
            "Insufficient balance"
        )


class TestFetchWebPageTool:
    """Tests for the agent's fetch_web_page tool."""

    def test_returns_title_and_content(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that the tool returns the title and Markdown content."""
        _mock_http(
            monkeypatch,
            lambda request: httpx.Response(
                200,
                json={"title": "Example", "content": "# Example\n\nHello."},
            ),
        )

        assert asyncio.run(fetch_web_page("https://example.com")) == (
            "# Example\n\n# Example\n\nHello."
        )

    def test_truncates_long_content(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that very long page content is truncated."""
        content = "a" * (DEEPSEEK_MAX_PAGE_CHARS + 100)
        _mock_http(
            monkeypatch,
            lambda request: httpx.Response(
                200,
                json={"title": "Long page", "content": content},
            ),
        )

        result = asyncio.run(fetch_web_page("https://example.com/long"))

        assert result.endswith("[Content truncated because the page is very long.]")
        assert "a" * DEEPSEEK_MAX_PAGE_CHARS in result

    def test_returns_error_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that fetch errors are returned instead of raised."""
        _mock_http(
            monkeypatch,
            lambda request: httpx.Response(
                404,
                json={"error": {"message": "No content found"}},
            ),
        )

        assert asyncio.run(fetch_web_page("https://example.com/missing")) == (
            "Could not fetch https://example.com/missing: "
            "DeepSeek Fetch API returned HTTP 404: No content found"
        )
