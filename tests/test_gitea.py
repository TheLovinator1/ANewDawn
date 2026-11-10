from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

import httpx
import pytest

from main import GiteaRepoSearchResponse
from main import create_issue
from main import parse_gitea_repo
from main import parse_repo_search_results
from main import search_gitea_repos

if TYPE_CHECKING:
    from collections.abc import Callable


def mock_gitea_http(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx.Request], httpx.Response],
) -> None:
    """Patch the Gitea HTTP client with a mock transport.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture.
        handler (Callable[[httpx.Request], httpx.Response]): The mock handler.
    """
    monkeypatch.setattr("main.gitea_token", "test-token")

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


class TestParseGiteaRepo:
    """Tests for the parse_gitea_repo function."""

    @pytest.mark.parametrize(
        ("input_text", "expected"),
        [
            ("TheLovinator/ANewDawn", ("TheLovinator", "ANewDawn")),
            (
                "git.lovinator.space/TheLovinator/ANewDawn",
                ("TheLovinator", "ANewDawn"),
            ),
            (
                "https://git.lovinator.space/TheLovinator/ANewDawn",
                ("TheLovinator", "ANewDawn"),
            ),
            (
                "https://git.lovinator.space/TheLovinator/ANewDawn.git",
                ("TheLovinator", "ANewDawn"),
            ),
            (
                "https://git.lovinator.space/TheLovinator/ANewDawn/",
                ("TheLovinator", "ANewDawn"),
            ),
            ("  TheLovinator/ANewDawn  ", ("TheLovinator", "ANewDawn")),
            (
                "TheLovinator/ANewDawn.git",
                ("TheLovinator", "ANewDawn"),
            ),
            (
                "https://example.com/TheLovinator/ANewDawn",
                ("TheLovinator", "ANewDawn"),
            ),
        ],
    )
    def test_valid_inputs(
        self,
        input_text: str,
        expected: tuple[str, str],
    ) -> None:
        """Test that valid inputs are parsed correctly."""
        assert parse_gitea_repo(input_text) == expected

    @pytest.mark.parametrize(
        "input_text",
        [
            "",
            "ANewDawn",
            "TheLovinator/",
            "https://git.lovinator.space/TheLovinator",
            "TheLovinator/ANewDawn/issues/1",
            "TheLovinator/ANew Dawn",
            "TheLovinator/ANewDawn extra",
        ],
    )
    def test_invalid_inputs(self, input_text: str) -> None:
        """Test that invalid inputs return None."""
        assert parse_gitea_repo(input_text) is None


class TestParseRepoSearchResults:
    """Tests for the parse_repo_search_results function."""

    def test_returns_repo_names(self) -> None:
        """Test that repository full names are extracted from the response."""
        response: GiteaRepoSearchResponse = {
            "ok": True,
            "data": [
                {"full_name": "TheLovinator/ANewDawn"},
                {"full_name": "TheLovinator/OtherRepo"},
            ],
        }
        assert parse_repo_search_results(response) == [
            "TheLovinator/ANewDawn",
            "TheLovinator/OtherRepo",
        ]

    def test_returns_empty_for_missing_data(self) -> None:
        """Test that a response without data returns an empty list."""
        response: GiteaRepoSearchResponse = {"ok": True}
        assert parse_repo_search_results(response) == []


class TestSearchGiteaRepos:
    """Tests for the search_gitea_repos function."""

    @pytest.fixture(autouse=True)  # ruff: ignore[pytest-fixture-autouse]
    def _clear_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ensure no network request is made by clearing the token."""
        monkeypatch.setattr("main.gitea_token", "")

    def test_returns_empty_without_token(self) -> None:
        """Test that an empty token returns an empty list."""
        assert asyncio.run(search_gitea_repos("TheLovinator")) == []

    def test_returns_repo_names(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that a successful search returns the repo full names."""
        mock_gitea_http(
            monkeypatch,
            lambda request: httpx.Response(
                200,
                json={
                    "ok": True,
                    "data": [
                        {"full_name": "TheLovinator/ANewDawn"},
                        {"full_name": "TheLovinator/OtherRepo"},
                    ],
                },
            ),
        )

        assert asyncio.run(search_gitea_repos("anew")) == [
            "TheLovinator/ANewDawn",
            "TheLovinator/OtherRepo",
        ]

    def test_sorts_by_most_recently_updated(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that the search request asks for updated-desc ordering."""
        seen_params: list[httpx.QueryParams] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_params.append(request.url.params)
            return httpx.Response(200, json={"ok": True, "data": []})

        mock_gitea_http(monkeypatch, handler)

        assert asyncio.run(search_gitea_repos("")) == []
        assert seen_params[0].get("sort") == "updated"
        assert seen_params[0].get("order") == "desc"
        assert seen_params[0].get("order_by") == "desc"

    def test_returns_empty_when_data_is_not_a_list(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that a malformed response returns an empty list."""
        mock_gitea_http(
            monkeypatch,
            lambda request: httpx.Response(200, json={"ok": True, "data": "nope"}),
        )

        assert asyncio.run(search_gitea_repos("anew")) == []

    def test_returns_empty_on_http_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that a non-success response returns an empty list."""
        mock_gitea_http(
            monkeypatch,
            lambda request: httpx.Response(401, json={"message": "Unauthorized"}),
        )

        assert asyncio.run(search_gitea_repos("anew")) == []


class TestCreateIssueTool:
    """Tests for the create_issue agent tool."""

    def test_creates_issue(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that an issue is created and the URL is returned."""
        seen_requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_requests.append(request)
            return httpx.Response(
                201,
                json={
                    "number": 42,
                    "html_url": (
                        "https://git.lovinator.space/TheLovinator/ANewDawn/issues/42"
                    ),
                },
            )

        mock_gitea_http(monkeypatch, handler)

        result: str = asyncio.run(
            create_issue(
                "TheLovinator/ANewDawn",
                "Bot crashes",
                "It crashes when creating an issue.",
            ),
        )

        assert result == (
            "Created issue `TheLovinator/ANewDawn#42`: "
            "https://git.lovinator.space/TheLovinator/ANewDawn/issues/42"
        )
        assert seen_requests[0].method == "POST"
        assert seen_requests[0].url.path == (
            "/api/v1/repos/TheLovinator/ANewDawn/issues"
        )
        assert json.loads(seen_requests[0].content) == {
            "title": "Bot crashes",
            "body": "It crashes when creating an issue.",
        }

    def test_creates_issue_with_a_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that a full repository URL is accepted."""
        mock_gitea_http(
            monkeypatch,
            lambda request: httpx.Response(
                201,
                json={"number": 7, "html_url": "https://example.test/issues/7"},
            ),
        )

        result: str = asyncio.run(
            create_issue(
                "https://git.lovinator.space/TheLovinator/ANewDawn",
                "Title",
            ),
        )

        assert "TheLovinator/ANewDawn#7" in result

    def test_uses_a_placeholder_title(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that a blank title gets a placeholder."""
        seen_requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_requests.append(request)
            return httpx.Response(201, json={"number": 1, "html_url": ""})

        mock_gitea_http(monkeypatch, handler)

        asyncio.run(create_issue("TheLovinator/ANewDawn", "   "))

        assert json.loads(seen_requests[0].content)["title"] == "Untitled issue"

    def test_resolves_a_bare_repository_name(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that a bare repository name is found through search."""
        seen_requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_requests.append(request)
            if request.method == "GET":
                return httpx.Response(
                    200,
                    json={
                        "ok": True,
                        "data": [{"full_name": "TheLovinator/ANewDawn"}],
                    },
                )
            return httpx.Response(
                201,
                json={"number": 9, "html_url": "https://example.test/issues/9"},
            )

        mock_gitea_http(monkeypatch, handler)

        result: str = asyncio.run(create_issue("ANewDawn", "Title"))

        assert "TheLovinator/ANewDawn#9" in result
        assert seen_requests[0].method == "GET"
        assert seen_requests[1].method == "POST"
        assert seen_requests[1].url.path == (
            "/api/v1/repos/TheLovinator/ANewDawn/issues"
        )

    def test_asks_when_a_repository_name_is_ambiguous(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that an ambiguous repository name is not guessed."""
        seen_requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_requests.append(request)
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "data": [
                        {"full_name": "TheLovinator/ANewDawn"},
                        {"full_name": "SomeoneElse/ANewDawn"},
                    ],
                },
            )

        mock_gitea_http(monkeypatch, handler)

        result: str = asyncio.run(create_issue("ANewDawn", "Title"))

        assert "Multiple repositories matched" in result
        assert "TheLovinator/ANewDawn" in result
        assert "SomeoneElse/ANewDawn" in result
        assert all(request.method == "GET" for request in seen_requests)

    def test_reports_a_missing_repository(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that a repository search with no matches is reported."""
        mock_gitea_http(
            monkeypatch,
            lambda request: httpx.Response(200, json={"ok": True, "data": []}),
        )

        result: str = asyncio.run(create_issue("ANewDawn", "Title"))

        assert "Could not find a repository matching `ANewDawn`" in result

    def test_rejects_an_invalid_repository(self) -> None:
        """Test that an invalid repository is rejected before any request."""
        result: str = asyncio.run(
            create_issue("TheLovinator/ANewDawn extra", "Title"),
        )

        assert "Could not parse" in result

    def test_reports_a_missing_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that a missing Gitea token is reported."""
        monkeypatch.setattr("main.gitea_token", "")

        result: str = asyncio.run(create_issue("TheLovinator/ANewDawn", "Title"))

        assert "GITEA_TOKEN is not set" in result

    def test_reports_an_api_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that a Gitea API error is returned to the model."""
        mock_gitea_http(
            monkeypatch,
            lambda request: httpx.Response(
                404,
                json={"message": "Not Found"},
            ),
        )

        result: str = asyncio.run(
            create_issue("TheLovinator/Missing", "Title"),
        )

        assert "Could not create the issue" in result
        assert "was not found" in result
