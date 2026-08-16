from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import httpx
import pytest

from main import GiteaRepoSearchResponse
from main import parse_gitea_repo
from main import parse_repo_search_results
from main import search_gitea_repos

if TYPE_CHECKING:
    from collections.abc import Callable


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

    @pytest.fixture(autouse=True)
    def _clear_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ensure no network request is made by clearing the token."""
        monkeypatch.setattr("main.gitea_token", "")

    @staticmethod
    def _mock_http(
        monkeypatch: pytest.MonkeyPatch,
        handler: Callable[[httpx.Request], httpx.Response],
    ) -> None:
        """Patch the HTTP client with a mock transport."""
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

    def test_returns_empty_without_token(self) -> None:
        """Test that an empty token returns an empty list."""
        assert asyncio.run(search_gitea_repos("TheLovinator")) == []

    def test_returns_repo_names(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that a successful search returns the repo full names."""
        self._mock_http(
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

        self._mock_http(monkeypatch, handler)

        assert asyncio.run(search_gitea_repos("")) == []
        assert seen_params[0].get("sort") == "updated"
        assert seen_params[0].get("order") == "desc"
        assert seen_params[0].get("order_by") == "desc"

    def test_returns_empty_when_data_is_not_a_list(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that a malformed response returns an empty list."""
        self._mock_http(
            monkeypatch,
            lambda request: httpx.Response(200, json={"ok": True, "data": "nope"}),
        )

        assert asyncio.run(search_gitea_repos("anew")) == []

    def test_returns_empty_on_http_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that a non-success response returns an empty list."""
        self._mock_http(
            monkeypatch,
            lambda request: httpx.Response(401, json={"message": "Unauthorized"}),
        )

        assert asyncio.run(search_gitea_repos("anew")) == []
