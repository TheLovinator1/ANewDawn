from __future__ import annotations

from unittest.mock import Mock

import pytest

from misc import get_trigger_keywords


@pytest.fixture
def mock_bot() -> Mock:
    """Create a mock bot instance.

    Returns:
        A mock bot instance.
    """
    return Mock()


def test_get_trigger_keywords_with_bot_user(mock_bot: Mock) -> None:
    """Test getting trigger keywords with a bot user."""
    mock_bot.get_me.return_value.id = 123456789
    expected_keywords: list[str] = ["lovibot", "<@123456789>"]

    result: list[str] = get_trigger_keywords(mock_bot)

    assert result == expected_keywords


def test_get_trigger_keywords_without_bot_user(mock_bot: Mock) -> None:
    """Test getting trigger keywords without a bot user."""
    mock_bot.get_me.return_value = None
    expected_keywords: list[str] = ["lovibot", ""]

    result: list[str] = get_trigger_keywords(mock_bot)

    assert result == expected_keywords
