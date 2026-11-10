from __future__ import annotations

import pytest
import sentry_sdk


@pytest.fixture(scope="session", autouse=True)  # ruff: ignore[pytest-fixture-autouse]
def disable_sentry() -> None:
    """Disable Sentry so expected test errors are not reported."""
    sentry_sdk.init(dsn="")
