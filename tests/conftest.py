"""Shared fixtures for the production-rag test suite.

Two invariants the whole suite rests on:

* **No network.** Nothing here contacts Qdrant, OpenAI or any other service, so
  the suite is green on a laptop with the wifi off and inside a CI container
  with no egress. Readiness is a configuration check by design (see
  ``production_rag.api.routes.ready``), which is what makes that possible.
* **No ambient configuration.** Settings are built from explicit values with
  the on-disk ``.env`` disabled, so a developer's local ``.env`` — or a stray
  ``ENVIRONMENT`` exported in a shell — cannot turn a passing suite red.

Tests never import this module: everything is reached through fixtures.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import ExitStack
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic_settings import SettingsConfigDict

from production_rag.config import Settings, get_settings
from production_rag.main import create_app

ENV_KEYS = (
    "APP_NAME",
    "APP_VERSION",
    "ENVIRONMENT",
    "LOG_LEVEL",
    "API_PREFIX",
    "HOST",
    "PORT",
    "QDRANT_URL",
    "QDRANT_COLLECTION",
    "CONFIG_PATH",
    "OPENAI_API_KEY",
    "QDRANT_API_KEY",
)
"""Every environment variable :class:`Settings` reads, cleared before each test."""

SettingsFactory = Callable[..., Settings]
ClientFactory = Callable[[Settings], TestClient]


class _IsolatedSettings(Settings):
    """Settings that ignore any ``.env`` file on disk.

    Environment variables are still honoured — several tests exercise exactly
    that path — but the file source is switched off so assertions about default
    values describe the code, not the machine the suite happens to run on.
    """

    model_config = SettingsConfigDict(
        env_file=None,
        extra="ignore",
        case_sensitive=False,
    )


@pytest.fixture(autouse=True)
def _isolated_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Remove app environment variables and reset the settings cache."""
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
        monkeypatch.delenv(key.lower(), raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings_factory() -> SettingsFactory:
    """Build settings from explicit values, with the ``.env`` source disabled."""

    def _factory(**overrides: Any) -> Settings:
        return _IsolatedSettings(**overrides)

    return _factory


@pytest.fixture
def settings(settings_factory: SettingsFactory) -> Settings:
    """Default settings, isolated from the filesystem."""
    return settings_factory()


@pytest.fixture
def client_factory() -> Iterator[ClientFactory]:
    """Build a test client around arbitrary settings.

    Use this when a test needs non-default configuration; every client created
    through the factory is closed when the test finishes.
    """
    with ExitStack() as stack:

        def _factory(custom_settings: Settings) -> TestClient:
            return stack.enter_context(TestClient(create_app(custom_settings)))

        yield _factory


@pytest.fixture
def client(client_factory: ClientFactory, settings: Settings) -> TestClient:
    """Test client for an app built on default, isolated settings."""
    return client_factory(settings)
