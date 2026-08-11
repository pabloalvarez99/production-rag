"""Settings: defaults, normalisation, secret handling and caching."""

from __future__ import annotations

import logging
from collections.abc import Callable

import pytest
from pydantic import ValidationError

from production_rag.config import Settings, get_settings

SettingsFactory = Callable[..., Settings]


def test_defaults(settings: Settings) -> None:
    """Defaults are a runnable local configuration with no credentials."""
    assert settings.app_name == "production-rag"
    assert settings.app_version == "0.1.0"
    assert settings.environment == "local"
    assert settings.log_level == "INFO"
    assert settings.api_prefix == "/v1"
    assert settings.host == "0.0.0.0"  # noqa: S104 - container-friendly default, asserted on purpose
    assert settings.port == 8000
    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.qdrant_collection == "production_rag"
    assert settings.config_path is None
    assert settings.openai_api_key is None
    assert settings.qdrant_api_key is None


def test_collection_default_matches_the_rest_of_the_repo(settings: Settings) -> None:
    """One collection name across settings, YAML, compose and the health script.

    Two defaults that disagree produce the worst kind of bug report: ingest
    reports success, the probe reports an empty collection, and both are telling
    the truth about different collections.
    """
    from production_rag.config_loader import QdrantConfig

    assert settings.qdrant_collection == QdrantConfig().collection == "production_rag"


def test_env_file_is_wired_for_the_real_settings_class() -> None:
    """Production settings still read ``.env``; only the tests opt out of it."""
    assert Settings.model_config["env_file"] == ".env"
    assert Settings.model_config["extra"] == "ignore"


def test_unknown_environment_variables_are_ignored(
    monkeypatch: pytest.MonkeyPatch, settings_factory: SettingsFactory
) -> None:
    """Compose and Qdrant share the ``.env``; their keys must not break startup."""
    monkeypatch.setenv("QDRANT__TELEMETRY_DISABLED", "true")
    monkeypatch.setenv("SOME_UNRELATED_TOOL_FLAG", "1")

    loaded = settings_factory()

    assert loaded.app_name == "production-rag"
    assert not hasattr(loaded, "some_unrelated_tool_flag")


def test_values_come_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, settings_factory: SettingsFactory
) -> None:
    monkeypatch.setenv("APP_NAME", "rag-prod")
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("PORT", "9001")
    monkeypatch.setenv("QDRANT_COLLECTION", "handbook")

    from_env = settings_factory()

    assert from_env.app_name == "rag-prod"
    assert from_env.environment == "production"
    assert from_env.port == 9001  # coerced from the string the environment gives us
    assert from_env.qdrant_collection == "handbook"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("/v1", "/v1"),
        ("v1", "/v1"),
        ("/v1/", "/v1"),
        ("  v2  ", "/v2"),
        ("/api/v1", "/api/v1"),
    ],
)
def test_api_prefix_is_normalised(
    settings_factory: SettingsFactory, raw: str, expected: str
) -> None:
    """One canonical form, so route mounting never produces a double slash."""
    assert settings_factory(api_prefix=raw).api_prefix == expected


@pytest.mark.parametrize("raw", ["", "/", "   ", "//"])
def test_empty_api_prefix_is_rejected(settings_factory: SettingsFactory, raw: str) -> None:
    """A root prefix would collide with the unversioned ``/health`` route.

    Failing at startup is much cheaper than shipping an app whose OpenAPI
    document has two operations on the same path.
    """
    with pytest.raises(ValidationError, match="version segment"):
        settings_factory(api_prefix=raw)


@pytest.mark.parametrize(("raw", "expected"), [("debug", "DEBUG"), ("Warning", "WARNING")])
def test_log_level_is_case_insensitive(
    settings_factory: SettingsFactory, raw: str, expected: str
) -> None:
    assert settings_factory(log_level=raw).log_level == expected


def test_unknown_log_level_is_rejected(settings_factory: SettingsFactory) -> None:
    with pytest.raises(ValidationError, match="log_level must be one of"):
        settings_factory(log_level="chatty")


def test_log_level_number(settings_factory: SettingsFactory) -> None:
    assert settings_factory(log_level="debug").log_level_number == logging.DEBUG
    assert settings_factory().log_level_number == logging.INFO


@pytest.mark.parametrize(
    ("qdrant_url", "expected"),
    [
        ("http://localhost:6333", True),
        ("https://qdrant.internal", True),
        ("", False),
        ("   ", False),
        ("localhost:6333", False),
    ],
)
def test_qdrant_configured(
    settings_factory: SettingsFactory, qdrant_url: str, expected: bool
) -> None:
    assert settings_factory(qdrant_url=qdrant_url).qdrant_configured is expected


def test_safe_dump_masks_the_api_key(settings_factory: SettingsFactory) -> None:
    """``safe_dump`` is the only sanctioned way to serialise settings."""
    secret = "sk-live-do-not-log-me"  # noqa: S105 - fake value for the assertion
    configured = settings_factory(openai_api_key=secret)

    dumped = configured.safe_dump()

    assert dumped["openai_api_key"] == "***"
    assert secret not in str(dumped)
    # Non-secret fields survive untouched, otherwise the dump is useless.
    assert dumped["app_name"] == "production-rag"
    # And the guard is needed: the plain pydantic dump does expose the value,
    # which is exactly why nothing in the codebase calls model_dump() for logs.
    assert configured.model_dump()["openai_api_key"] == secret


def test_safe_dump_masks_every_declared_secret(settings_factory: SettingsFactory) -> None:
    """Every field in ``SECRET_FIELDS`` is masked, not only the OpenAI one."""
    configured = settings_factory(
        openai_api_key="sk-live-not-real",  # noqa: S106 - fake value for the assertion
        qdrant_api_key="qdrant-live-not-real",  # noqa: S106 - fake value for the assertion
    )
    dumped = configured.safe_dump()

    assert set(Settings.SECRET_FIELDS) == {"openai_api_key", "qdrant_api_key"}
    assert all(dumped[name] == "***" for name in Settings.SECRET_FIELDS)


def test_safe_dump_leaves_absent_secrets_as_none(settings: Settings) -> None:
    """An unset credential stays ``None``, so ``***`` always means "one is set"."""
    assert settings.safe_dump()["openai_api_key"] is None


def test_secret_fields_is_not_a_pydantic_field(settings: Settings) -> None:
    """The registry of secret names must not become part of the settings schema."""
    assert "SECRET_FIELDS" not in Settings.model_fields
    assert "openai_api_key" in settings.SECRET_FIELDS


def test_get_settings_is_cached() -> None:
    """One settings object per process: ``.env`` is parsed once, not per request."""
    assert get_settings() is get_settings()

    get_settings.cache_clear()

    assert get_settings() is not None
