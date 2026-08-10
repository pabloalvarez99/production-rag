"""Liveness endpoint behaviour.

Both the unversioned probe (wired into the Dockerfile and Compose
healthchecks) and the versioned one are covered: they are separate route
registrations, so a change that breaks only one of them is a realistic
regression.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient

from production_rag.config import Settings

HEALTH_PATHS = ["/health", "/v1/health"]
EXPECTED_KEYS = {"status", "service", "version", "environment"}

ClientFactory = Callable[[Settings], TestClient]
SettingsFactory = Callable[..., Settings]


@pytest.mark.parametrize("path", HEALTH_PATHS)
def test_health_returns_200(client: TestClient, path: str) -> None:
    response = client.get(path)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")


@pytest.mark.parametrize("path", HEALTH_PATHS)
def test_health_payload_shape(client: TestClient, path: str) -> None:
    payload = client.get(path).json()

    assert set(payload) == EXPECTED_KEYS
    assert payload == {
        "status": "ok",
        "service": "production-rag",
        "version": "0.1.0",
        "environment": "local",
    }


def test_versioned_and_unversioned_payloads_are_identical(client: TestClient) -> None:
    """One payload, two paths: they must not drift apart."""
    assert client.get("/health").json() == client.get("/v1/health").json()


def test_health_reflects_configured_identity(
    client_factory: ClientFactory, settings_factory: SettingsFactory
) -> None:
    """Identity fields come from settings, not from hardcoded literals."""
    custom = settings_factory(app_name="rag-staging", app_version="9.9.9", environment="staging")

    payload = client_factory(custom).get("/health").json()

    assert payload == {
        "status": "ok",
        "service": "rag-staging",
        "version": "9.9.9",
        "environment": "staging",
    }


def test_health_follows_a_custom_api_prefix(
    client_factory: ClientFactory, settings_factory: SettingsFactory
) -> None:
    """The versioned path tracks ``API_PREFIX``; the unversioned one never moves."""
    api_client = client_factory(settings_factory(api_prefix="v2"))

    assert api_client.get("/v2/health").status_code == 200
    assert api_client.get("/health").status_code == 200
    assert api_client.get("/v1/health").status_code == 404


@pytest.mark.parametrize("path", HEALTH_PATHS)
def test_health_never_leaks_the_api_key(
    client_factory: ClientFactory, settings_factory: SettingsFactory, path: str
) -> None:
    """A credential in settings must not reach the response body or headers."""
    secret = "sk-test-should-never-be-serialised"  # noqa: S105 - fake value for the assertion

    response = client_factory(settings_factory(openai_api_key=secret)).get(path)

    assert secret not in response.text
    assert secret not in str(response.headers)


def test_health_is_documented_with_distinct_operation_ids(client: TestClient) -> None:
    """Both probes appear in the schema with distinct operation ids.

    Duplicate operation ids are what a client generator chokes on, and they are
    the easy mistake when the same payload is served on two paths.
    """
    schema = client.get("/openapi.json").json()

    operation_ids = [
        schema["paths"][path]["get"]["operationId"]
        for path in HEALTH_PATHS
        if path in schema["paths"]
    ]
    assert sorted(operation_ids) == ["health", "health_unversioned"]
