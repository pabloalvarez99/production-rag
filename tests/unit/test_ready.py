"""Readiness endpoint behaviour.

The point of these tests is that readiness stays *offline*: they assert real
behaviour with no Qdrant anywhere, which is the property that keeps the suite
runnable in CI without a service container.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient

from production_rag.config import Settings

ClientFactory = Callable[[Settings], TestClient]
SettingsFactory = Callable[..., Settings]


def test_ready_returns_200(client: TestClient) -> None:
    response = client.get("/v1/ready")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")


def test_ready_payload_with_defaults(client: TestClient) -> None:
    """The default QDRANT_URL counts as configured, without contacting it."""
    payload = client.get("/v1/ready").json()

    assert payload["status"] == "ready"
    assert payload["qdrant_configured"] is True
    assert payload["checks"]["settings"] == "ok"
    assert "identity" in payload["checks"]
    assert payload["collection"]
    # Identity is offline: either computed from CORPUS_ROOT or nulls when absent.
    assert "embedder_id" in payload
    assert "chunker_version" in payload
    assert "doc_count" in payload
    assert "corpus_hash" in payload


@pytest.mark.parametrize(
    "qdrant_url",
    [
        pytest.param("", id="empty"),
        pytest.param("   ", id="whitespace"),
        pytest.param("localhost:6333", id="missing-scheme"),
        pytest.param("tcp://qdrant:6334", id="unsupported-scheme"),
    ],
)
def test_ready_reports_unconfigured_qdrant(
    client_factory: ClientFactory, settings_factory: SettingsFactory, qdrant_url: str
) -> None:
    """A missing or unusable endpoint is reported, not guessed at."""
    payload = client_factory(settings_factory(qdrant_url=qdrant_url)).get("/v1/ready").json()

    assert payload["qdrant_configured"] is False
    # Still 200 at M0: configuration parsed, so the process can serve traffic.
    # M1 turns a failed dependency check into a 503 (see the route docstring).
    assert payload["status"] == "ready"


def test_ready_accepts_an_https_endpoint(
    client_factory: ClientFactory, settings_factory: SettingsFactory
) -> None:
    custom = settings_factory(qdrant_url="https://qdrant.internal:6333")

    assert client_factory(custom).get("/v1/ready").json()["qdrant_configured"] is True


def test_ready_is_versioned_only(client: TestClient) -> None:
    """Readiness is an API concern, so it lives under the version prefix only.

    Infrastructure probes use ``/health``; if this 404 ever becomes a 200 it
    means someone mounted the readiness router at the root, where a strict
    orchestrator would start restarting the process over a downstream outage.
    """
    assert client.get("/ready").status_code == 404


def test_ready_does_not_leak_the_api_key(
    client_factory: ClientFactory, settings_factory: SettingsFactory
) -> None:
    secret = "sk-test-readiness-must-not-echo"  # noqa: S105 - fake value for the assertion

    response = client_factory(settings_factory(openai_api_key=secret)).get("/v1/ready")

    assert secret not in response.text
