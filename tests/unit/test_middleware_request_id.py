"""Correlation id propagation and the access log line.

The correlation id is the thread that will make a fanned-out RAG request
debuggable in M4+, so its contract is pinned here rather than left implicit:
honour a sane inbound id, replace a hostile one, always emit exactly one
access log event carrying it.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from production_rag.api import middleware
from production_rag.api.middleware import (
    REQUEST_ID_HEADER,
    RESPONSE_TIME_HEADER,
    resolve_request_id,
)


class _RecordingLogger:
    """Minimal structlog stand-in that records events instead of writing them."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def info(self, event: str, **kwargs: Any) -> None:
        self.events.append((event, kwargs))

    def exception(self, event: str, **kwargs: Any) -> None:
        self.events.append((event, kwargs))


def test_inbound_request_id_is_echoed_back(client: TestClient) -> None:
    response = client.get("/health", headers={REQUEST_ID_HEADER: "test-123"})

    assert response.headers[REQUEST_ID_HEADER] == "test-123"


def test_request_id_is_generated_when_absent(client: TestClient) -> None:
    response = client.get("/health")

    generated = response.headers[REQUEST_ID_HEADER]
    assert generated
    # A UUID4, not an empty string or a placeholder.
    assert UUID(generated).version == 4


def test_generated_request_ids_are_unique_per_request(client: TestClient) -> None:
    first = client.get("/health").headers[REQUEST_ID_HEADER]
    second = client.get("/health").headers[REQUEST_ID_HEADER]

    assert first != second


@pytest.mark.parametrize(
    "hostile",
    [
        pytest.param("", id="empty"),
        pytest.param("a" * 129, id="too-long"),
        pytest.param("id with spaces", id="spaces"),
        pytest.param("id;with,punctuation", id="punctuation"),
        pytest.param("<script>alert(1)</script>", id="markup"),
    ],
)
def test_unusable_inbound_request_id_is_replaced(client: TestClient, hostile: str) -> None:
    """A client-supplied id lands in logs and in a response header.

    Anything outside a conservative character set is dropped in favour of a
    generated id, so an attacker cannot shape either sink — and a malformed
    header never turns into a failed request.
    """
    response = client.get("/health", headers={REQUEST_ID_HEADER: hostile})

    returned = response.headers[REQUEST_ID_HEADER]
    assert response.status_code == 200
    assert returned != hostile
    assert UUID(returned).version == 4


@pytest.mark.parametrize(
    "accepted",
    ["test-123", "trace_id.42", "A1b2:C3d4", "0123456789abcdef", "a" * 128],
)
def test_reasonable_ids_are_accepted_verbatim(accepted: str) -> None:
    """Formats real tracing systems emit must survive untouched."""
    assert resolve_request_id(accepted) == accepted


def test_missing_id_yields_a_fresh_uuid4() -> None:
    assert UUID(resolve_request_id(None)).version == 4


def test_request_id_is_available_to_handlers(client: TestClient) -> None:
    """``request.state.request_id`` is what later milestones will log against."""
    response = client.get("/v1/health", headers={REQUEST_ID_HEADER: "state-check"})

    assert response.status_code == 200
    assert response.headers[REQUEST_ID_HEADER] == "state-check"


def test_response_carries_server_timing(client: TestClient) -> None:
    response = client.get("/health")

    assert float(response.headers[RESPONSE_TIME_HEADER]) >= 0.0


def test_one_access_log_line_per_request(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exactly one structured event, carrying the fields an incident needs."""
    recorder = _RecordingLogger()
    monkeypatch.setattr(middleware, "_log", recorder)

    client.get("/v1/health", headers={REQUEST_ID_HEADER: "log-check"})

    assert len(recorder.events) == 1
    event, fields = recorder.events[0]
    assert event == "request_completed"
    assert fields["request_id"] == "log-check"
    assert fields["method"] == "GET"
    assert fields["path"] == "/v1/health"
    assert fields["status_code"] == 200
    assert fields["duration_ms"] >= 0.0


def test_access_log_records_the_path_of_a_404(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unmatched routes are logged too — a 404 storm is a real signal."""
    recorder = _RecordingLogger()
    monkeypatch.setattr(middleware, "_log", recorder)

    client.get("/v1/does-not-exist")

    event, fields = recorder.events[0]
    assert event == "request_completed"
    assert fields["status_code"] == 404
    assert fields["path"] == "/v1/does-not-exist"
