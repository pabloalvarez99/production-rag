from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from urllib.parse import urlparse

from fastapi import FastAPI
from fastapi.testclient import TestClient

from production_rag.api.routes import query as query_route
from production_rag.api.schemas import CitationOut, QueryDebug, QueryRequest, QueryResponse
from production_rag.config import Settings
from production_rag.retrieval.filters import FILTER_NOT_ALLOWED, FilterError

SettingsFactory = Callable[..., Settings]
_ORIGIN = re.compile(r"https?://[^\"'\s)<]+")


@dataclass
class FakeExecutor:
    """Offline query executor used by the HTML adapter tests."""

    response: QueryResponse | None = None
    error: Exception | None = None
    payload: QueryRequest | None = None

    def __call__(
        self,
        payload: QueryRequest,
        *,
        settings: Settings,
        request_id: str,
    ) -> QueryResponse:
        del settings, request_id
        self.payload = payload
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response


def _override(client: TestClient, executor: FakeExecutor) -> None:
    app = cast(FastAPI, client.app)
    app.dependency_overrides[query_route.get_query_executor] = lambda: executor


def _grounded() -> QueryResponse:
    return QueryResponse(
        answer="RRF combines ranked results [1] without score calibration [2].",
        citations=[
            CitationOut(
                marker=1,
                chunk_id="rrf",
                source_path="sample/hybrid.md",
                text="RRF combines ranked result lists.",
                rank=1,
            ),
            CitationOut(
                marker=2,
                chunk_id="scores",
                source_path="sample/scores.md",
                text="The source scores need not share a scale.",
                rank=2,
            ),
        ],
        refused=False,
        refusal_reason=None,
        debug=QueryDebug(timings_ms={"retrieve": 1.25, "generate": 2.5}),
    )


def test_get_root_returns_html(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Ask the corpus" in response.text
    assert 'hx-post="/ui/query"' in response.text


def test_grounded_answer_maps_markers_to_citations(client: TestClient) -> None:
    executor = FakeExecutor(response=_grounded())
    _override(client, executor)

    response = client.post(
        "/ui/query",
        headers={"X-Request-ID": "ui-grounded"},
        data={"question": "Why use RRF?"},
    )

    assert response.status_code == 200
    assert 'data-outcome="grounded"' in response.text
    assert 'href="#source-1"' in response.text
    assert 'href="#source-2"' in response.text
    assert 'id="source-1"' in response.text
    assert 'id="source-2"' in response.text
    assert "sample/hybrid.md" in response.text
    assert "1.25 ms" in response.text
    assert "ui-grounded" in response.text
    assert executor.payload is not None
    assert executor.payload.debug is True
    assert executor.payload.llm == "fake"
    assert executor.payload.embedder == "fake"


def test_refusal_is_not_rendered_as_error(client: TestClient) -> None:
    _override(
        client,
        FakeExecutor(
            response=QueryResponse(
                answer="I could not find support for that in the indexed documents.",
                refused=True,
                refusal_reason="no supporting evidence",
            )
        ),
    )

    response = client.post("/ui/query", data={"question": "What is on the Moon?"})

    assert response.status_code == 200
    assert 'data-outcome="refused"' in response.text
    assert "Deliberate refusal" in response.text
    assert 'data-outcome="error"' not in response.text


def test_failure_is_not_rendered_as_refusal(client: TestClient) -> None:
    _override(client, FakeExecutor(error=RuntimeError("provider unavailable")))

    response = client.post("/ui/query", data={"question": "Why use RRF?"})

    assert response.status_code == 200
    assert 'data-outcome="error"' in response.text
    assert "Service failure" in response.text
    assert "provider unavailable" not in response.text
    assert 'data-outcome="refused"' not in response.text


def test_missing_collection_renders_recovery_command(client: TestClient) -> None:
    _override(client, FakeExecutor(error=RuntimeError("collection demo not found")))

    response = client.post("/ui/query", data={"question": "Why use RRF?"})

    assert response.status_code == 200
    assert 'data-outcome="empty-store"' in response.text
    assert "--embedder fake" in response.text
    assert "Traceback" not in response.text


def test_form_offers_only_allowlisted_filter_fields(client: TestClient) -> None:
    """The control is built from configuration, not from a hand-written list."""
    page = client.get("/").text

    assert 'name="filter_field"' in page
    assert '<option value="source">' in page
    assert '<option value="title">' in page
    assert '<option value="tags">' in page
    assert '<option value="">No filter</option>' in page
    # Removed from the allowlist in M7: no ingest payload ever carried the key.
    assert '<option value="created_at">' not in page
    assert '<option value="source_path">' not in page


def test_filter_reaches_the_query_executor(client: TestClient) -> None:
    executor = FakeExecutor(response=_grounded())
    _override(client, executor)

    response = client.post(
        "/ui/query",
        data={"question": "Why use RRF?", "filter_field": "source", "filter_value": "sample"},
    )

    assert response.status_code == 200
    assert 'data-outcome="grounded"' in response.text
    assert executor.payload is not None
    assert executor.payload.filters == {"source": "sample"}
    # The answer names the narrowing it was produced under.
    assert 'data-filter="source = sample"' in response.text


def test_unfiltered_submission_is_unchanged(client: TestClient) -> None:
    """An unselected field calls the executor exactly as it was called before."""
    executor = FakeExecutor(response=_grounded())
    _override(client, executor)

    response = client.post(
        "/ui/query",
        data={"question": "Why use RRF?", "filter_field": "", "filter_value": "ignored"},
    )

    assert response.status_code == 200
    assert executor.payload is not None
    assert executor.payload.filters is None
    assert "data-filter=" not in response.text


def test_unknown_filter_field_is_a_typed_422(client: TestClient) -> None:
    executor = FakeExecutor(response=_grounded())
    _override(client, executor)

    response = client.post(
        "/ui/query",
        data={
            "question": "Why use RRF?",
            "filter_field": "source_path",
            "filter_value": "sample/hybrid.md",
        },
    )

    assert response.status_code == 422
    assert 'data-outcome="invalid-filter"' in response.text
    assert "filter_not_allowed" in response.text
    assert "source_path" in response.text
    # Rejected before anything ran, and rendered as a fragment, not a stack trace.
    assert executor.payload is None
    assert "Traceback" not in response.text


def test_allowlisted_field_with_no_value_is_rejected(client: TestClient) -> None:
    """An empty value matches nothing, which reads as a corpus gap. Refuse it."""
    executor = FakeExecutor(response=_grounded())
    _override(client, executor)

    response = client.post(
        "/ui/query",
        data={"question": "Why use RRF?", "filter_field": "tags", "filter_value": "   "},
    )

    assert response.status_code == 422
    assert 'data-outcome="invalid-filter"' in response.text
    assert "filter_invalid_value" in response.text
    assert executor.payload is None


def test_filter_rejected_by_the_pipeline_is_not_a_service_failure(client: TestClient) -> None:
    """The executor is the second gate; its rejection keeps the typed rendering."""
    _override(
        client,
        FakeExecutor(
            error=FilterError(
                "filtering on 'author' is not allowed",
                error_type=FILTER_NOT_ALLOWED,
                field="author",
            )
        ),
    )

    response = client.post(
        "/ui/query",
        data={"question": "Why use RRF?", "filter_field": "source", "filter_value": "sample"},
    )

    assert response.status_code == 422
    assert 'data-outcome="invalid-filter"' in response.text
    assert 'data-outcome="error"' not in response.text


def test_static_assets_are_local_and_served(client: TestClient) -> None:
    page = client.get("/")
    script = client.get("/static/htmx.min.js")
    stylesheet = client.get("/static/app.css")

    assert script.status_code == 200
    assert stylesheet.status_code == 200
    assert 'version:"2.0.4"' in script.text
    assert "/static/htmx.min.js" in page.text
    assert "/static/app.css" in page.text


def test_ui_assets_reference_no_external_origin(client: TestClient) -> None:
    """Keep the clone-and-run UI independent of CDNs and external hosts."""
    rendered = client.get("/").text
    asset_root = Path(__file__).parents[2] / "src" / "production_rag"
    sources = [rendered]
    sources.extend(
        path.read_text(encoding="utf-8")
        for path in (asset_root / "templates").glob("*.html")
    )
    sources.extend(
        path.read_text(encoding="utf-8")
        for path in (asset_root / "static").iterdir()
        if path.is_file()
    )
    origins = {urlparse(match).netloc for source in sources for match in _ORIGIN.findall(source)}

    assert origins <= {"testserver"}
