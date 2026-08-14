"""Record free-path HTTP transcripts for grounded / refuse / filter / stream.

Uses TestClient + a deterministic fake executor so CI does not need Compose.
When Compose + Playwright are available, ``scripts/capture_ui.py`` still owns
PNG stills (with CACHE_ENABLED=false).

Output: docs/assets/transcripts/*.json
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from production_rag.api.routes import query as query_route
from production_rag.api.schemas import CitationOut, QueryRequest, QueryResponse
from production_rag.config import Settings, get_settings
from production_rag.main import create_app

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets" / "transcripts"


def _executor_for(kind: str) -> object:
    """Return a deterministic query executor for *kind*."""

    def _run(
        payload: QueryRequest,
        *,
        settings: Settings,
        request_id: str,
        embedder_kind: str | None = None,
        on_delta: object = None,
    ) -> QueryResponse:
        if kind == "refuse":
            return QueryResponse(
                answer="I cannot answer from the retrieved evidence.",
                citations=[],
                refused=True,
                refusal_reason="no_evidence",
            )
        if kind == "error":
            from production_rag.retrieval.store import VectorStoreError

            raise VectorStoreError("simulated qdrant down")
        if on_delta is not None and callable(on_delta):
            on_delta("Hybrid ")
            on_delta("search ")
            on_delta("uses RRF [1].")
        filters_note = ""
        if payload.filters:
            filters_note = f" filtered={payload.filters}"
        return QueryResponse(
            answer=f"Hybrid search uses reciprocal rank fusion [1].{filters_note}",
            citations=[
                CitationOut(
                    marker=1,
                    chunk_id="c1",
                    source_path="qdrant/search/filtering.md"
                    if payload.filters
                    else "sample/01-hybrid-search.md",
                    text="Reciprocal rank fusion merges dense and sparse ranks.",
                    rank=1,
                    title="Filtering" if payload.filters else "Hybrid search",
                )
            ],
            refused=False,
            refusal_reason=None,
        )

    return _run


def _client(kind: str) -> TestClient:
    get_settings.cache_clear()
    settings = Settings(cache_enabled=False)
    app = create_app(settings)
    app.dependency_overrides[query_route.get_query_executor] = lambda: _executor_for(kind)
    app.dependency_overrides[query_route.get_streaming_query_executor] = (
        lambda: _executor_for(kind)
    )
    return TestClient(app)


def main() -> int:
    """Write grounded/refuse/filter/stream/error transcripts under docs/assets."""
    OUT.mkdir(parents=True, exist_ok=True)
    transcripts: dict[str, object] = {}

    grounded = _client("grounded").post(
        "/v1/query",
        json={
            "question": "Why does hybrid search use reciprocal rank fusion?",
            "llm": "fake",
            "embedder": "fake",
        },
    )
    transcripts["grounded"] = {
        "method": "POST",
        "path": "/v1/query",
        "status": grounded.status_code,
        "body": grounded.json(),
        "cache_enabled": False,
    }

    refuse = _client("refuse").post(
        "/v1/query",
        json={
            "question": "Who won the Antarctic underwater chess championship?",
            "llm": "fake",
            "embedder": "fake",
        },
    )
    transcripts["refuse"] = {
        "method": "POST",
        "path": "/v1/query",
        "status": refuse.status_code,
        "body": refuse.json(),
    }

    filtered = _client("grounded").post(
        "/v1/query",
        json={
            "question": "How does filtering work in Qdrant?",
            "llm": "fake",
            "embedder": "fake",
            "filters": {"title": "Filtering"},
        },
    )
    transcripts["title_filtering"] = {
        "method": "POST",
        "path": "/v1/query",
        "status": filtered.status_code,
        "body": filtered.json(),
        "filters": {"title": "Filtering"},
        "note": "DEMO-DAY chip; source=sample is the wrong beat on sample corpus",
    }

    stream = _client("grounded").post(
        "/v1/query/stream",
        json={
            "question": "Why does hybrid search use reciprocal rank fusion?",
            "llm": "fake",
            "embedder": "fake",
        },
    )
    transcripts["stream"] = {
        "method": "POST",
        "path": "/v1/query/stream",
        "status": stream.status_code,
        "content_type": stream.headers.get("content-type"),
        "sse_preview": stream.text[:800],
        "cache_enabled": False,
    }

    error = _client("error").post(
        "/v1/query",
        json={"question": "anything", "llm": "fake", "embedder": "fake"},
    )
    transcripts["qdrant_down"] = {
        "method": "POST",
        "path": "/v1/query",
        "status": error.status_code,
        "body": error.json(),
        "cache_enabled": False,
        "note": "store failure is not a refusal",
    }

    for name, payload in transcripts.items():
        path = OUT / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
