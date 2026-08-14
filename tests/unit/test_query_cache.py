"""Unit tests for the filter-aware in-process query cache.

The load-bearing property is negative: a filtered answer must never serve an
unfiltered query (or the reverse). Hit and miss are the positive checks that
the map actually works when the key is complete.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from production_rag.api.routes import query as query_route
from production_rag.api.schemas import CitationOut, QueryRequest, QueryResponse
from production_rag.config import Settings
from production_rag.main import create_app
from production_rag.query_cache import (
    CacheKey,
    QueryResultCache,
    canonical_filters,
    reset_query_cache,
    retrieval_fingerprint,
)


@pytest.fixture(autouse=True)
def _clean_cache() -> Iterator[None]:
    reset_query_cache()
    yield
    reset_query_cache()


def test_canonical_filters_empty_forms_agree() -> None:
    assert canonical_filters(None) == ""
    assert canonical_filters({}) == ""


def test_canonical_filters_order_independent() -> None:
    left = canonical_filters({"title": "Filtering", "source": "sample"})
    right = canonical_filters({"source": "sample", "title": "Filtering"})
    assert left == right


def test_cache_hit_and_miss() -> None:
    cache = QueryResultCache(max_entries=4)
    key = CacheKey(
        collection="prag_demo",
        query="What is hybrid search?",
        filters="",
        embedder_id="fake",
        llm_id="fake",
        retrieval=retrieval_fingerprint(mode="hybrid", top_k=12),
    )
    response = QueryResponse(
        answer="Hybrid search fuses dense and sparse [1].",
        citations=[
            CitationOut(
                marker=1,
                chunk_id="c1",
                source_path="sample/01.md",
                text="Hybrid search fuses dense and sparse.",
                rank=1,
            )
        ],
        refused=False,
        refusal_reason=None,
    )
    assert cache.get(key) == (None, "miss")
    cache.put(key, response)
    hit, status = cache.get(key)
    assert status == "hit"
    assert hit is not None
    assert hit.answer == response.answer
    assert hit is not response


def test_filter_mismatch_is_a_miss() -> None:
    cache = QueryResultCache(max_entries=4)
    unfiltered = CacheKey(
        collection="prag_demo",
        query="How does filtering work?",
        filters="",
        embedder_id="fake",
        llm_id="fake",
        retrieval=retrieval_fingerprint(mode="hybrid", top_k=12),
    )
    filtered = CacheKey(
        collection="prag_demo",
        query="How does filtering work?",
        filters=canonical_filters({"title": "Filtering"}),
        embedder_id="fake",
        llm_id="fake",
        retrieval=retrieval_fingerprint(mode="hybrid", top_k=12),
    )
    cache.put(
        unfiltered,
        QueryResponse(
            answer="Unfiltered answer [1].",
            citations=[],
            refused=False,
            refusal_reason=None,
        ),
    )
    hit, status = cache.get(filtered)
    assert status == "miss"
    assert hit is None


def test_execute_query_reports_cache_status_on_debug(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"n": 0}

    def fake_run(question: str, **kwargs: Any) -> Any:
        calls["n"] += 1

        class _Result:
            answer = f"Answer for {question} [1]."
            citations = ()
            refused = False
            refusal_reason = None
            timings_ms: dict[str, float]
            invalid_markers = ()

            def __init__(self) -> None:
                self.timings_ms = {"retrieve": 1.0, "generate": 2.0}

        return _Result()

    monkeypatch.setattr(query_route, "_run_query", fake_run)
    monkeypatch.setattr(query_route, "_build_llm", lambda *a, **k: object())
    monkeypatch.setattr(query_route, "resolve_embedder", lambda *a, **k: object())
    monkeypatch.setattr(query_route, "resolve_searchable_store", lambda *a, **k: object())
    monkeypatch.setattr(query_route, "build_reranker", lambda *a, **k: None)
    monkeypatch.setattr(
        query_route,
        "Retriever",
        type("R", (), {"from_config": staticmethod(lambda **k: object())}),
    )

    settings = Settings(cache_enabled=True, config_path=None)
    payload = QueryRequest(
        question="Why use RRF?",
        llm="fake",
        embedder="fake",
        debug=True,
    )
    first = query_route.execute_query(payload, settings=settings, request_id="r1")
    second = query_route.execute_query(payload, settings=settings, request_id="r2")
    assert first.debug is not None
    assert first.debug.cache == "miss"
    assert second.debug is not None
    assert second.debug.cache == "hit"
    assert calls["n"] == 1

    silent = query_route.execute_query(
        QueryRequest(question="Why use RRF?", llm="fake", embedder="fake", debug=False),
        settings=settings,
        request_id="r3",
    )
    assert silent.debug is None
    assert calls["n"] == 1


def test_http_filter_mismatch_cannot_reuse_unfiltered_hit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any] | None] = []

    def fake_run(question: str, **kwargs: Any) -> Any:
        calls.append(kwargs.get("filters"))

        class _Result:
            answer: str
            citations = ()
            refused = False
            refusal_reason = None
            timings_ms: dict[str, float]
            invalid_markers = ()

            def __init__(self) -> None:
                self.answer = f"n={len(calls)} filters={kwargs.get('filters')}"
                self.timings_ms = {}

        return _Result()

    monkeypatch.setattr(query_route, "_run_query", fake_run)
    monkeypatch.setattr(query_route, "_build_llm", lambda *a, **k: object())
    monkeypatch.setattr(query_route, "resolve_embedder", lambda *a, **k: object())
    monkeypatch.setattr(query_route, "resolve_searchable_store", lambda *a, **k: object())
    monkeypatch.setattr(query_route, "build_reranker", lambda *a, **k: None)
    monkeypatch.setattr(
        query_route,
        "Retriever",
        type("R", (), {"from_config": staticmethod(lambda **k: object())}),
    )

    app = create_app(Settings(cache_enabled=True, config_path=None))
    client = TestClient(app)
    body = {
        "question": "How does filtering work in Qdrant?",
        "llm": "fake",
        "embedder": "fake",
        "debug": True,
    }
    unfiltered = client.post("/v1/query", json=body)
    assert unfiltered.status_code == 200
    assert unfiltered.json()["debug"]["cache"] == "miss"

    filtered = client.post(
        "/v1/query",
        json={**body, "filters": {"title": "Filtering"}},
    )
    assert filtered.status_code == 200
    assert filtered.json()["debug"]["cache"] == "miss"
    assert filtered.json()["answer"] != unfiltered.json()["answer"]
    assert len(calls) == 2
    assert calls[0] is None
    assert calls[1] == {"title": "Filtering"}
