"""Replay free-path scorecard items through /v1/query and /v1/query/stream.

Week-3 loop: pick item k from the committed golden set (the free-path scorecard
fixture's source of questions), hit the public routes, and record hit/miss plus
refused/grounded. These are contract tests under fake providers — not quality
claims — so they monkeypatch the pipeline the same way the cache tests do.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from production_rag.api.routes import query as query_route
from production_rag.api.schemas import QueryRequest
from production_rag.config import Settings
from production_rag.main import create_app
from production_rag.query_cache import reset_query_cache

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "data" / "eval" / "golden.jsonl"


def _load_golden() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for line in GOLDEN.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def _item(item_id: str) -> dict[str, Any]:
    for item in _load_golden():
        if item["id"] == item_id:
            return item
    raise KeyError(item_id)


@pytest.fixture(autouse=True)
def _clean_cache() -> Iterator[None]:
    reset_query_cache()
    yield
    reset_query_cache()


def _grounded_result(question: str) -> Any:
    class _Result:
        answer = f"Grounded replay for {question[:40]} [1]."
        citations = (
            type(
                "C",
                (),
                {
                    "marker": 1,
                    "chunk_id": "c1",
                    "source_path": "sample/01-hybrid-search.md",
                    "text": "RRF fuses ranks, not raw scores.",
                    "rank": 1,
                    "score": 0.9,
                },
            )(),
        )
        refused = False
        refusal_reason = None
        timings_ms: dict[str, float]
        invalid_markers = ()

        def __init__(self) -> None:
            self.timings_ms = {"retrieve": 1.0, "generate": 2.0}

    return _Result()


def _refusal_result(question: str) -> Any:
    del question

    class _Result:
        answer = "I could not find support for that in the indexed documents."
        citations = ()
        refused = True
        refusal_reason = "no_evidence"
        timings_ms: dict[str, float]
        invalid_markers = ()

        def __init__(self) -> None:
            self.timings_ms = {"retrieve": 1.0}

    return _Result()


def _wire_fake_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    *,
    refused: bool,
) -> dict[str, int]:
    calls = {"n": 0}

    def fake_run(question: str, **kwargs: Any) -> Any:
        del kwargs
        calls["n"] += 1
        return _refusal_result(question) if refused else _grounded_result(question)

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
    return calls


def test_golden_item_k_grounded_keeps_valid_markers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scorecard item q-0002 is answerable: replay must stay grounded with [1]."""
    item = _item("q-0002")
    assert item["answerable"] is True
    _wire_fake_pipeline(monkeypatch, refused=False)

    settings = Settings(cache_enabled=False, config_path=None)
    app = create_app(settings)
    client = TestClient(app)
    body = {
        "question": item["question"],
        "llm": "fake",
        "embedder": "fake",
        "debug": True,
    }

    query = client.post("/v1/query", json=body)
    assert query.status_code == 200
    payload = query.json()
    assert payload["refused"] is False
    assert "[1]" in payload["answer"]
    assert payload["citations"]
    assert all(c["marker"] >= 1 for c in payload["citations"])
    # Cache off by default: debug omits cache status rather than inventing a hit.
    assert "cache" not in (payload.get("debug") or {})

    stream = client.post("/v1/query/stream", json=body)
    assert stream.status_code == 200
    text = stream.text
    assert "event: result" in text
    assert '"refused":false' in text.replace(" ", "")
    assert "[1]" in text


def test_golden_unanswerable_refuses_on_query_and_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scorecard item q-0012 is unanswerable: both routes must refuse."""
    item = _item("q-0012")
    assert item["answerable"] is False
    _wire_fake_pipeline(monkeypatch, refused=True)

    app = create_app(Settings(cache_enabled=False, config_path=None))
    client = TestClient(app)
    body = {
        "question": item["question"],
        "llm": "fake",
        "embedder": "fake",
        "debug": True,
    }

    query = client.post("/v1/query", json=body)
    assert query.status_code == 200
    payload = query.json()
    assert payload["refused"] is True
    assert payload["refusal_reason"]
    assert payload["citations"] == []

    stream = client.post("/v1/query/stream", json=body)
    assert stream.status_code == 200
    text = stream.text
    assert "event: result" in text
    assert "event: error" not in text
    assert '"refused":true' in text.replace(" ", "")


def test_scorecard_replay_cache_hit_on_second_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When CACHE_ENABLED, replaying the same scorecard item is a cache hit."""
    item = _item("q-0002")
    calls = _wire_fake_pipeline(monkeypatch, refused=False)

    settings = Settings(cache_enabled=True, config_path=None)
    payload = QueryRequest(
        question=item["question"],
        llm="fake",
        embedder="fake",
        debug=True,
    )
    first = query_route.execute_query(payload, settings=settings, request_id="replay-1")
    second = query_route.execute_query(payload, settings=settings, request_id="replay-2")

    assert first.debug is not None
    assert first.debug.cache == "miss"
    assert second.debug is not None
    assert second.debug.cache == "hit"
    assert first.refused is False
    assert second.refused is False
    assert "[1]" in first.answer
    assert second.answer == first.answer
    assert calls["n"] == 1


def test_scorecard_replay_records_miss_then_hit_over_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = _item("q-0002")
    calls = _wire_fake_pipeline(monkeypatch, refused=False)
    app = create_app(Settings(cache_enabled=True, config_path=None))
    client = TestClient(app)
    body = {
        "question": item["question"],
        "llm": "fake",
        "embedder": "fake",
        "debug": True,
    }

    first = client.post("/v1/query", json=body).json()
    second = client.post("/v1/query", json=body).json()

    assert first["debug"]["cache"] == "miss"
    assert second["debug"]["cache"] == "hit"
    assert first["refused"] is False
    assert second["answer"] == first["answer"]
    assert calls["n"] == 1
