from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from production_rag.api.routes import query as query_route
from production_rag.api.schemas import CitationOut, QueryDebug, QueryRequest, QueryResponse
from production_rag.config import Settings
from production_rag.retrieval.hybrid import Retriever

ClientFactory = Callable[[Settings], TestClient]
SettingsFactory = Callable[..., Settings]


@dataclass
class FakeQueryExecutor:
    response: QueryResponse
    calls: list[tuple[QueryRequest, Settings, str]] = field(default_factory=list)

    def __call__(
        self,
        payload: QueryRequest,
        *,
        settings: Settings,
        request_id: str,
    ) -> QueryResponse:
        self.calls.append((payload, settings, request_id))
        return self.response


def _answer(*, debug: QueryDebug | None = None) -> QueryResponse:
    response = QueryResponse(
        answer="RRF fuses rank positions instead of incomparable scores [1].",
        citations=[
            CitationOut(
                marker=1,
                chunk_id="chunk-rrf",
                source_path="sample/01-hybrid-search.md",
                text="RRF combines ranked lists without calibrating score scales.",
                rank=1,
                title="Hybrid search",
                heading_path="Reciprocal rank fusion",
            )
        ],
        refused=False,
        refusal_reason=None,
    )
    if debug is not None:
        return response.model_copy(update={"debug": debug})
    return response


def _override(client: TestClient, executor: FakeQueryExecutor) -> None:
    app = cast(FastAPI, client.app)
    app.dependency_overrides[query_route.get_query_executor] = lambda: executor


def test_query_returns_grounded_answer_and_request_id(client: TestClient) -> None:
    executor = FakeQueryExecutor(
        _answer(
            debug=QueryDebug(
                timings_ms={"retrieve": 1.25, "generate": 2.5},
                invalid_markers=[],
            )
        )
    )
    _override(client, executor)

    response = client.post(
        "/v1/query",
        headers={"X-Request-ID": "test-query-123"},
        json={
            "question": "  Why use RRF?  ",
            "mode": "hybrid",
            "rerank": "fake",
            "llm": "fake",
            "debug": True,
        },
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "test-query-123"
    assert response.json() == {
        "answer": "RRF fuses rank positions instead of incomparable scores [1].",
        "citations": [
            {
                "marker": 1,
                "chunk_id": "chunk-rrf",
                "source_path": "sample/01-hybrid-search.md",
                "text": "RRF combines ranked lists without calibrating score scales.",
                "rank": 1,
                "title": "Hybrid search",
                "heading_path": "Reciprocal rank fusion",
            }
        ],
        "refused": False,
        "refusal_reason": None,
        "debug": {
            "timings_ms": {"retrieve": 1.25, "generate": 2.5},
            "invalid_markers": [],
        },
    }
    payload, _, request_id = executor.calls[0]
    assert payload.question == "Why use RRF?"
    assert payload.mode == "hybrid"
    assert payload.rerank == "fake"
    assert payload.llm == "fake"
    assert payload.debug is True
    assert request_id == "test-query-123"


def test_query_returns_explicit_refusal(client: TestClient) -> None:
    executor = FakeQueryExecutor(
        QueryResponse(
            answer="I could not find support for that in the indexed documents.",
            citations=[],
            refused=True,
            refusal_reason="no supporting evidence",
        )
    )
    _override(client, executor)

    response = client.post("/v1/query", json={"question": "What is not in the corpus?"})

    assert response.status_code == 200
    assert response.json()["citations"] == []
    assert response.json()["refused"] is True
    assert response.json()["refusal_reason"] == "no supporting evidence"


def test_query_defaults_are_forwarded_without_network(client: TestClient) -> None:
    executor = FakeQueryExecutor(_answer())
    _override(client, executor)

    response = client.post("/v1/query", json={"question": "Why use RRF?"})

    assert response.status_code == 200
    payload = executor.calls[0][0]
    assert payload.mode is None
    assert payload.rerank is None
    assert payload.llm == "fake"
    assert payload.debug is False


@pytest.mark.parametrize("debug_requested", [False, True])
def test_default_executor_composes_a1_run_query_without_network(
    debug_requested: bool,
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = SimpleNamespace(
        ingest=SimpleNamespace(embedding=SimpleNamespace(provider="openai")),
        qdrant=SimpleNamespace(collection="yaml-collection"),
        rerank=SimpleNamespace(api_key_env="COHERE_API_KEY"),
        generation=SimpleNamespace(api_key_env="OPENAI_API_KEY"),
    )
    embedder = object()
    store = object()
    reranker = object()
    llm = object()
    retriever = object()
    calls: dict[str, Any] = {}

    monkeypatch.setattr(query_route, "load_yaml_config", lambda _: config)
    monkeypatch.setattr(query_route, "resolve_embedder", lambda *args, **kwargs: embedder)
    monkeypatch.setattr(
        query_route,
        "resolve_searchable_store",
        lambda **kwargs: store,
    )
    monkeypatch.setattr(query_route, "build_reranker", lambda *args, **kwargs: reranker)
    monkeypatch.setattr(query_route, "_build_llm", lambda *args, **kwargs: llm)
    monkeypatch.setattr(
        Retriever,
        "from_config",
        classmethod(lambda cls, **kwargs: retriever),
    )

    def fake_run_query(question: str, *, request_id: str, **kwargs: Any) -> SimpleNamespace:
        calls["question"] = question
        calls["request_id"] = request_id
        calls.update(kwargs)
        answer = _answer()
        return SimpleNamespace(
            answer=answer.answer,
            citations=answer.citations,
            refused=answer.refused,
            refusal_reason=answer.refusal_reason,
            timings_ms={"retrieve": 1.25, "generate": 2.5},
            invalid_markers=(7,),
            api_key="sensitive-provider-value",
            system_prompt="internal system instructions",
        )

    monkeypatch.setattr(query_route, "_run_query", fake_run_query)

    response = client.post(
        "/v1/query",
        json={
            "question": "Why use RRF?",
            "mode": "dense",
            "rerank": "fake",
            "llm": "fake",
            "debug": debug_requested,
        },
    )

    assert response.status_code == 200
    if debug_requested:
        assert response.json()["debug"] == {
            "timings_ms": {"retrieve": 1.25, "generate": 2.5},
            "invalid_markers": [7],
        }
    else:
        assert "debug" not in response.json()
        assert "timings_ms" not in response.text
    assert "api_key" not in response.text
    assert "sensitive-provider-value" not in response.text
    assert "system_prompt" not in response.text
    assert "internal system instructions" not in response.text
    assert calls == {
        "question": "Why use RRF?",
        "request_id": response.headers["X-Request-ID"],
        "retriever": retriever,
        "llm": llm,
        "config": config,
        "mode": "dense",
        "reranker": reranker,
    }


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({"question": "   "}, id="blank-question"),
        pytest.param({"question": "ok", "mode": "magic"}, id="unknown-mode"),
        pytest.param({"question": "ok", "rerank": "unknown"}, id="unknown-reranker"),
        pytest.param({"question": "ok", "llm": "live"}, id="unknown-llm"),
        pytest.param({"question": "ok", "api_key": "never"}, id="extra-field"),
    ],
)
def test_query_rejects_invalid_requests(client: TestClient, body: dict[str, object]) -> None:
    executor = FakeQueryExecutor(_answer())
    _override(client, executor)

    response = client.post("/v1/query", json=body)

    assert response.status_code == 422
    assert executor.calls == []


def test_query_returns_503_when_a1_pipeline_is_missing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(query_route, "_run_query", None)

    response = client.post("/v1/query", json={"question": "Why use RRF?"})

    assert response.status_code == 503
    assert response.json() == {"detail": "query pipeline not installed"}
    assert "X-Request-ID" in response.headers


def test_query_follows_custom_api_prefix(
    client_factory: ClientFactory, settings_factory: SettingsFactory
) -> None:
    client = client_factory(settings_factory(api_prefix="v2"))
    executor = FakeQueryExecutor(_answer())
    _override(client, executor)

    assert client.post("/v2/query", json={"question": "Why use RRF?"}).status_code == 200
    assert client.post("/v1/query", json={"question": "Why use RRF?"}).status_code == 404


def test_query_is_described_by_openapi(client: TestClient) -> None:
    operation = client.get("/openapi.json").json()["paths"]["/v1/query"]["post"]

    assert operation["operationId"] == "query"
    assert operation["requestBody"]["required"] is True
    assert "200" in operation["responses"]
    assert "422" in operation["responses"]
    assert "503" in operation["responses"]
