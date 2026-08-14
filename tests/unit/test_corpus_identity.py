"""Collection identity, cache isolation, wrong-collection typing."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from production_rag.api.routes import query as query_route
from production_rag.api.schemas import CitationOut, QueryRequest, QueryResponse
from production_rag.config import Settings
from production_rag.corpus_identity import build_corpus_identity, hash_corpus_root
from production_rag.main import create_app
from production_rag.query_cache import (
    CacheKey,
    QueryResultCache,
    reset_query_cache,
    retrieval_fingerprint,
)

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "data" / "raw" / "sample"
ALT = ROOT / "data" / "raw" / "alt"

ClientFactory = Callable[[Settings], TestClient]
SettingsFactory = Callable[..., Settings]


@pytest.fixture(autouse=True)
def _clean_cache() -> Iterator[None]:
    reset_query_cache()
    yield
    reset_query_cache()


def test_two_corpora_have_different_hashes() -> None:
    sample_hash, sample_n = hash_corpus_root(SAMPLE)
    alt_hash, alt_n = hash_corpus_root(ALT)
    assert sample_n >= 1
    assert alt_n >= 1
    assert sample_hash != alt_hash


def test_build_identity_fields() -> None:
    identity = build_corpus_identity(
        corpus_root=SAMPLE, embedder_id="fake", collection="production_rag"
    )
    assert identity.embedder_id == "fake"
    assert identity.chunker_version
    assert identity.doc_count >= 1
    assert len(identity.corpus_hash) == 64
    assert identity.collection == "production_rag"


def test_ready_exposes_identity(client: TestClient) -> None:
    payload = client.get("/v1/ready").json()
    assert payload["status"] == "ready"
    assert payload["collection"]
    assert "embedder_id" in payload
    assert payload["corpus_hash"] is None or len(payload["corpus_hash"]) == 64
    assert "identity" in payload["checks"]


def test_cache_no_cross_hit_across_corpus_identity() -> None:
    cache = QueryResultCache(max_entries=8)
    base = {
        "collection": "prag_demo",
        "query": "What is hybrid search?",
        "filters": "",
        "embedder_id": "fake",
        "llm_id": "fake",
        "retrieval": retrieval_fingerprint(mode="hybrid", top_k=12),
    }
    left = CacheKey(**base, corpus_identity="hash-a|chunker|9|fake")
    right = CacheKey(**base, corpus_identity="hash-b|chunker|2|fake")
    response = QueryResponse(
        answer="Hybrid fuses ranks [1].",
        citations=[
            CitationOut(
                marker=1,
                chunk_id="c1",
                source_path="sample/01-hybrid-search.md",
                text="Hybrid fuses ranks.",
                rank=1,
            )
        ],
        refused=False,
        refusal_reason=None,
    )
    cache.put(left, response)
    assert cache.get(left)[1] == "hit"
    assert cache.get(right)[1] == "miss"


def test_wrong_collection_is_typed_404(
    client_factory: ClientFactory, settings_factory: SettingsFactory
) -> None:
    client = client_factory(settings_factory())
    response = client.post(
        "/v1/query",
        json={
            "question": "Why hybrid?",
            "collection": "totally-other-collection",
            "llm": "fake",
            "embedder": "fake",
        },
    )
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert detail["error_type"] == "wrong_collection"
    assert detail["refused"] is False


def test_provider_failure_is_not_refusal(settings_factory: SettingsFactory) -> None:
    from production_rag.generation.llm import LLMError

    def boom(
        payload: QueryRequest,
        *,
        settings: Settings,
        request_id: str,
        embedder_kind: str | None = None,
        on_delta: object = None,
    ) -> QueryResponse:
        raise LLMError("provider exploded")

    app = create_app(settings_factory())
    app.dependency_overrides[query_route.get_query_executor] = lambda: boom
    client = TestClient(app)
    response = client.post(
        "/v1/query",
        json={"question": "anything", "llm": "fake", "embedder": "fake"},
    )
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["error_type"] == "provider_error"
    assert detail["refused"] is False


def test_store_failure_is_not_refusal(settings_factory: SettingsFactory) -> None:
    from production_rag.retrieval.store import VectorStoreError

    def boom(
        payload: QueryRequest,
        *,
        settings: Settings,
        request_id: str,
        embedder_kind: str | None = None,
        on_delta: object = None,
    ) -> QueryResponse:
        raise VectorStoreError("qdrant down")

    app = create_app(settings_factory())
    app.dependency_overrides[query_route.get_query_executor] = lambda: boom
    client = TestClient(app)
    response = client.post(
        "/v1/query",
        json={"question": "anything", "llm": "fake", "embedder": "fake"},
    )
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["error_type"] == "store_unavailable"
    assert detail["refused"] is False
