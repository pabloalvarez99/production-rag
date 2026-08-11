import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from production_rag.config_loader import YamlConfig
from production_rag.evals.costs import (
    CallCounts,
    derive_call_counts,
    estimate_cost,
    require_spending_consent,
)
from production_rag.evals.matrix import CONFIGURATIONS, build_scorecard
from production_rag.evals.provenance import assert_collection_embedder
from production_rag.evals.scorecard import CONFIG_NAMES, METRIC_NAMES, validate_scorecard
from production_rag.evals.source_hit import GoldenCase
from production_rag.evals.tier1_retrieval import evaluate_tier1
from production_rag.evals.tier2_answer import evaluate_tier2
from production_rag.generation.llm import FakeLLM, LLMResponse
from production_rag.generation.prompts import ChatMessage
from production_rag.ingest.models import Chunk, Document
from production_rag.retrieval.embeddings import FakeEmbeddingProvider
from production_rag.retrieval.hybrid import Retriever
from production_rag.retrieval.rerank import build_reranker
from production_rag.retrieval.sparse import Bm25Encoder
from production_rag.retrieval.store import CollectionMismatchError, InMemoryVectorStore


def test_emitted_scorecard_validates_exact_contract(tmp_path: Path) -> None:
    cases = [
        GoldenCase(
            id=f"case-{index}",
            question=f"question {index}",
            expected_source_paths=("source.md",),
            category=f"slice_{index // 10}",
            answerable=True,
        )
        for index in range(60)
    ]
    zero_metrics = dict.fromkeys(METRIC_NAMES, 0.0)
    checkpoint = {
        "bootstrap_seed": 20_260_811,
        "bootstrap_resamples": 1_000,
        "embedder": "fake",
        "llm": "fake",
        "judge": "none",
        "configs": {
            name: {"metrics": dict(zero_metrics), "hit_vector": [False] * 60}
            for name in CONFIG_NAMES
        },
    }
    payload = build_scorecard(
        checkpoint=checkpoint,
        cases=cases,
        commit="abc123",
        documents=3_067,
        chunks=9_000,
        golden_path="data/eval/golden-corpus.jsonl",
        corpus_path="data/corpus",
        cost_usd=0.0,
        cost_estimate_usd=0.0,
        cost_expected_usd=0.0,
        billed=False,
    )
    destination = tmp_path / "scorecard.json"
    destination.write_text(json.dumps(payload), encoding="utf-8")
    emitted = json.loads(destination.read_text(encoding="utf-8"))
    validate_scorecard(emitted)
    assert emitted["provenance"]["billed"] is False
    assert emitted["provenance"]["bootstrap_seed"] == 20_260_811
    assert all(not item["reportable"] for item in emitted["comparisons"])


def test_spending_gate_requires_explicit_consent() -> None:
    estimate = estimate_cost(
        calls=CallCounts(100, 60, 60, 0),
        embedding_tokens=1_000,
        query_tokens=300,
        max_context_tokens=6_000,
        max_output_tokens=800,
        billed=True,
    )
    assert estimate.estimated_usd > 0.0
    with pytest.raises(ValueError, match="--yes-spend"):
        require_spending_consent(billed=True, yes_spend=False)
    require_spending_consent(billed=True, yes_spend=True)
    require_spending_consent(billed=False, yes_spend=False)


def test_call_counts_come_from_modes_and_item_count() -> None:
    counts = derive_call_counts(
        items=7,
        configurations={
            "sparse": ("sparse", "off"),
            "dense": ("dense", "off"),
            "hybrid": ("hybrid", "off"),
            "hybrid_rerank": ("hybrid", "fake"),
        },
        chunks_to_embed=11,
        ingest=True,
        llm_provider="fake",
        judge_provider="none",
    )
    assert counts == CallCounts(11, 42, 28, 0)


def test_estimator_matches_instrumented_fake_matrix() -> None:
    class CountingEmbedder(FakeEmbeddingProvider):
        query_calls = 0

        def embed_query(self, text: str) -> list[float]:
            self.query_calls += 1
            return super().embed_query(text)

    class CountingLLM(FakeLLM):
        calls = 0

        def complete(self, messages: Sequence[ChatMessage]) -> LLMResponse:
            self.calls += 1
            return super().complete(messages)

    embedder = CountingEmbedder()
    llm = CountingLLM()
    store = InMemoryVectorStore(collection="counted")
    store.ensure_collection(vector_size=embedder.dimensions, with_sparse=True)
    document = Document(source_path="source.md", text="hybrid retrieval evidence", title="x")
    chunk = Chunk.build(
        document=document, chunk_index=0, text=document.text, embed_text=document.text
    )
    encoder = Bm25Encoder()
    encoder.fit([chunk.embed_text])
    store.upsert_chunks(
        [chunk],
        embedder.embed_documents([chunk.embed_text]),
        ingest_run_id="counted",
        embedded_model=embedder.model,
        sparse_vectors=encoder.encode_documents([chunk.embed_text]),
    )
    case = GoldenCase(
        id="one", question="hybrid retrieval evidence", expected_source_paths=("source.md",)
    )
    config = YamlConfig()
    for mode, rerank_kind in CONFIGURATIONS.values():
        reranker = build_reranker(rerank_kind)
        retriever = Retriever.from_config(
            store=store, embedder=embedder, config=config, reranker=reranker
        )
        evaluate_tier1(retriever=retriever, cases=[case], k=5, mode=mode)
        evaluate_tier2(
            retriever=retriever,
            llm=llm,
            cases=[case],
            mode=mode,
            k=5,
            reranker=reranker,
            config=config,
        )
    predicted = derive_call_counts(
        items=1,
        configurations=CONFIGURATIONS,
        chunks_to_embed=1,
        ingest=True,
        llm_provider="fake",
        judge_provider="none",
    )
    assert predicted == CallCounts(1, embedder.query_calls, llm.calls, 0)


@pytest.mark.parametrize(
    ("billed", "cost_usd"), [(False, 1.0), (True, 0.0)]
)
def test_actual_cost_invariant_is_fail_closed(billed: bool, cost_usd: float) -> None:
    fixture = json.loads(Path("tests/fixtures/scorecard_sample.json").read_text(encoding="utf-8"))
    fixture["provenance"]["billed"] = billed
    fixture["provenance"]["cost_usd"] = cost_usd
    with pytest.raises(ValueError, match="actual cost_usd"):
        validate_scorecard(fixture)


class _Collection:
    def __init__(self) -> None:
        self.collection = "test"
        self._payloads = {"one": {"embedded_model": "fake-deterministic-v1"}}


def test_cross_space_query_is_rejected() -> None:
    with pytest.raises(CollectionMismatchError, match="queries use"):
        assert_collection_embedder(_Collection(), expected_model="text-embedding-3-small")  # type: ignore[arg-type]


def test_matching_collection_embedder_is_accepted() -> None:
    assert_collection_embedder(_Collection(), expected_model="fake-deterministic-v1")  # type: ignore[arg-type]
