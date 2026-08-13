import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from production_rag.config_loader import YamlConfig
from production_rag.evals.costs import (
    CallCounts,
    CostAccountingError,
    UsageLedger,
    derive_call_counts,
    estimate_cost,
    exact_token_count,
    format_preflight,
    format_spend_summary,
    require_spending_consent,
)
from production_rag.evals.matrix import CONFIGURATIONS, build_scorecard
from production_rag.evals.provenance import assert_collection_embedder
from production_rag.evals.scorecard import CONFIG_NAMES, METRIC_NAMES, validate_scorecard
from production_rag.evals.source_hit import GoldenCase
from production_rag.evals.tier1_retrieval import evaluate_tier1
from production_rag.evals.tier2_answer import evaluate_tier2
from production_rag.generation.llm import FakeLLM
from production_rag.ingest.models import Chunk, Document
from production_rag.retrieval.embeddings import FakeEmbeddingProvider
from production_rag.retrieval.hybrid import Retriever
from production_rag.retrieval.rerank import build_reranker
from production_rag.retrieval.sparse import Bm25Encoder
from production_rag.retrieval.store import CollectionMismatchError, InMemoryVectorStore


class _StubEncoding:
    """The slice of ``tiktoken.Encoding`` the preflight uses.

    Faithful on the one behaviour under test: the real encoder refuses text that
    looks like a control token unless the caller allows it through, and the
    preflight has to allow it. Stubbed rather than imported because
    ``tiktoken.encoding_for_model`` downloads its BPE table from an OpenAI host
    on a cold cache, and this suite runs with no network (see
    ``tests/conftest.py``).
    """

    def __init__(self) -> None:
        self.encoded: list[str] = []

    def encode(self, text: str, *, disallowed_special: tuple[str, ...] | str = "all") -> list[int]:
        if disallowed_special == "all" and "<|endoftext|>" in text:
            raise ValueError("Encountered text corresponding to disallowed special token")
        self.encoded.append(text)
        return list(range(len(text.split())))


class _StubTiktoken(ModuleType):
    def __init__(self, encoding: _StubEncoding) -> None:
        super().__init__("tiktoken")
        self._encoding = encoding

    def encoding_for_model(self, model: str) -> _StubEncoding:
        return self._encoding


@pytest.fixture
def stub_tiktoken(monkeypatch: pytest.MonkeyPatch) -> _StubEncoding:
    encoding = _StubEncoding()
    monkeypatch.setitem(sys.modules, "tiktoken", _StubTiktoken(encoding))
    return encoding


def test_control_token_text_is_counted_as_content(stub_tiktoken: _StubEncoding) -> None:
    text = "literal <|endoftext|> text"
    assert exact_token_count([text], model="text-embedding-3-small") == 3
    assert stub_tiktoken.encoded == [text]


def test_missing_tiktoken_fails_the_preflight_instead_of_approximating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    with pytest.raises(RuntimeError, match="requires the tiktoken package"):
        exact_token_count(["text"], model="text-embedding-3-small")


def test_fake_providers_report_zero_labelled_as_unbilled() -> None:
    estimate = estimate_cost(
        calls=CallCounts(100, 60, 60, 0),
        embedding_tokens=1_000,
        query_tokens=300,
        max_context_tokens=6_000,
        max_output_tokens=800,
        billed=False,
    )
    assert (estimate.billed, estimate.expected_usd, estimate.upper_bound_usd) == (False, 0.0, 0.0)
    preflight = format_preflight(estimate)
    assert "$0.00" in preflight[0]
    assert "unbilled" in preflight[0]
    # A hosted rate table over a run that calls no hosted provider is the thing
    # that makes a $0.00 read as a priced result.
    assert not any("text-embedding-3-small" in line for line in preflight)
    assert "generations=60" in preflight[1]
    summary = format_spend_summary(estimate=estimate, actual_usd=0.0)
    assert "$0.00" in summary
    assert "unbilled" in summary


def test_unbilled_summary_refuses_to_call_a_non_zero_total_free() -> None:
    estimate = estimate_cost(
        calls=CallCounts(0, 0, 0, 0),
        embedding_tokens=0,
        query_tokens=0,
        max_context_tokens=6_000,
        max_output_tokens=800,
        billed=False,
    )
    with pytest.raises(CostAccountingError, match="unbilled run measured"):
        format_spend_summary(estimate=estimate, actual_usd=0.01)


def test_billed_preflight_publishes_its_rate_assumptions() -> None:
    estimate = estimate_cost(
        calls=CallCounts(100, 60, 60, 0),
        embedding_tokens=1_000,
        query_tokens=300,
        max_context_tokens=6_000,
        max_output_tokens=800,
        billed=True,
    )
    preflight = format_preflight(estimate)
    assert estimate.billed is True
    assert any("text-embedding-3-small: $0.0200" in line for line in preflight)
    assert f"upper bound=${estimate.upper_bound_usd:.4f}" in preflight[-1]
    assert "unbilled" not in format_spend_summary(estimate=estimate, actual_usd=0.004)


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
    ledger = UsageLedger(billed=False)
    embedder = FakeEmbeddingProvider(usage_recorder=ledger.record_embedding)
    llm = FakeLLM(usage_recorder=ledger.record_generation)
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
    assert predicted == ledger.calls
    assert ledger.cost_usd == 0.0


def test_paid_response_without_usage_fails_accounting() -> None:
    ledger = UsageLedger(billed=True)
    with pytest.raises(CostAccountingError, match="embedding response"):
        ledger.record_embedding(kind="query", items=1, prompt_tokens=None)
    with pytest.raises(CostAccountingError, match="generation response"):
        ledger.record_generation(prompt_tokens=10, completion_tokens=None)


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
