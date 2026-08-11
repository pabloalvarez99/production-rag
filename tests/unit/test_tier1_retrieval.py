"""Tier 1 retrieval metrics. Offline: in-memory store and the fake embedder."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import structlog

from production_rag.config_loader import RetrievalConfig
from production_rag.evals.source_hit import CaseOutcome, EvalError, GoldenCase
from production_rag.evals.tier1_retrieval import (
    TIER1_METRICS,
    Tier1Report,
    evaluate_tier1,
    score_case,
)
from production_rag.ingest.models import Chunk, Document
from production_rag.retrieval.embeddings import FakeEmbeddingProvider
from production_rag.retrieval.hybrid import MODE_SPARSE, Retriever
from production_rag.retrieval.sparse import Bm25Encoder
from production_rag.retrieval.store import InMemoryVectorStore

CORPUS = {
    "sample/00-intro.md": "Production RAG combines retrieval with generation over a corpus.",
    "sample/01-hybrid-search.md": "Reciprocal rank fusion merges ranked lists by position.",
    "sample/02-reranking.md": "A cross-encoder scores a query and passage together.",
    "sample/03-chunking.md": "The chunker splits markdown on heading boundaries.",
}


@pytest.fixture(autouse=True)
def _reset_structlog() -> Iterator[None]:
    """Undo any global logging configuration between tests."""
    yield
    structlog.reset_defaults()


def _retriever(**overrides: object) -> Retriever:
    embedder = FakeEmbeddingProvider()
    chunks = []
    for index, (path, text) in enumerate(CORPUS.items()):
        document = Document(source_path=path, text=text, title=path, source="sample")
        chunks.append(Chunk.build(document=document, chunk_index=index, text=text, embed_text=text))
    texts = [chunk.embed_text for chunk in chunks]
    encoder = Bm25Encoder()
    encoder.fit(texts)
    store = InMemoryVectorStore(collection="eval_collection")
    store.ensure_collection(vector_size=embedder.dimensions, with_sparse=True)
    store.upsert_chunks(
        chunks,
        embedder.embed_documents(texts),
        ingest_run_id="run-test",
        embedded_model=embedder.model,
        sparse_vectors=encoder.encode_documents(texts),
    )
    return Retriever(
        store=store,
        embedder=embedder,
        config=RetrievalConfig(**overrides),  # type: ignore[arg-type]
    )


def _case(case_id: str, question: str, *paths: str, category: str | None = None) -> GoldenCase:
    return GoldenCase(id=case_id, question=question, expected_source_paths=paths, category=category)


def _outcome(
    *,
    expected: tuple[str, ...],
    retrieved: tuple[str, ...],
    category: str | None = None,
) -> CaseOutcome:
    """Build an outcome the way ``evaluate_source_hit`` would have."""
    hit_rank = next(
        (index for index, path in enumerate(retrieved, start=1) if path in set(expected)),
        None,
    )
    return CaseOutcome(
        id="q",
        question="why",
        hit=hit_rank is not None,
        expected_source_paths=expected,
        retrieved_source_paths=retrieved,
        hit_rank=hit_rank,
        category=category,
        scored=bool(expected),
    )


class TestPerCaseScores:
    def test_a_first_rank_hit_scores_everything_at_one(self) -> None:
        scores = score_case(_outcome(expected=("a.md",), retrieved=("a.md", "b.md")))
        assert (scores.hit, scores.recall, scores.reciprocal_rank, scores.ndcg) == (
            True,
            1.0,
            1.0,
            1.0,
        )

    def test_reciprocal_rank_follows_the_position(self) -> None:
        scores = score_case(_outcome(expected=("c.md",), retrieved=("a.md", "b.md", "c.md")))
        assert scores.reciprocal_rank == pytest.approx(1 / 3)

    def test_a_miss_scores_zero_everywhere(self) -> None:
        scores = score_case(_outcome(expected=("z.md",), retrieved=("a.md", "b.md")))
        assert (scores.hit, scores.recall, scores.reciprocal_rank, scores.ndcg) == (
            False,
            0.0,
            0.0,
            0.0,
        )

    def test_recall_sees_what_hit_cannot(self) -> None:
        # The reason both are reported: a multi-source question that found one of
        # two documents is a hit and is also half an answer.
        scores = score_case(_outcome(expected=("a.md", "b.md"), retrieved=("a.md", "z.md")))
        assert scores.hit is True
        assert scores.recall == 0.5

    def test_repeated_chunks_from_one_document_count_once(self) -> None:
        # Several chunks of one file is one document found. Counting them twice
        # would let a well-chunked file score full recall on a two-file question.
        scores = score_case(_outcome(expected=("a.md", "b.md"), retrieved=("a.md", "a.md")))
        assert scores.recall == 0.5

    def test_ndcg_cannot_reach_one_with_a_source_missing(self) -> None:
        scores = score_case(_outcome(expected=("a.md", "b.md"), retrieved=("a.md", "z.md")))
        assert 0.0 < scores.ndcg < 1.0

    def test_ndcg_is_one_when_both_sources_lead(self) -> None:
        scores = score_case(_outcome(expected=("a.md", "b.md"), retrieved=("a.md", "b.md")))
        assert scores.ndcg == pytest.approx(1.0)

    def test_ndcg_prefers_the_better_ranking(self) -> None:
        early = score_case(_outcome(expected=("a.md",), retrieved=("a.md", "z.md")))
        late = score_case(_outcome(expected=("a.md",), retrieved=("z.md", "a.md")))
        assert early.ndcg > late.ndcg

    def test_an_unanswerable_case_is_marked_unscored(self) -> None:
        scores = score_case(_outcome(expected=(), retrieved=("a.md",)))
        assert scores.scored is False

    def test_it_serialises(self) -> None:
        payload = score_case(_outcome(expected=("a.md",), retrieved=("a.md",))).to_dict()
        assert payload["hit"] is True
        assert payload["recall"] == 1.0
        assert payload["expected_source_paths"] == ["a.md"]


class TestAggregates:
    def _report(self, *outcomes: CaseOutcome) -> Tier1Report:
        return Tier1Report(
            mode="hybrid",
            k=5,
            collection="c",
            embedded_model="m",
            cases=tuple(score_case(outcome) for outcome in outcomes),
        )

    def test_unscored_cases_are_excluded_from_every_aggregate(self) -> None:
        # An unanswerable question has no source to hit; scoring it as a miss
        # would depress a number retrieval cannot improve.
        report = self._report(
            _outcome(expected=("a.md",), retrieved=("a.md",)),
            _outcome(expected=(), retrieved=("z.md",)),
        )
        assert report.hit_at_k == 1.0
        assert len(report.scored) == 1

    def test_hit_rate_is_the_mean_over_scored_cases(self) -> None:
        report = self._report(
            _outcome(expected=("a.md",), retrieved=("a.md",)),
            _outcome(expected=("b.md",), retrieved=("z.md",)),
        )
        assert report.hit_at_k == 0.5

    def test_mrr_averages_the_reciprocal_ranks(self) -> None:
        report = self._report(
            _outcome(expected=("a.md",), retrieved=("a.md",)),
            _outcome(expected=("b.md",), retrieved=("z.md", "b.md")),
        )
        assert report.mrr == pytest.approx(0.75)

    def test_an_empty_report_scores_zero_rather_than_dividing_by_zero(self) -> None:
        empty = Tier1Report(mode="hybrid", k=5, collection="c", embedded_model="m")
        assert (empty.hit_at_k, empty.recall_at_k, empty.mrr, empty.ndcg_at_k) == (
            0.0,
            0.0,
            0.0,
            0.0,
        )

    def test_the_category_breakdown_splits_by_label(self) -> None:
        report = self._report(
            _outcome(expected=("a.md",), retrieved=("a.md",), category="conceptual"),
            _outcome(expected=("b.md",), retrieved=("z.md",), category="exact_token"),
        )
        assert report.by_category["conceptual"]["hit_at_k"] == 1.0
        assert report.by_category["exact_token"]["hit_at_k"] == 0.0

    def test_the_summary_names_its_metrics_and_label_level(self) -> None:
        # The labels are source paths, not chunk ids. A report that did not say
        # so would be read as chunk-level recall, which it is not.
        summary = self._report(_outcome(expected=("a.md",), retrieved=("a.md",))).to_summary()
        assert summary["tier"] == 1
        assert summary["metrics"] == list(TIER1_METRICS)
        assert summary["label_level"] == "source_path"

    def test_the_summary_lists_the_misses_separately(self) -> None:
        summary = self._report(
            _outcome(expected=("a.md",), retrieved=("a.md",)),
            _outcome(expected=("b.md",), retrieved=("z.md",)),
        ).to_summary()
        assert [miss["expected_source_paths"] for miss in summary["misses"]] == [["b.md"]]
        assert len(summary["results"]) == 2


class TestEvaluateTier1:
    def test_it_scores_the_golden_cases_against_a_live_retriever(self) -> None:
        cases = [
            _case("q-1", "reciprocal rank fusion", "sample/01-hybrid-search.md"),
            _case("q-2", "cross-encoder", "sample/02-reranking.md"),
        ]
        report = evaluate_tier1(retriever=_retriever(), cases=cases, k=3, mode=MODE_SPARSE)
        assert report.hit_at_k == 1.0
        assert report.mode == MODE_SPARSE
        assert report.k == 3

    def test_the_collection_and_model_are_recorded(self) -> None:
        report = evaluate_tier1(
            retriever=_retriever(),
            cases=[_case("q-1", "fusion", "sample/01-hybrid-search.md")],
            k=2,
        )
        assert report.collection == "eval_collection"
        assert report.embedded_model

    def test_a_question_with_no_matching_document_is_a_miss(self) -> None:
        cases = [_case("q-1", "kubernetes autoscaling", "sample/99-missing.md")]
        report = evaluate_tier1(retriever=_retriever(), cases=cases, k=3, mode=MODE_SPARSE)
        assert report.hit_at_k == 0.0
        assert report.to_summary()["misses"]

    def test_an_unanswerable_case_runs_but_does_not_score(self) -> None:
        cases = [
            _case("q-1", "fusion", "sample/01-hybrid-search.md"),
            _case("q-2", "how many concurrent requests", category="unanswerable"),
        ]
        report = evaluate_tier1(retriever=_retriever(), cases=cases, k=3, mode=MODE_SPARSE)
        assert len(report.cases) == 2
        assert len(report.scored) == 1

    def test_a_non_positive_k_is_rejected(self) -> None:
        with pytest.raises(EvalError, match="k must be positive"):
            evaluate_tier1(retriever=_retriever(), cases=[_case("q", "why", "a.md")], k=0)
