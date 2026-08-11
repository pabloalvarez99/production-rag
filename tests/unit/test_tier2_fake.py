"""Tier 2 answer metrics on the offline path: fake embedder, fake LLM, fake judge.

Everything here runs the real query pipeline. Substituting a stub for
``run_query`` would measure the stub, which is the failure an eval is supposed to
catch rather than commit.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import pytest
import structlog

from production_rag.config_loader import RetrievalConfig
from production_rag.evals.judges import FakeJudge, JudgeError, Judgement
from production_rag.evals.source_hit import GoldenCase
from production_rag.evals.tier2_answer import (
    TIER2_METRICS,
    CaseAnswer,
    Tier2Report,
    evaluate_tier2,
    is_unanswerable,
    score_answer,
)
from production_rag.generation.citations import Citation
from production_rag.generation.llm import FakeLLM, LLMResponse
from production_rag.generation.prompts import ChatMessage
from production_rag.ingest.models import Chunk, Document
from production_rag.query_pipeline import QueryResult
from production_rag.retrieval.embeddings import FakeEmbeddingProvider
from production_rag.retrieval.hybrid import MODE_SPARSE, Retriever
from production_rag.retrieval.sparse import Bm25Encoder
from production_rag.retrieval.store import InMemoryVectorStore

CORPUS = {
    "sample/01-hybrid-search.md": "Reciprocal rank fusion merges ranked lists by position.",
    "sample/02-reranking.md": "A cross-encoder scores a query and passage together.",
}


@pytest.fixture(autouse=True)
def _reset_structlog() -> Iterator[None]:
    """Undo any global logging configuration between tests."""
    yield
    structlog.reset_defaults()


def _retriever(*, empty: bool = False) -> Retriever:
    embedder = FakeEmbeddingProvider()
    store = InMemoryVectorStore(collection="eval_collection")
    store.ensure_collection(vector_size=embedder.dimensions, with_sparse=True)
    if empty:
        return Retriever(store=store, embedder=embedder, config=RetrievalConfig())

    chunks = [
        Chunk.build(
            document=Document(source_path=path, text=text, title=path, source="sample"),
            chunk_index=index,
            text=text,
            embed_text=text,
        )
        for index, (path, text) in enumerate(CORPUS.items())
    ]
    texts = [chunk.embed_text for chunk in chunks]
    encoder = Bm25Encoder()
    encoder.fit(texts)
    store.upsert_chunks(
        chunks,
        embedder.embed_documents(texts),
        ingest_run_id="run-test",
        embedded_model=embedder.model,
        sparse_vectors=encoder.encode_documents(texts),
    )
    return Retriever(store=store, embedder=embedder, config=RetrievalConfig())


def _case(case_id: str, question: str, *paths: str, category: str | None = None) -> GoldenCase:
    return GoldenCase(id=case_id, question=question, expected_source_paths=paths, category=category)


def _citation(marker: int, source_path: str, text: str = "passage text") -> Citation:
    return Citation(
        marker=marker,
        chunk_id=f"chunk-{marker}",
        source_path=source_path,
        text=text,
        score=1.0,
        rank=marker,
    )


def _result(
    *,
    answer: str = "Grounded [1].",
    citations: tuple[Citation, ...] = (),
    refused: bool = False,
    invalid_markers: tuple[int, ...] = (),
) -> QueryResult:
    return QueryResult(
        query="why",
        answer=answer,
        citations=citations,
        refused=refused,
        invalid_markers=invalid_markers,
        hits_used=len(citations),
        model="fake-extractive-v1",
    )


class ScriptedLLM:
    """Emits a fixed answer, so a metric can be driven to a known value."""

    def __init__(self, text: str) -> None:
        self._text = text

    @property
    def model(self) -> str:
        return "scripted"

    def complete(self, messages: Sequence[ChatMessage]) -> LLMResponse:
        return LLMResponse(text=self._text, model=self.model, finish_reason="stop")


class BrokenJudge:
    """A judge that always fails."""

    @property
    def name(self) -> str:
        return "broken"

    def score(
        self,
        question: str,
        answer: str,
        contexts: Sequence[str],
        *,
        refused: bool = False,
    ) -> Judgement:
        raise JudgeError("judge model failed: upstream")


class TestUnanswerable:
    def test_the_category_marks_it(self) -> None:
        assert is_unanswerable(_case("q", "why", "a.md", category="unanswerable")) is True

    def test_an_empty_source_list_marks_it(self) -> None:
        # Two signals, because the golden set is hand-written and either one
        # alone can be forgotten.
        assert is_unanswerable(_case("q", "why")) is True

    def test_an_ordinary_case_is_answerable(self) -> None:
        assert is_unanswerable(_case("q", "why", "a.md", category="conceptual")) is False


class TestScoreAnswer:
    def test_citations_on_the_expected_source_score_one(self) -> None:
        scored = score_answer(
            _case("q", "why", "sample/01-hybrid-search.md"),
            _result(citations=(_citation(1, "sample/01-hybrid-search.md"),)),
            judge=None,
        )
        assert scored.citation_precision == 1.0

    def test_a_citation_on_the_wrong_document_lowers_precision(self) -> None:
        scored = score_answer(
            _case("q", "why", "sample/01-hybrid-search.md"),
            _result(
                citations=(
                    _citation(1, "sample/01-hybrid-search.md"),
                    _citation(2, "sample/99-elsewhere.md"),
                )
            ),
            judge=None,
        )
        assert scored.citation_precision == 0.5

    def test_citation_paths_are_normalised_like_the_golden_set(self) -> None:
        scored = score_answer(
            _case("q", "why", "sample/01-hybrid-search.md"),
            _result(citations=(_citation(1, ".\\sample\\01-hybrid-search.md"),)),
            judge=None,
        )
        assert scored.citation_precision == 1.0

    def test_precision_is_undefined_rather_than_zero_without_citations(self) -> None:
        # A refusal has no citations to be imprecise about; a zero here would
        # drag the aggregate down for correct behaviour.
        scored = score_answer(_case("q", "why", "a.md"), _result(refused=True), judge=None)
        assert scored.citation_precision is None

    def test_invalid_markers_are_counted(self) -> None:
        scored = score_answer(
            _case("q", "why", "a.md"),
            _result(citations=(_citation(1, "a.md"),), invalid_markers=(7, 9)),
            judge=None,
        )
        assert scored.invalid_markers == 2

    def test_refusing_an_unanswerable_case_is_correct(self) -> None:
        scored = score_answer(
            _case("q", "why", category="unanswerable"), _result(refused=True), judge=None
        )
        assert scored.refusal_correct is True

    def test_answering_an_unanswerable_case_is_wrong(self) -> None:
        scored = score_answer(
            _case("q", "why", category="unanswerable"), _result(refused=False), judge=None
        )
        assert scored.refusal_correct is False

    def test_refusing_an_answerable_case_is_wrong(self) -> None:
        scored = score_answer(_case("q", "why", "a.md"), _result(refused=True), judge=None)
        assert scored.refusal_correct is False

    def test_the_judge_scores_the_cited_passages(self) -> None:
        scored = score_answer(
            _case("q", "what merges ranked lists?", "a.md"),
            _result(
                answer="Fusion merges ranked lists by position [1].",
                citations=(_citation(1, "a.md", "Fusion merges ranked lists by position."),),
            ),
            judge=FakeJudge(),
        )
        assert scored.judgement is not None
        assert scored.judgement.faithfulness == 1.0

    def test_a_broken_judge_is_recorded_not_raised(self) -> None:
        # One flaky judge call must not discard a whole run's deterministic
        # metrics.
        scored = score_answer(
            _case("q", "why", "a.md"),
            _result(citations=(_citation(1, "a.md"),)),
            judge=BrokenJudge(),
        )
        assert scored.judgement is None
        assert scored.judge_error is not None
        assert scored.citation_precision == 1.0

    def test_the_answer_can_be_kept_out_of_the_report(self) -> None:
        scored = score_answer(_case("q", "why", "a.md"), _result(), judge=None)
        assert "answer" in scored.to_dict()
        assert "answer" not in scored.to_dict(include_answer=False)


class TestAggregates:
    def _report(self, *cases: CaseAnswer) -> Tier2Report:
        return Tier2Report(judge="fake", model="m", mode="hybrid", k=5, cases=cases)

    def _scored(
        self,
        *,
        expected: tuple[str, ...] = ("a.md",),
        citations: tuple[Citation, ...] = (),
        refused: bool = False,
        invalid_markers: tuple[int, ...] = (),
        category: str | None = None,
    ) -> CaseAnswer:
        return score_answer(
            GoldenCase(id="q", question="why", expected_source_paths=expected, category=category),
            _result(citations=citations, refused=refused, invalid_markers=invalid_markers),
            judge=None,
        )

    def test_refusal_accuracy_counts_answerable_cases_too(self) -> None:
        # Otherwise a system that refuses everything scores 1.0 on the metric
        # that exists to catch exactly that.
        report = self._report(
            self._scored(expected=(), refused=True, category="unanswerable"),
            self._scored(refused=True),
        )
        assert report.refusal_accuracy == 0.5

    def test_citation_precision_averages_per_case(self) -> None:
        report = self._report(
            self._scored(citations=(_citation(1, "a.md"),)),
            self._scored(citations=(_citation(1, "z.md"),)),
        )
        assert report.citation_precision == 0.5

    def test_cases_without_citations_do_not_dilute_precision(self) -> None:
        report = self._report(
            self._scored(citations=(_citation(1, "a.md"),)),
            self._scored(expected=(), refused=True, category="unanswerable"),
        )
        assert report.citation_precision == 1.0

    def test_the_invalid_marker_rate_pools_over_markers(self) -> None:
        report = self._report(
            self._scored(citations=(_citation(1, "a.md"),), invalid_markers=(9,)),
            self._scored(citations=(_citation(1, "a.md"),)),
        )
        assert report.invalid_marker_rate == pytest.approx(1 / 3)

    def test_an_empty_report_scores_zero_rather_than_dividing_by_zero(self) -> None:
        empty = Tier2Report(judge="fake", model="m", mode="hybrid", k=5)
        assert empty.refusal_accuracy == 0.0
        assert empty.invalid_marker_rate == 0.0
        assert empty.faithfulness is None

    def test_judge_columns_are_none_when_nothing_was_judged(self) -> None:
        # None, not zero: a judge that declined every case and a judge that
        # scored every case zero are different facts.
        summary = self._report(self._scored()).to_summary()
        assert summary["faithfulness"] is None
        assert summary["relevance"] is None

    def test_the_summary_names_the_judge_and_its_metrics(self) -> None:
        summary = self._report(self._scored()).to_summary()
        assert summary["tier"] == 2
        assert summary["judge"] == "fake"
        assert summary["metrics"] == list(TIER2_METRICS)

    def test_the_summary_lists_the_refusal_failures(self) -> None:
        summary = self._report(self._scored(refused=True)).to_summary()
        assert len(summary["refusal_failures"]) == 1


class TestEvaluateTier2:
    def test_the_offline_path_answers_and_cites(self) -> None:
        cases = [_case("q-1", "reciprocal rank fusion", "sample/01-hybrid-search.md")]
        report = evaluate_tier2(
            retriever=_retriever(), llm=FakeLLM(), cases=cases, mode=MODE_SPARSE, k=3
        )
        assert report.cases[0].refused is False
        assert report.citation_precision == 1.0
        assert report.refusal_accuracy == 1.0

    def test_an_empty_index_produces_a_refusal(self) -> None:
        cases = [_case("q-1", "fusion", category="unanswerable")]
        report = evaluate_tier2(
            retriever=_retriever(empty=True), llm=FakeLLM(), cases=cases, mode=MODE_SPARSE, k=3
        )
        assert report.cases[0].refused is True
        assert report.refusal_accuracy == 1.0

    def test_the_default_judge_is_the_offline_one(self) -> None:
        report = evaluate_tier2(
            retriever=_retriever(),
            llm=FakeLLM(),
            cases=[_case("q-1", "cross-encoder", "sample/02-reranking.md")],
            mode=MODE_SPARSE,
            k=3,
        )
        assert report.judge == FakeJudge().name
        assert report.faithfulness is not None

    def test_the_judge_can_be_switched_off_entirely(self) -> None:
        report = evaluate_tier2(
            retriever=_retriever(),
            llm=FakeLLM(),
            cases=[_case("q-1", "cross-encoder", "sample/02-reranking.md")],
            judge=None,
            mode=MODE_SPARSE,
            k=3,
        )
        # ``None`` means the default judge, not "no judge" -- the runner is where
        # a caller opts out, and the default has to be the offline one.
        assert report.judge == FakeJudge().name

    def test_an_invented_marker_shows_up_in_the_rate(self) -> None:
        report = evaluate_tier2(
            retriever=_retriever(),
            llm=ScriptedLLM("Fusion merges ranked lists [1] and elsewhere [9]."),
            cases=[_case("q-1", "fusion", "sample/01-hybrid-search.md")],
            mode=MODE_SPARSE,
            k=3,
        )
        assert report.invalid_marker_rate == 0.5

    def test_the_report_records_the_model_that_answered(self) -> None:
        report = evaluate_tier2(
            retriever=_retriever(),
            llm=FakeLLM(),
            cases=[_case("q-1", "fusion", "sample/01-hybrid-search.md")],
            mode=MODE_SPARSE,
            k=3,
        )
        assert report.model == FakeLLM().model
