"""Tier 2: answer-side metrics over the real query path.

Three metrics need no judge at all, and they are the ones this tier leads with,
because a number that is deterministic and free is worth more than a number that
is neither:

    citation_precision   share of emitted citations pointing at an expected source
    invalid_marker_rate  share of emitted markers that resolved to nothing
    refusal_accuracy     did the system refuse exactly the unanswerable questions

Two more need one, and are optional: ``faithfulness`` and ``relevance`` from an
:class:`~production_rag.evals.judges.AnswerJudge`. The default judge is offline
and lexical, so those two columns are present in CI and mean very little there;
the report always carries the judge's name so the difference is visible rather
than assumed.

Answers come from :func:`~production_rag.query_pipeline.run_query` — the same
entry point the API calls. An eval that reimplements the pipeline measures the
reimplementation, and the divergence shows up as a passing eval over a broken
service.

``citation_precision`` is scored against ``expected_source_paths``, which makes
it a check that citations point at the right *documents*. A citation can be
document-correct and still point at a passage that does not support the
sentence; that gap is what a judge is for, and why the two kinds of metric are
reported side by side rather than averaged into one score.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import structlog

from production_rag.config_loader import YamlConfig
from production_rag.evals.judges import AnswerJudge, FakeJudge, JudgeError, Judgement
from production_rag.evals.source_hit import GoldenCase
from production_rag.generation.llm import LLM
from production_rag.ingest.hashing import normalise_source_path
from production_rag.query_pipeline import QueryResult, run_query
from production_rag.retrieval.hybrid import Retriever

if TYPE_CHECKING:
    from production_rag.retrieval.rerank import Reranker

_log = structlog.get_logger(__name__)

TIER2_METRICS = (
    "citation_precision",
    "invalid_marker_rate",
    "refusal_accuracy",
    "faithfulness",
    "relevance",
)
"""What this tier reports. The last two are the judge's and carry its name."""

UNANSWERABLE_CATEGORY = "unanswerable"
"""Golden category whose correct outcome is a refusal.

Used as a fallback by :func:`is_unanswerable` when a record carries no explicit
``answerable`` field, alongside the expected source list being empty. Two
fallback signals rather than one because the golden set is hand-written: a case
labelled ``unanswerable`` that someone gave a source, or a sourceless case
someone forgot to label, must not silently score as a system failure.
"""


@dataclass(frozen=True, slots=True)
class CaseAnswer:
    """What the pipeline did with one golden case, scored."""

    id: str
    question: str
    category: str | None
    unanswerable: bool
    refused: bool
    refusal_correct: bool
    answer: str
    citations: int
    citations_on_expected_source: int
    citation_precision: float | None
    invalid_markers: int
    cited_source_paths: tuple[str, ...]
    expected_source_paths: tuple[str, ...]
    hits_used: int
    model: str | None
    judgement: Judgement | None = None
    judge_error: str | None = None

    def to_dict(self, *, include_answer: bool = True) -> dict[str, Any]:
        """JSON-serialisable form for the per-case section of the report.

        Args:
            include_answer: Whether to embed the generated answer. On by default
                because a per-case report without the answer cannot be reviewed
                by hand — but a report file is an artefact that gets pasted
                around, and the answer quotes corpus text, so the runner can
                turn it off.
        """
        payload: dict[str, Any] = {
            "id": self.id,
            "question": self.question,
            "category": self.category,
            "unanswerable": self.unanswerable,
            "refused": self.refused,
            "refusal_correct": self.refusal_correct,
            "citations": self.citations,
            "citations_on_expected_source": self.citations_on_expected_source,
            "citation_precision": self.citation_precision,
            "invalid_markers": self.invalid_markers,
            "cited_source_paths": list(self.cited_source_paths),
            "expected_source_paths": list(self.expected_source_paths),
            "hits_used": self.hits_used,
            "model": self.model,
            "judgement": self.judgement.to_dict() if self.judgement else None,
            "judge_error": self.judge_error,
        }
        if include_answer:
            payload["answer"] = self.answer
        return payload


@dataclass(frozen=True, slots=True)
class Tier2Report:
    """Answer-side aggregates plus every per-case outcome."""

    judge: str
    model: str
    mode: str
    k: int | None
    cases: tuple[CaseAnswer, ...] = ()

    @property
    def refusal_accuracy(self) -> float:
        """Share of cases whose refuse-or-answer decision was correct.

        Over *every* case, not only the unanswerable ones. A system that refuses
        everything would otherwise score 1.0 on the metric meant to catch
        exactly that.
        """
        return _mean([1.0 if case.refusal_correct else 0.0 for case in self.cases])

    @property
    def citation_precision(self) -> float:
        """Mean per-case citation precision over cases that cited anything.

        Averaged per case rather than pooled over all citations, so one verbose
        answer with twelve citations cannot outvote ten ordinary ones.
        """
        scored = [
            case.citation_precision for case in self.cases if case.citation_precision is not None
        ]
        return _mean(scored)

    @property
    def invalid_marker_rate(self) -> float:
        """Share of emitted markers that resolved to nothing.

        Pooled over markers, not per case: this one is about how often the model
        invents a number, and every invented number counts once.
        """
        emitted = sum(case.citations + case.invalid_markers for case in self.cases)
        invalid = sum(case.invalid_markers for case in self.cases)
        return 0.0 if emitted == 0 else invalid / emitted

    @property
    def faithfulness(self) -> float | None:
        """Mean judge faithfulness, or ``None`` when nothing was scored."""
        return _optional_mean(
            [case.judgement.faithfulness for case in self.cases if case.judgement]
        )

    @property
    def relevance(self) -> float | None:
        """Mean judge relevance, or ``None`` when nothing was scored."""
        return _optional_mean([case.judgement.relevance for case in self.cases if case.judgement])

    @property
    def refusals(self) -> int:
        """How many cases the system refused."""
        return sum(1 for case in self.cases if case.refused)

    @property
    def judge_errors(self) -> int:
        """Cases the judge could not score. Excluded from its averages."""
        return sum(1 for case in self.cases if case.judge_error)

    def to_summary(self, *, include_answers: bool = True) -> dict[str, Any]:
        """The tier-2 section of the report."""
        return {
            "tier": 2,
            "metrics": list(TIER2_METRICS),
            "judge": self.judge,
            "model": self.model,
            "mode": self.mode,
            "k": self.k,
            "cases": len(self.cases),
            "unanswerable_cases": sum(1 for case in self.cases if case.unanswerable),
            "refusals": self.refusals,
            "citation_precision": round(self.citation_precision, 4),
            "invalid_marker_rate": round(self.invalid_marker_rate, 4),
            "refusal_accuracy": round(self.refusal_accuracy, 4),
            "faithfulness": _round(self.faithfulness),
            "relevance": _round(self.relevance),
            "judge_errors": self.judge_errors,
            # The cases that got the refusal decision wrong are the actionable
            # part: a missed refusal is a hallucination with citations on it.
            "refusal_failures": [
                case.to_dict(include_answer=include_answers)
                for case in self.cases
                if not case.refusal_correct
            ],
            "results": [case.to_dict(include_answer=include_answers) for case in self.cases],
        }


def _mean(values: Sequence[float]) -> float:
    """Mean, or zero for an empty sequence."""
    return 0.0 if not values else sum(values) / len(values)


def _optional_mean(values: Sequence[float | None]) -> float | None:
    """Mean of the values that exist, or ``None`` when none do.

    ``None`` rather than zero: a judge that declined to score every case and a
    judge that scored every case zero are different facts.
    """
    present = [value for value in values if value is not None]
    return None if not present else sum(present) / len(present)


def _round(value: float | None) -> float | None:
    """Round a score, preserving ``None``."""
    return None if value is None else round(value, 4)


def is_unanswerable(case: GoldenCase) -> bool:
    """Whether the correct behaviour for *case* is a refusal.

    The golden set's explicit ``answerable`` field wins when it carries one: it
    is the author's stated intent, and it is the only signal that can mark a
    question unanswerable *despite* having a plausible source, or answerable
    despite the sources being listed elsewhere. The category and the empty
    source list remain as fallbacks for records written before the field.
    """
    if case.answerable is not None:
        return not case.answerable
    return case.category == UNANSWERABLE_CATEGORY or not case.expected_source_paths


def score_answer(case: GoldenCase, result: QueryResult, *, judge: AnswerJudge | None) -> CaseAnswer:
    """Score one answered case.

    Args:
        case: The golden case.
        result: What the pipeline returned for it.
        judge: The judge, or ``None`` to skip the judged metrics entirely.

    Returns:
        The per-case scores. A judge failure is recorded on the case rather than
        raised: one flaky judge call must not discard a whole run's deterministic
        metrics, and the failure count is reported so a run where the judge fell
        over is not mistaken for a run where it agreed.
    """
    expected = set(case.expected_source_paths)
    unanswerable = is_unanswerable(case)
    cited_paths = tuple(
        normalise_source_path(citation.source_path) for citation in result.citations
    )
    on_expected = sum(1 for path in cited_paths if path in expected)
    # Undefined rather than zero when nothing was cited or nothing was expected:
    # a refusal has no citations to be imprecise about.
    precision = round(on_expected / len(cited_paths), 4) if cited_paths and expected else None

    judgement: Judgement | None = None
    judge_error: str | None = None
    if judge is not None:
        try:
            judgement = judge.score(
                case.question,
                result.answer,
                [citation.text for citation in result.citations],
                refused=result.refused,
            )
        except JudgeError as exc:
            judge_error = str(exc)
            _log.warning("judge_failed", case_id=case.id, error=str(exc))

    return CaseAnswer(
        id=case.id,
        question=case.question,
        category=case.category,
        unanswerable=unanswerable,
        refused=result.refused,
        refusal_correct=result.refused == unanswerable,
        answer=result.answer,
        citations=len(result.citations),
        citations_on_expected_source=on_expected,
        citation_precision=precision,
        invalid_markers=len(result.invalid_markers),
        cited_source_paths=cited_paths,
        expected_source_paths=case.expected_source_paths,
        hits_used=result.hits_used,
        model=result.model,
        judgement=judgement,
        judge_error=judge_error,
    )


def evaluate_tier2(
    *,
    retriever: Retriever,
    llm: LLM,
    cases: Sequence[GoldenCase],
    judge: AnswerJudge | None = None,
    config: YamlConfig | None = None,
    mode: str | None = None,
    k: int | None = None,
    reranker: Reranker | None = None,
) -> Tier2Report:
    """Answer every golden case through the real pipeline and score the answers.

    Args:
        retriever: A configured retriever.
        llm: The generator. :class:`~production_rag.generation.llm.FakeLLM` keeps
            the whole tier offline.
        cases: The golden set, already loaded and possibly sampled.
        judge: The judge, or ``None`` to skip judged metrics.
            Pass ``None`` explicitly for deterministic metrics only.
        config: The YAML profile; documented defaults when omitted.
        mode: Retrieval mode for the run.
        k: Chunks to retrieve per question.
        reranker: Optional second-stage reranker.

    Returns:
        The tier-2 report.

    Raises:
        RetrievalError: A query could not be run.
        LLMError: The generation provider failed. Not caught here: an outage
            mid-run would otherwise be averaged into the scores as if the system
            had answered badly.
    """
    resolved_judge = judge or FakeJudge()
    settings = config or YamlConfig()
    outcomes: list[CaseAnswer] = []
    model = ""
    for case in cases:
        result = run_query(
            case.question,
            retriever=retriever,
            llm=llm,
            config=settings,
            mode=mode,
            top_k=k,
            reranker=reranker,
            request_id=f"eval-{case.id}",
        )
        model = result.model or model
        outcomes.append(score_answer(case, result, judge=resolved_judge))
        _log.debug(
            "tier2_case_evaluated",
            case_id=case.id,
            refused=result.refused,
            citations=len(result.citations),
        )

    report = Tier2Report(
        judge=resolved_judge.name,
        model=model or llm.model,
        mode=mode or retriever.config.mode,
        k=k,
        cases=tuple(outcomes),
    )
    _log.info(
        "tier2_evaluated",
        judge=report.judge,
        cases=len(report.cases),
        refusals=report.refusals,
        citation_precision=round(report.citation_precision, 4),
        invalid_marker_rate=round(report.invalid_marker_rate, 4),
        refusal_accuracy=round(report.refusal_accuracy, 4),
    )
    return report
