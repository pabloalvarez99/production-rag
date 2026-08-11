"""Tier 1: retrieval metrics. Deterministic, free, and runnable on every change.

Built on :mod:`production_rag.evals.source_hit` rather than beside it. That
module already owns golden parsing, source-path normalisation, hit attribution
and the unanswerable-case exclusion; a second implementation of any of those
would drift, and the two numbers would disagree for reasons nobody could
attribute. This module runs that scorer once and derives the rest of ADR-0003's
retrieval metrics from the per-case outcomes it returns.

    hit@k     did any expected source appear in the top k
    recall@k  what share of a case's expected sources appeared
    mrr       1 / rank of the first expected source, averaged
    ndcg@k    rank-weighted, binary gain

**These are source-level metrics, not chunk-level.** ADR-0003 specifies
``relevant_chunk_ids``; the golden set labels ``expected_source_paths``, because
chunk-level labels are invalidated by any change to chunk size or overlap and
the corpus is small enough that the document is the useful unit. The names here
say ``source`` for exactly that reason — reporting a document-level number under
the label ``recall@k`` would overstate what was measured. Chunk-level labels
remain the upgrade, and the metric functions do not change when they arrive.

Multi-source cases are why ``hit@k`` and ``recall@k`` are both reported: a
question whose answer spans two documents scores 1.0 on hit and 0.5 on recall
when retrieval finds one of them, and only recall notices the difference.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import log2
from typing import Any

import structlog

from production_rag.evals.source_hit import (
    DEFAULT_K,
    CaseOutcome,
    EvalError,
    GoldenCase,
    SourceHitReport,
    evaluate_source_hit,
)
from production_rag.retrieval.hybrid import Retriever

_log = structlog.get_logger(__name__)

TIER1_METRICS = ("source_hit_at_k", "source_recall_at_k", "mrr", "ndcg_at_k")
"""What this tier reports. Named in the report so a reader need not infer it."""


@dataclass(frozen=True, slots=True)
class CaseScores:
    """Per-case retrieval metrics, derived from one :class:`CaseOutcome`."""

    id: str
    question: str
    category: str | None
    scored: bool
    hit: bool
    hit_rank: int | None
    recall: float
    reciprocal_rank: float
    ndcg: float
    expected_source_paths: tuple[str, ...]
    retrieved_source_paths: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form for the per-case section of the report."""
        return {
            "id": self.id,
            "question": self.question,
            "category": self.category,
            "scored": self.scored,
            "hit": self.hit,
            "hit_rank": self.hit_rank,
            "recall": round(self.recall, 4),
            "reciprocal_rank": round(self.reciprocal_rank, 4),
            "ndcg": round(self.ndcg, 4),
            "expected_source_paths": list(self.expected_source_paths),
            "retrieved_source_paths": list(self.retrieved_source_paths),
        }


def score_case(outcome: CaseOutcome) -> CaseScores:
    """Derive the rank-aware metrics for one already-retrieved case.

    Args:
        outcome: What retrieval returned for one golden case.

    Returns:
        The per-case scores. An unscored case (no expected source, i.e. an
        unanswerable question) gets zeros that the aggregates never read — the
        aggregate is over scored cases only, and the zeros exist so the per-case
        report has a uniform shape rather than holes.
    """
    expected = set(outcome.expected_source_paths)
    if not expected:
        return CaseScores(
            id=outcome.id,
            question=outcome.question,
            category=outcome.category,
            scored=False,
            hit=False,
            hit_rank=None,
            recall=0.0,
            reciprocal_rank=0.0,
            ndcg=0.0,
            expected_source_paths=outcome.expected_source_paths,
            retrieved_source_paths=outcome.retrieved_source_paths,
        )

    retrieved = outcome.retrieved_source_paths
    # Deduplicated: several chunks from one document are one document found, and
    # counting them twice would let a single well-chunked file score recall 1.0
    # on a two-document question.
    seen: list[str] = []
    for path in retrieved:
        if path not in seen:
            seen.append(path)

    found = [path for path in seen if path in expected]
    recall = len(found) / len(expected)
    reciprocal = 1.0 / outcome.hit_rank if outcome.hit_rank else 0.0
    return CaseScores(
        id=outcome.id,
        question=outcome.question,
        category=outcome.category,
        scored=True,
        hit=outcome.hit,
        hit_rank=outcome.hit_rank,
        recall=recall,
        reciprocal_rank=reciprocal,
        ndcg=_ndcg(seen, expected),
        expected_source_paths=outcome.expected_source_paths,
        retrieved_source_paths=outcome.retrieved_source_paths,
    )


def _ndcg(retrieved: Sequence[str], expected: set[str]) -> float:
    """Binary-gain nDCG over deduplicated source paths.

    The ideal ranking puts every expected source first, so the denominator is
    bounded by how many of them exist rather than by how many were retrieved —
    a case with two expected sources cannot reach 1.0 by finding one of them at
    rank 1, which is the whole reason to report this alongside ``mrr``.
    """
    gains = sum(
        1.0 / log2(position + 1)
        for position, path in enumerate(retrieved, start=1)
        if path in expected
    )
    ideal = sum(1.0 / log2(position + 1) for position in range(1, len(expected) + 1))
    return 0.0 if ideal == 0 else gains / ideal


@dataclass(frozen=True, slots=True)
class Tier1Report:
    """Retrieval aggregates plus every per-case score."""

    mode: str
    k: int
    collection: str
    embedded_model: str
    cases: tuple[CaseScores, ...] = ()

    @property
    def scored(self) -> tuple[CaseScores, ...]:
        """Cases with an expected source; the only ones the aggregates read."""
        return tuple(case for case in self.cases if case.scored)

    @property
    def hit_at_k(self) -> float:
        """Share of scored cases with at least one expected source in the top k."""
        return self._mean([1.0 if case.hit else 0.0 for case in self.scored])

    @property
    def recall_at_k(self) -> float:
        """Mean share of each case's expected sources that were retrieved."""
        return self._mean([case.recall for case in self.scored])

    @property
    def mrr(self) -> float:
        """Mean reciprocal rank of the first expected source."""
        return self._mean([case.reciprocal_rank for case in self.scored])

    @property
    def ndcg_at_k(self) -> float:
        """Mean binary-gain nDCG."""
        return self._mean([case.ndcg for case in self.scored])

    @property
    def by_category(self) -> dict[str, dict[str, float]]:
        """Per-category hit and recall. The first breakdown anyone asks for."""
        grouped: dict[str, list[CaseScores]] = {}
        for case in self.scored:
            if case.category:
                grouped.setdefault(case.category, []).append(case)
        return {
            category: {
                "cases": float(len(members)),
                "hit_at_k": round(self._mean([1.0 if case.hit else 0.0 for case in members]), 4),
                "recall_at_k": round(self._mean([case.recall for case in members]), 4),
            }
            for category, members in sorted(grouped.items())
        }

    @staticmethod
    def _mean(values: Sequence[float]) -> float:
        """Mean, or zero for an empty sequence.

        Zero rather than a division by zero: an eval harness that crashes on an
        empty input is one nobody wires into CI.
        """
        return 0.0 if not values else sum(values) / len(values)

    def to_summary(self) -> dict[str, Any]:
        """The tier-1 section of the report."""
        return {
            "tier": 1,
            "metrics": list(TIER1_METRICS),
            "label_level": "source_path",
            "mode": self.mode,
            "k": self.k,
            "collection": self.collection,
            "embedded_model": self.embedded_model,
            "cases": len(self.cases),
            "scored_cases": len(self.scored),
            "unscored_cases": len(self.cases) - len(self.scored),
            "source_hit_at_k": round(self.hit_at_k, 4),
            "source_recall_at_k": round(self.recall_at_k, 4),
            "mrr": round(self.mrr, 4),
            "ndcg_at_k": round(self.ndcg_at_k, 4),
            "by_category": self.by_category,
            # The misses are the actionable part; the aggregate on its own is not.
            "misses": [case.to_dict() for case in self.scored if not case.hit],
            "results": [case.to_dict() for case in self.cases],
        }


def evaluate_tier1(
    *,
    retriever: Retriever,
    cases: Sequence[GoldenCase],
    k: int = DEFAULT_K,
    mode: str | None = None,
) -> Tier1Report:
    """Run the golden set through retrieval and score every tier-1 metric.

    Args:
        retriever: A configured retriever; its store and embedder define "the
            index" for this run.
        cases: The golden set, already loaded and possibly sampled.
        k: Cutoff. Passed through as ``top_k``, so exactly the hits being scored
            are the hits returned.
        mode: ``dense``, ``sparse`` or ``hybrid``. A hybrid number without the
            two single-branch numbers is not evidence of anything.

    Returns:
        The tier-1 report.

    Raises:
        EvalError: *k* is not positive.
        RetrievalError: A query could not be run.
    """
    source_report: SourceHitReport = evaluate_source_hit(
        retriever=retriever, cases=cases, k=k, mode=mode
    )
    report = Tier1Report(
        mode=source_report.mode,
        k=source_report.k,
        collection=source_report.collection,
        embedded_model=source_report.embedded_model,
        cases=tuple(score_case(outcome) for outcome in source_report.cases),
    )
    _log.info(
        "tier1_evaluated",
        mode=report.mode,
        k=report.k,
        scored_cases=len(report.scored),
        source_hit_at_k=round(report.hit_at_k, 4),
        source_recall_at_k=round(report.recall_at_k, 4),
        mrr=round(report.mrr, 4),
        ndcg_at_k=round(report.ndcg_at_k, 4),
    )
    return report


__all__ = [
    "TIER1_METRICS",
    "CaseScores",
    "EvalError",
    "Tier1Report",
    "evaluate_tier1",
    "score_case",
]
