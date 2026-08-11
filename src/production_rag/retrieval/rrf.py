"""Reciprocal Rank Fusion: combine ranked lists that have incomparable scores.

    score(d) = Σ_branches  weight_b / (k + rank_b(d))

Cosine similarity and BM25 live on different scales, and normalising them
requires corpus-dependent calibration that drifts as the corpus grows. Rank
position does not drift. That is the whole argument for RRF here, and it is why
this module takes **ranks, not scores** — see
[ADR 0001](../../../docs/adr/0001-hybrid-qdrant.md).

Pure functions over plain identifiers: no Qdrant types, no embeddings, no IO.
Fusion is the one part of the retrieval path that must be reproducible in an
offline eval run years from now, so it is kept free of every dependency.

Ranks are **1-based**: the top hit of a branch is rank 1, contributing
``1 / (k + 1)``. Using 0-based ranks would make the constant ``k`` mean something
different from the value in the literature and in the config file.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

DEFAULT_RRF_K = 60
"""Smoothing constant from the original RRF paper.

Larger values flatten the curve, so deep ranks matter more; smaller values make
the fusion behave more like "whatever was first". 60 is the published default and
changing it is an eval decision, not a taste decision.
"""


@dataclass(frozen=True, slots=True)
class FusedHit:
    """One document after fusion, with the evidence for its position.

    ``ranks`` and ``contributions`` exist so a retrieval result can be explained:
    "this chunk is second because BM25 ranked it 1st and dense ranked it 14th" is
    a debuggable statement, while a bare fused score is not.
    """

    key: str
    score: float
    ranks: dict[str, int] = field(default_factory=dict)
    contributions: dict[str, float] = field(default_factory=dict)

    @property
    def branches(self) -> tuple[str, ...]:
        """Names of the branches that retrieved this document, in fusion order."""
        return tuple(self.ranks)


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[str]],
    *,
    k: int = DEFAULT_RRF_K,
    weights: Mapping[str, float] | None = None,
    limit: int | None = None,
) -> list[FusedHit]:
    """Fuse per-branch ranked lists into one ranking.

    Args:
        rankings: Branch name to that branch's keys in rank order, best first.
            Duplicate keys within one branch are ignored after the first
            occurrence — a branch cannot vote twice for the same document.
        k: RRF smoothing constant. Must be positive.
        weights: Optional per-branch multiplier. A missing branch defaults to
            ``1.0``; a weight of ``0.0`` disables a branch's contribution while
            keeping it visible in ``ranks``, which is what makes an ablation
            ("what does dense alone give us?") a config change rather than a code
            change.
        limit: Keep at most this many fused hits.

    Returns:
        Hits sorted by fused score, descending. Ties break on the key so the
        output is deterministic — an eval number that changes between runs
        because of dict ordering is worse than no number.

    Raises:
        ValueError: *k* is not positive, or *limit* is negative.
    """
    if k <= 0:
        raise ValueError(f"RRF k must be positive, got {k}")
    if limit is not None and limit < 0:
        raise ValueError(f"limit must not be negative, got {limit}")

    resolved_weights = dict(weights or {})
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    contributions: dict[str, dict[str, float]] = {}

    for branch, keys in rankings.items():
        weight = resolved_weights.get(branch, 1.0)
        for position, key in enumerate(_deduplicated(keys), start=1):
            if key in ranks and branch in ranks[key]:
                continue
            contribution = weight / (k + position)
            scores[key] = scores.get(key, 0.0) + contribution
            ranks.setdefault(key, {})[branch] = position
            contributions.setdefault(key, {})[branch] = contribution

    fused = [
        FusedHit(
            key=key,
            score=scores[key],
            ranks=dict(ranks[key]),
            contributions=dict(contributions[key]),
        )
        for key in scores
    ]
    fused.sort(key=lambda hit: (-hit.score, hit.key))
    return fused if limit is None else fused[:limit]


def rank_keys(scored: Sequence[tuple[str, float]]) -> list[str]:
    """Turn ``(key, score)`` pairs into a rank-ordered list of keys.

    The adapter between a store that returns scores and fusion that consumes
    ranks. Sorting is descending by score with the key as tie-break, for the same
    determinism reason as above.
    """
    return [key for key, _ in sorted(scored, key=lambda pair: (-pair[1], pair[0]))]


def _deduplicated(keys: Iterable[str]) -> list[str]:
    """Keys in order, first occurrence only."""
    seen: set[str] = set()
    unique: list[str] = []
    for key in keys:
        if key not in seen:
            seen.add(key)
            unique.append(key)
    return unique
