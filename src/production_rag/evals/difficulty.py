"""Mechanical difficulty predicates for free-path evaluation slices.

A golden set can pass schema checks and still measure nothing: if every labelled
target is already rank 1 under the baseline retriever the slice claims to stress,
hit@k becomes a free win. These predicates reject that class of set.

The free-path program set is ``data/eval/golden-free-path.jsonl`` with slices:

* ``answerable`` — grounded path, expected sources present
* ``unanswerable`` — correct outcome is refusal
* ``filter`` — item carries filters that narrow evidence
* ``hybrid-vs-dense`` — lexical pressure (rare token / sparse should help)

Ranks come from a committed fixture (``difficulty-ranks.json``), not from a
live billed embedder. The fixture is a difficulty *claim* about the labels; it
is not a quality scorecard. Fake-provider runs remain plumbing.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FREE_PATH_SLICES = (
    "answerable",
    "unanswerable",
    "filter",
    "hybrid-vs-dense",
)
"""Slice ids required by the v1 season free-path program."""

MIN_PROGRAM_N = 50
"""Season gate: free-path program must hold at least this many items."""

MAX_RANK1_FRACTION = 0.5
"""A slice fails when half or more of its items are already target rank 1."""

TRIVIAL_RANK = 1


@dataclass(frozen=True, slots=True)
class DifficultyFinding:
    """One failing check with a machine-readable reason."""

    slice: str
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class DifficultyReport:
    """Aggregate difficulty check over a golden + ranks fixture."""

    n_items: int
    slice_counts: Mapping[str, int]
    findings: tuple[DifficultyFinding, ...]

    @property
    def ok(self) -> bool:
        return not self.findings


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a golden JSONL file; blank lines ignored."""
    items: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected object")
        items.append(value)
    return items


def load_rank_fixture(path: Path) -> dict[str, Any]:
    """Load the committed difficulty rank table."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "items" not in payload:
        raise ValueError(f"{path}: expected object with 'items'")
    return payload


def _structural_findings(items: Sequence[Mapping[str, Any]]) -> list[DifficultyFinding]:
    findings: list[DifficultyFinding] = []
    counts = Counter(str(item.get("category", "")) for item in items)
    for required in FREE_PATH_SLICES:
        if counts.get(required, 0) < 1:
            findings.append(
                DifficultyFinding(
                    slice=required,
                    code="missing_slice",
                    message=f"slice {required!r} has no items",
                )
            )
    if len(items) < MIN_PROGRAM_N:
        findings.append(
            DifficultyFinding(
                slice="overall",
                code="n_below_minimum",
                message=f"program n={len(items)} < {MIN_PROGRAM_N}",
            )
        )
    for item in items:
        item_id = str(item.get("id", "<missing>"))
        category = str(item.get("category", ""))
        paths = item.get("expected_source_paths", [])
        answerable = item.get("answerable")
        if category == "answerable":
            if answerable is False:
                findings.append(
                    DifficultyFinding(
                        slice=category,
                        code="answerable_flag",
                        message=f"{item_id}: answerable slice requires answerable=true",
                    )
                )
            if not isinstance(paths, list) or not paths:
                findings.append(
                    DifficultyFinding(
                        slice=category,
                        code="answerable_paths",
                        message=f"{item_id}: answerable items need expected_source_paths",
                    )
                )
        elif category == "unanswerable":
            if answerable is not False:
                findings.append(
                    DifficultyFinding(
                        slice=category,
                        code="unanswerable_flag",
                        message=f"{item_id}: unanswerable requires answerable=false",
                    )
                )
            if paths:
                findings.append(
                    DifficultyFinding(
                        slice=category,
                        code="unanswerable_paths",
                        message=f"{item_id}: unanswerable requires empty expected_source_paths",
                    )
                )
        elif category == "filter":
            filters = item.get("filters")
            if not isinstance(filters, dict) or not filters:
                findings.append(
                    DifficultyFinding(
                        slice=category,
                        code="filter_required",
                        message=f"{item_id}: filter slice items must carry non-empty filters",
                    )
                )
        elif category == "hybrid-vs-dense":
            token = item.get("rare_token")
            question = str(item.get("question", ""))
            if not token or str(token) not in question:
                findings.append(
                    DifficultyFinding(
                        slice=category,
                        code="rare_token",
                        message=(
                            f"{item_id}: hybrid-vs-dense requires rare_token present in question"
                        ),
                    )
                )
    return findings


def _rank_triviality_findings(
    items: Sequence[Mapping[str, Any]],
    rank_fixture: Mapping[str, Any],
) -> list[DifficultyFinding]:
    """Fail a slice when too many targets are already rank 1 under the fixture."""
    findings: list[DifficultyFinding] = []
    rank_table = rank_fixture.get("items", {})
    if not isinstance(rank_table, Mapping):
        return [
            DifficultyFinding(
                slice="overall",
                code="bad_rank_fixture",
                message="difficulty-ranks.json items must be an object",
            )
        ]
    by_slice: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for item in items:
        item_id = str(item.get("id", ""))
        category = str(item.get("category", ""))
        if category not in FREE_PATH_SLICES:
            continue
        entry = rank_table.get(item_id)
        if not isinstance(entry, Mapping):
            findings.append(
                DifficultyFinding(
                    slice=category,
                    code="missing_rank",
                    message=f"{item_id}: no rank fixture entry",
                )
            )
            continue
        raw_rank = entry.get("target_rank", TRIVIAL_RANK)
        try:
            target_rank = int(raw_rank)
        except (TypeError, ValueError):
            findings.append(
                DifficultyFinding(
                    slice=category,
                    code="bad_rank",
                    message=f"{item_id}: target_rank is not an int",
                )
            )
            continue
        by_slice[category].append((item_id, target_rank))

    for category, pairs in sorted(by_slice.items()):
        if category == "unanswerable":
            # Unanswerable items have no target document to rank; skip rank-1 rule.
            continue
        if not pairs:
            continue
        rank1_ids = [item_id for item_id, rank in pairs if rank <= TRIVIAL_RANK]
        fraction = len(rank1_ids) / len(pairs)
        if fraction >= MAX_RANK1_FRACTION:
            findings.append(
                DifficultyFinding(
                    slice=category,
                    code="all_trivial_rank1",
                    message=(
                        f"slice {category!r}: {len(rank1_ids)}/{len(pairs)} items are "
                        f"target rank ≤ {TRIVIAL_RANK} (fraction {fraction:.2f} ≥ "
                        f"{MAX_RANK1_FRACTION}); set is trivial for ranking stress. "
                        f"examples: {', '.join(rank1_ids[:5])}"
                    ),
                )
            )
        # Absolute form of the CONTINUE rule: if *every* item is rank-1, fail even
        # when the fraction threshold is tuned later.
        if pairs and all(rank <= TRIVIAL_RANK for _, rank in pairs):
            findings.append(
                DifficultyFinding(
                    slice=category,
                    code="slice_all_rank1",
                    message=(
                        f"slice {category!r}: every item is target rank ≤ {TRIVIAL_RANK}; "
                        "a test must fail this set"
                    ),
                )
            )
    return findings


def check_difficulty(
    golden: Path,
    ranks: Path,
) -> DifficultyReport:
    """Run structural + rank-triviality predicates on the free-path program set."""
    items = load_jsonl(golden)
    rank_fixture = load_rank_fixture(ranks)
    findings = _structural_findings(items) + _rank_triviality_findings(items, rank_fixture)
    counts = Counter(str(item.get("category", "")) for item in items)
    return DifficultyReport(
        n_items=len(items),
        slice_counts=dict(counts),
        findings=tuple(findings),
    )


def assert_program_not_trivial(golden: Path, ranks: Path) -> DifficultyReport:
    """Raise ``AssertionError`` when any difficulty predicate fails."""
    report = check_difficulty(golden, ranks)
    if not report.ok:
        lines = [f"{finding.slice}: [{finding.code}] {finding.message}" for finding in report.findings]
        raise AssertionError(
            "free-path difficulty predicates failed:\n- " + "\n- ".join(lines)
        )
    return report


__all__ = [
    "FREE_PATH_SLICES",
    "MAX_RANK1_FRACTION",
    "MIN_PROGRAM_N",
    "DifficultyFinding",
    "DifficultyReport",
    "assert_program_not_trivial",
    "check_difficulty",
    "load_jsonl",
    "load_rank_fixture",
]
