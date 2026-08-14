"""Free-path difficulty predicates: trivial all-rank-1 slices must fail CI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from production_rag.evals.difficulty import (
    FREE_PATH_SLICES,
    MIN_PROGRAM_N,
    assert_program_not_trivial,
    check_difficulty,
)

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "data" / "eval" / "golden-free-path.jsonl"
RANKS = ROOT / "data" / "eval" / "difficulty-ranks.json"


def test_free_path_program_meets_n_and_slices() -> None:
    report = check_difficulty(GOLDEN, RANKS)
    assert report.ok, report.findings
    assert report.n_items >= MIN_PROGRAM_N
    for slice_name in FREE_PATH_SLICES:
        assert report.slice_counts.get(slice_name, 0) >= 1


def test_committed_program_is_not_trivial() -> None:
    """Season invariant I12: pytest fails if a slice is all trivial rank-1."""
    assert_program_not_trivial(GOLDEN, RANKS)


def test_all_rank1_slice_fails(tmp_path: Path) -> None:
    golden = tmp_path / "golden.jsonl"
    ranks = tmp_path / "ranks.json"
    items = []
    rank_table: dict[str, dict[str, int]] = {}
    for index in range(MIN_PROGRAM_N):
        category = FREE_PATH_SLICES[index % len(FREE_PATH_SLICES)]
        item_id = f"t-{index:03d}"
        if category == "unanswerable":
            items.append(
                {
                    "id": item_id,
                    "question": f"unanswerable {index}?",
                    "expected_source_paths": [],
                    "answerable": False,
                    "category": category,
                }
            )
            rank_table[item_id] = {"target_rank": 99}
        elif category == "filter":
            items.append(
                {
                    "id": item_id,
                    "question": f"filter {index}?",
                    "expected_source_paths": ["sample/00-intro.md"],
                    "answerable": True,
                    "category": category,
                    "filters": {"tags": "rag"},
                }
            )
            rank_table[item_id] = {"target_rank": 1}
        elif category == "hybrid-vs-dense":
            items.append(
                {
                    "id": item_id,
                    "question": f"token{index} appears here",
                    "expected_source_paths": ["sample/00-intro.md"],
                    "answerable": True,
                    "category": category,
                    "rare_token": f"token{index}",
                }
            )
            rank_table[item_id] = {"target_rank": 1}
        else:
            items.append(
                {
                    "id": item_id,
                    "question": f"answerable {index}?",
                    "expected_source_paths": ["sample/00-intro.md"],
                    "answerable": True,
                    "category": category,
                }
            )
            rank_table[item_id] = {"target_rank": 1}
    golden.write_text(
        "\n".join(json.dumps(item) for item in items) + "\n", encoding="utf-8"
    )
    ranks.write_text(
        json.dumps({"baseline": "test", "items": rank_table}), encoding="utf-8"
    )
    report = check_difficulty(golden, ranks)
    assert not report.ok
    codes = {finding.code for finding in report.findings}
    assert "slice_all_rank1" in codes or "all_trivial_rank1" in codes
    with pytest.raises(AssertionError, match="difficulty predicates failed"):
        assert_program_not_trivial(golden, ranks)


def test_scorecard_ablation_labeled_plumbing() -> None:
    """Published free-path scorecard must stay billed=false and plumbing-labeled."""
    scorecard_path = ROOT / "data" / "eval" / "reports" / "scorecard.json"
    html_path = ROOT / "docs" / "assets" / "scorecard.html"
    payload = json.loads(scorecard_path.read_text(encoding="utf-8"))
    assert payload["provenance"]["billed"] is False
    assert payload["provenance"]["embedder"] in {"fake", "hash", "FakeEmbedding"} or (
        "fake" in str(payload["provenance"]["embedder"]).lower()
    )
    # Config names are the ablation surface; HTML must call them plumbing.
    for name in ("dense", "sparse", "hybrid", "hybrid_rerank"):
        assert name in payload["configs"]
    html = html_path.read_text(encoding="utf-8")
    assert "plumbing" in html.lower()
    assert "billed" in html.lower()
    assert "not SOTA" in html or "not-SOTA" in html or "not SOTA" in html
