import json
from pathlib import Path

import pytest

from production_rag.evals.costs import estimate_cost, require_spending_consent
from production_rag.evals.matrix import build_scorecard
from production_rag.evals.scorecard import CONFIG_NAMES, METRIC_NAMES, validate_scorecard
from production_rag.evals.source_hit import GoldenCase


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
        "embedder": "fake",
        "llm": "fake",
        "judge": "fake",
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
        billed=False,
    )
    destination = tmp_path / "scorecard.json"
    destination.write_text(json.dumps(payload), encoding="utf-8")
    emitted = json.loads(destination.read_text(encoding="utf-8"))
    validate_scorecard(emitted)
    assert emitted["provenance"]["billed"] is False
    assert all(not item["reportable"] for item in emitted["comparisons"])


def test_spending_gate_requires_explicit_consent() -> None:
    estimate = estimate_cost(
        chunks_to_embed=100, queries=60, generations=60, judge_calls=60, billed=True
    )
    assert estimate.estimated_usd > 0.0
    with pytest.raises(ValueError, match="--yes-spend"):
        require_spending_consent(billed=True, yes_spend=False)
    require_spending_consent(billed=True, yes_spend=True)
    require_spending_consent(billed=False, yes_spend=False)
