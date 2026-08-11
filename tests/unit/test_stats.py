import json

import pytest

from production_rag.evals.matrix import build_scorecard
from production_rag.evals.scorecard import CONFIG_NAMES, METRIC_NAMES
from production_rag.evals.source_hit import GoldenCase
from production_rag.evals.stats import (
    exact_mcnemar,
    paired_percentile_bootstrap,
    reportability,
)


def test_exact_mcnemar_known_discordant_table() -> None:
    a = [True] * 12 + [False] * 3
    b = [False] * 12 + [True] * 3
    result = exact_mcnemar(a, b)
    assert (result.a_only, result.b_only) == (12, 3)
    assert result.p_exact == pytest.approx(0.03515625)


def test_paired_bootstrap_fixed_seed_has_pinned_interval() -> None:
    a = [True, True, True, False, False, True, False, True]
    b = [False, True, False, False, True, False, False, True]
    assert paired_percentile_bootstrap(a, b, resamples=1_000, seed=7) == (-0.25, 0.75)


def test_same_seed_produces_byte_identical_comparisons() -> None:
    cases = [
        GoldenCase(str(index), f"q{index}", ("source.md",), "slice", True)
        for index in range(30)
    ]
    checkpoint = {
        "bootstrap_seed": 7,
        "bootstrap_resamples": 100,
        "embedder": "fake",
        "llm": "fake",
        "judge": "none",
        "configs": {
            name: {
                "metrics": dict.fromkeys(METRIC_NAMES, 0.0),
                "hit_vector": [index % 2 == 0 for index in range(30)],
            }
            for name in CONFIG_NAMES
        },
    }
    first = build_scorecard(
        checkpoint=checkpoint,
        cases=cases,
        commit="abc",
        documents=1,
        chunks=1,
        golden_path="golden.jsonl",
        corpus_path="corpus",
        cost_usd=0.0,
        cost_estimate_usd=0.0,
        cost_expected_usd=0.0,
        billed=False,
    )
    second = build_scorecard(
        checkpoint=checkpoint,
        cases=cases,
        commit="abc",
        documents=1,
        chunks=1,
        golden_path="golden.jsonl",
        corpus_path="corpus",
        cost_usd=0.0,
        cost_estimate_usd=0.0,
        cost_expected_usd=0.0,
        billed=False,
    )
    assert json.dumps(first["comparisons"], sort_keys=True) == json.dumps(
        second["comparisons"], sort_keys=True
    )


@pytest.mark.parametrize(
    ("n", "ci", "expected", "reason"),
    [
        (29, (0.1, 0.2), False, "n < 30"),
        (30, (0.0, 0.2), False, "ci95 includes zero"),
        (30, (-0.2, 0.0), False, "ci95 includes zero"),
        (30, (0.01, 0.2), True, ""),
        (30, (-0.2, -0.01), True, ""),
    ],
)
def test_reportability_boundaries(
    n: int, ci: tuple[float, float], expected: bool, reason: str
) -> None:
    result = reportability(n, ci)
    assert result.reportable is expected
    assert result.reason == reason
