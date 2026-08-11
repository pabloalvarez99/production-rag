"""Stable scorecard contract and its mechanical validator."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

CONFIG_NAMES = ("sparse", "dense", "hybrid", "hybrid_rerank")
METRIC_NAMES = (
    "hit_at_5",
    "recall_at_5",
    "mrr",
    "ndcg_at_5",
    "citation_precision",
    "invalid_marker_rate",
    "refusal_accuracy",
)


def validate_scorecard(payload: Mapping[str, Any]) -> None:
    """Raise ValueError unless *payload* has exactly schema version 1's shape."""
    _keys(
        payload,
        {"schema_version", "generated_at", "provenance", "configs", "slices", "comparisons"},
        "root",
    )
    if payload["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    provenance = _mapping(payload["provenance"], "provenance")
    _keys(
        provenance,
        {
            "commit",
            "corpus",
            "golden",
            "embedder",
            "llm",
            "judge",
            "cost_usd",
            "cost_estimate_usd",
            "cost_expected_usd",
            "billed",
            "bootstrap_seed",
            "bootstrap_resamples",
        },
        "provenance",
    )
    billed = provenance["billed"]
    cost_usd = provenance["cost_usd"]
    if not isinstance(billed, bool) or not isinstance(cost_usd, int | float):
        raise ValueError("provenance billed/cost_usd types are invalid")
    if billed != (float(cost_usd) > 0.0):
        raise ValueError("billed is false iff actual cost_usd is 0.0")
    if not isinstance(provenance["cost_estimate_usd"], int | float):
        raise ValueError("cost_estimate_usd must be numeric")
    if not isinstance(provenance["cost_expected_usd"], int | float):
        raise ValueError("cost_expected_usd must be numeric")
    if not isinstance(provenance["bootstrap_seed"], int):
        raise ValueError("bootstrap seed must be a non-null integer")
    if not isinstance(provenance["bootstrap_resamples"], int) or provenance[
        "bootstrap_resamples"
    ] <= 0:
        raise ValueError("bootstrap resamples must be a positive integer")
    _keys(
        _mapping(provenance["corpus"], "provenance.corpus"),
        {"path", "documents", "chunks"},
        "provenance.corpus",
    )
    _keys(
        _mapping(provenance["golden"], "provenance.golden"), {"path", "items"}, "provenance.golden"
    )
    configs = _mapping(payload["configs"], "configs")
    _keys(configs, set(CONFIG_NAMES), "configs")
    for name in CONFIG_NAMES:
        _keys(_mapping(configs[name], f"configs.{name}"), set(METRIC_NAMES), f"configs.{name}")
    slices = _mapping(payload["slices"], "slices")
    for name, raw_slice in slices.items():
        one_slice = _mapping(raw_slice, f"slices.{name}")
        _keys(one_slice, {"n", "powered", "mde_at_80_power", "configs"}, f"slices.{name}")
        slice_configs = _mapping(one_slice["configs"], f"slices.{name}.configs")
        _keys(slice_configs, set(CONFIG_NAMES), f"slices.{name}.configs")
        for config_name in CONFIG_NAMES:
            _keys(
                _mapping(slice_configs[config_name], "slice config"), {"hit_at_5"}, "slice config"
            )
    comparisons = payload["comparisons"]
    if not isinstance(comparisons, list):
        raise ValueError("comparisons must be a list")
    for index, raw in enumerate(comparisons):
        comparison = _mapping(raw, f"comparisons[{index}]")
        _keys(
            comparison,
            {"a", "b", "metric", "scope", "n", "delta", "ci95", "mcnemar", "reportable", "reason"},
            f"comparisons[{index}]",
        )
        _keys(
            _mapping(comparison["mcnemar"], "mcnemar"), {"a_only", "b_only", "p_exact"}, "mcnemar"
        )
        ci95 = comparison["ci95"]
        if not isinstance(ci95, list) or len(ci95) != 2:
            raise ValueError("ci95 must be a two-element list")


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    return value


def _keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{path} keys differ: expected {sorted(expected)}, got {sorted(value)}")
