"""Dependency-free paired statistics for evaluation comparisons."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20_260_811


@dataclass(frozen=True, slots=True)
class McNemarResult:
    """Discordant counts and the exact two-sided binomial p-value."""

    a_only: int
    b_only: int
    p_exact: float


@dataclass(frozen=True, slots=True)
class Reportability:
    """Mechanical publication decision for one comparison."""

    reportable: bool
    reason: str


def _validate_pair(a: Sequence[bool], b: Sequence[bool]) -> None:
    if len(a) != len(b):
        raise ValueError("paired outcome vectors must have equal length")
    if not a:
        raise ValueError("paired outcome vectors must not be empty")


def paired_delta(a: Sequence[bool], b: Sequence[bool]) -> float:
    """Return mean(a) - mean(b) over aligned binary outcomes."""
    _validate_pair(a, b)
    return sum(int(left) - int(right) for left, right in zip(a, b, strict=True)) / len(a)


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    """Linearly interpolated percentile, matching the common type-7 definition."""
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def paired_percentile_bootstrap(
    a: Sequence[bool],
    b: Sequence[bool],
    *,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Bootstrap a paired mean delta by resampling shared item indices."""
    _validate_pair(a, b)
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    differences = [int(left) - int(right) for left, right in zip(a, b, strict=True)]
    rng = random.Random(seed)  # noqa: S311 - deterministic statistical simulation
    n = len(differences)
    draws = sorted(
        sum(differences[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples)
    )
    return (_percentile(draws, 0.025), _percentile(draws, 0.975))


def exact_mcnemar(a: Sequence[bool], b: Sequence[bool]) -> McNemarResult:
    """Compute exact two-sided McNemar using the discordant-pair binomial."""
    _validate_pair(a, b)
    a_only = sum(left and not right for left, right in zip(a, b, strict=True))
    b_only = sum(right and not left for left, right in zip(a, b, strict=True))
    discordant = a_only + b_only
    if discordant == 0:
        return McNemarResult(a_only, b_only, 1.0)
    tail = sum(math.comb(discordant, k) for k in range(min(a_only, b_only) + 1))
    p_exact = min(1.0, 2.0 * tail / (2**discordant))
    return McNemarResult(a_only, b_only, p_exact)


def reportability(n: int, ci95: tuple[float, float]) -> Reportability:
    """Apply the fixed n>=30 and CI-excludes-zero publication rule."""
    if n < 30:
        return Reportability(False, "n < 30")
    if ci95[0] <= 0.0 <= ci95[1]:
        return Reportability(False, "ci95 includes zero")
    return Reportability(True, "")


def minimum_detectable_effect(
    n: int,
    base_rate: float,
    *,
    power: float = 0.8,
    simulations: int = 2_000,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    step: float = 0.01,
) -> float:
    """Simulate the smallest paired improvement reaching the requested power.

    Outcomes use shared uniforms, a reproducible paired model with maximum
    positive correlation. The simulation uses whichever direction from the
    base rate has more room; MDE is an absolute magnitude, not a preference for
    improvement over degradation. The result is rounded to the grid.
    """
    if n <= 0 or not 0.0 <= base_rate <= 1.0:
        raise ValueError("n must be positive and base_rate must be in [0, 1]")
    if not 0.0 < power < 1.0 or simulations <= 0 or not 0.0 < step <= 1.0:
        raise ValueError("invalid simulation parameters")
    improving = base_rate <= 0.5
    maximum = 1.0 - base_rate if improving else base_rate
    steps = math.floor(maximum / step + 1e-12)
    for index in range(1, steps + 1):
        effect = index * step
        rng = random.Random(seed + index)  # noqa: S311 - deterministic simulation
        significant = 0
        for _ in range(simulations):
            uniforms = [rng.random() for _ in range(n)]
            baseline = [value < base_rate for value in uniforms]
            changed_rate = base_rate + effect if improving else base_rate - effect
            changed = [value < changed_rate for value in uniforms]
            if exact_mcnemar(changed, baseline).p_exact < 0.05:
                significant += 1
        if significant / simulations >= power:
            return round(effect, 2)
    return round(maximum, 2)
