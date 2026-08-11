"""Preflight cost estimates and the explicit spending gate."""

from __future__ import annotations

from dataclasses import dataclass

# Assumptions, not live prices. Update deliberately when model selection changes.
# USD per one million tokens; fake providers are always zero.
MODEL_RATES_USD_PER_MILLION = {
    "text-embedding-3-small": 0.02,
    "gpt-4o-mini-input": 0.15,
    "gpt-4o-mini-output": 0.60,
}


@dataclass(frozen=True, slots=True)
class CostEstimate:
    """Calls, token assumptions, and estimated total spend."""

    chunks_to_embed: int
    queries: int
    generations: int
    judge_calls: int
    estimated_usd: float


def estimate_cost(
    *,
    chunks_to_embed: int,
    queries: int,
    generations: int,
    judge_calls: int,
    billed: bool,
) -> CostEstimate:
    """Estimate hosted cost with explicit, conservative token assumptions."""
    if not billed:
        return CostEstimate(chunks_to_embed, queries, generations, judge_calls, 0.0)
    embedding_tokens = chunks_to_embed * 250
    generation_input = generations * 3_000
    generation_output = generations * 500
    judge_input = judge_calls * 3_500
    judge_output = judge_calls * 200
    usd = (
        embedding_tokens * MODEL_RATES_USD_PER_MILLION["text-embedding-3-small"]
        + (generation_input + judge_input) * MODEL_RATES_USD_PER_MILLION["gpt-4o-mini-input"]
        + (generation_output + judge_output) * MODEL_RATES_USD_PER_MILLION["gpt-4o-mini-output"]
    ) / 1_000_000
    return CostEstimate(chunks_to_embed, queries, generations, judge_calls, round(usd, 4))


def require_spending_consent(*, billed: bool, yes_spend: bool) -> None:
    """Refuse any hosted-provider run without the explicit flag."""
    if billed and not yes_spend:
        raise ValueError("hosted providers require --yes-spend after reviewing the estimate")
