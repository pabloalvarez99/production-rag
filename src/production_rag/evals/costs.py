"""Configuration-derived preflight cost bounds and the explicit spending gate."""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

MODEL_RATES_USD_PER_MILLION = {
    "text-embedding-3-small": 0.02,
    "gpt-4o-mini-input": 0.15,
    "gpt-4o-mini-output": 0.60,
}


@dataclass(frozen=True, slots=True)
class CallCounts:
    """Provider operations implied by one resolved matrix configuration."""

    document_embeddings: int
    query_embeddings: int
    generations: int
    judge_calls: int


@dataclass(frozen=True, slots=True)
class CostEstimate:
    """Expected spend and a conservative upper bound, never actual spend."""

    calls: CallCounts
    expected_usd: float
    upper_bound_usd: float
    embedding_tokens: int
    query_tokens: int

    @property
    def estimated_usd(self) -> float:
        """Compatibility alias for callers that display the expected value."""
        return self.expected_usd


def derive_call_counts(
    *,
    items: int,
    configurations: Mapping[str, tuple[str, str]],
    chunks_to_embed: int,
    ingest: bool,
    llm_provider: str,
    judge_provider: str,
) -> CallCounts:
    """Derive operations from the exact modes and providers the matrix will run."""
    query_embeddings = sum(
        items * 2 for mode, _ in configurations.values() if mode in {"dense", "hybrid"}
    )
    generations = items * len(configurations) if llm_provider != "none" else 0
    judge_calls = generations if judge_provider != "none" else 0
    return CallCounts(
        document_embeddings=chunks_to_embed if ingest else 0,
        query_embeddings=query_embeddings,
        generations=generations,
        judge_calls=judge_calls,
    )


def exact_token_count(texts: Iterable[str], *, model: str) -> int:
    """Count provider input tokens with tiktoken; fail rather than approximate."""
    try:
        tiktoken: Any = importlib.import_module("tiktoken")
    except ImportError as exc:
        raise RuntimeError("exact hosted-cost preflight requires the tiktoken package") from exc
    encoding = tiktoken.encoding_for_model(model)
    return sum(len(encoding.encode(text)) for text in texts)


def estimate_cost(
    *,
    calls: CallCounts,
    embedding_tokens: int,
    query_tokens: int,
    max_context_tokens: int,
    max_output_tokens: int,
    billed: bool,
) -> CostEstimate:
    """Compute a labelled expectation and upper bound from resolved ceilings."""
    if not billed:
        return CostEstimate(calls, 0.0, 0.0, embedding_tokens, query_tokens)
    embedding_total = embedding_tokens + query_tokens
    generation_input_bound = calls.generations * max_context_tokens
    generation_output_bound = calls.generations * max_output_tokens
    # Expected usage is deliberately separate from the guardrail ceiling. Until
    # provider usage history exists, half-utilisation is a visible planning
    # assumption rather than a number masquerading as measured spend.
    expected_input = generation_input_bound // 2
    expected_output = generation_output_bound // 2
    rate = MODEL_RATES_USD_PER_MILLION
    embedding_usd = embedding_total * rate["text-embedding-3-small"]
    expected = embedding_usd + expected_input * rate["gpt-4o-mini-input"]
    expected += expected_output * rate["gpt-4o-mini-output"]
    upper = embedding_usd + generation_input_bound * rate["gpt-4o-mini-input"]
    upper += generation_output_bound * rate["gpt-4o-mini-output"]
    return CostEstimate(
        calls=calls,
        expected_usd=round(expected / 1_000_000, 4),
        upper_bound_usd=round(upper / 1_000_000, 4),
        embedding_tokens=embedding_tokens,
        query_tokens=query_tokens,
    )


def require_spending_consent(*, billed: bool, yes_spend: bool) -> None:
    """Refuse any hosted-provider run without the explicit flag."""
    if billed and not yes_spend:
        raise ValueError("hosted providers require --yes-spend after reviewing the estimate")
