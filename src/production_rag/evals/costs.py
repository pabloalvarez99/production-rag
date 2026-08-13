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

UNBILLED_NOTE = "unbilled: every provider in this run is fake, so nothing is priced"
"""How a zero is reported when no provider could have charged for the run.

A bare ``$0.0000`` is ambiguous: it reads either as "no provider was called" or
as "a paid run rounded to nothing". Only the first is ever true here, and the
report says which.
"""


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
    # Carried on the estimate rather than left to the caller: a zero without it
    # cannot be told apart from a priced run that came to nothing.
    billed: bool

    @property
    def estimated_usd(self) -> float:
        """Compatibility alias for callers that display the expected value."""
        return self.expected_usd


class CostAccountingError(RuntimeError):
    """A paid response cannot be accounted for exactly or crossed its bound."""


@dataclass(slots=True)
class UsageLedger:
    """Accumulate response usage and price it with the preflight rate table."""

    billed: bool
    upper_bound_usd: float | None = None
    document_embeddings: int = 0
    query_embeddings: int = 0
    generations: int = 0
    embedding_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def record_embedding(
        self, *, kind: str, items: int, prompt_tokens: int | None
    ) -> None:
        """Record one embedding response; paid responses require provider usage."""
        if self.billed and (prompt_tokens is None or prompt_tokens < 0):
            raise CostAccountingError("paid embedding response carried no valid usage")
        if kind == "documents":
            self.document_embeddings += items
        elif kind == "query":
            self.query_embeddings += items
        else:
            raise CostAccountingError(f"unknown embedding call kind {kind!r}")
        self.embedding_tokens += prompt_tokens or 0
        self._enforce_bound()

    def record_generation(
        self, *, prompt_tokens: int | None, completion_tokens: int | None
    ) -> None:
        """Record one completion response; paid responses require both counters."""
        if self.billed and (
            prompt_tokens is None
            or completion_tokens is None
            or prompt_tokens < 0
            or completion_tokens < 0
        ):
            raise CostAccountingError("paid generation response carried no valid usage")
        self.generations += 1
        self.prompt_tokens += prompt_tokens or 0
        self.completion_tokens += completion_tokens or 0
        self._enforce_bound()

    @property
    def calls(self) -> CallCounts:
        """Observed billable operations in estimator-compatible units."""
        return CallCounts(
            self.document_embeddings,
            self.query_embeddings,
            self.generations,
            0,
        )

    @property
    def cost_usd(self) -> float:
        """Measured spend from response usage, never from an estimate."""
        if not self.billed:
            return 0.0
        rate = MODEL_RATES_USD_PER_MILLION
        total = self.embedding_tokens * rate["text-embedding-3-small"]
        total += self.prompt_tokens * rate["gpt-4o-mini-input"]
        total += self.completion_tokens * rate["gpt-4o-mini-output"]
        return round(total / 1_000_000, 6)

    def _enforce_bound(self) -> None:
        if self.upper_bound_usd is not None and self.cost_usd > self.upper_bound_usd:
            raise CostAccountingError(
                f"measured spend ${self.cost_usd:.6f} exceeded preflight bound "
                f"${self.upper_bound_usd:.6f}"
            )


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
    # Corpus text can legitimately contain strings that look like tokenizer
    # control tokens — `<|endoftext|>` appears in any document about tokenizers.
    # They are document content to be counted, not instructions to inject a
    # special token, and tiktoken's default would abort the whole preflight over
    # one such string rather than count it.
    return sum(len(encoding.encode(text, disallowed_special=())) for text in texts)


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
        return CostEstimate(
            calls=calls,
            expected_usd=0.0,
            upper_bound_usd=0.0,
            embedding_tokens=embedding_tokens,
            query_tokens=query_tokens,
            billed=False,
        )
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
        billed=True,
    )


def format_preflight(estimate: CostEstimate) -> list[str]:
    """Render the preflight an operator authorizes, labelled by billing state."""
    operations = (
        f"chunks to embed={estimate.calls.document_embeddings}; "
        f"query embeddings={estimate.calls.query_embeddings}; "
        f"generations={estimate.calls.generations}; "
        f"judge calls={estimate.calls.judge_calls}"
    )
    if not estimate.billed:
        # Printing a hosted rate table over a run that calls no hosted provider
        # makes a local sweep look priced. The operations still matter — they are
        # what the run does — so they are reported without a currency column.
        return [f"Cost preflight: $0.00 ({UNBILLED_NOTE}).", f"  local operations: {operations}"]
    lines = ["Cost preflight (USD; rate assumptions per 1M tokens):"]
    lines.extend(f"  {model}: ${rate:.4f}" for model, rate in MODEL_RATES_USD_PER_MILLION.items())
    lines.append(
        f"  {operations}; expected=${estimate.expected_usd:.4f}; "
        f"upper bound=${estimate.upper_bound_usd:.4f}"
    )
    return lines


def format_spend_summary(*, estimate: CostEstimate, actual_usd: float) -> str:
    """Render measured spend, refusing to call a non-zero total unbilled."""
    if not estimate.billed:
        if actual_usd != 0.0:
            raise CostAccountingError(
                f"unbilled run measured ${actual_usd:.6f} of spend; the run called a "
                "provider the preflight did not price"
            )
        return f"Cost summary: $0.00 ({UNBILLED_NOTE})."
    return (
        f"Cost summary: bound=${estimate.upper_bound_usd:.6f}; "
        f"expected=${estimate.expected_usd:.6f}; actual=${actual_usd:.6f}"
    )


def require_spending_consent(*, billed: bool, yes_spend: bool) -> None:
    """Refuse any hosted-provider run without the explicit flag."""
    if billed and not yes_spend:
        raise ValueError("hosted providers require --yes-spend after reviewing the estimate")
