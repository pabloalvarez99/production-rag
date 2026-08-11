"""Grounded generation: retrieved chunks in, a cited answer or a refusal out.

    hits -> guardrail (evidence?) -> prompt -> LLM -> citations -> guardrail -> answer

This module is the composition, and it is deliberately a plain function. The
LangGraph nodes in :mod:`production_rag.graph` call the same four steps
individually so each gets its own timing and its own edge (ADR-0002), but
:func:`generate_answer` runs the whole stage with no framework at all — which is
what makes generation testable without a graph, and gives the eval harness a
callable that is not an orchestrator.

Two outcomes, never an exception for the ordinary case: an answer with
citations, or a refusal with a reason code. A provider failure *is* an
exception (:class:`~production_rag.generation.llm.LLMError`), because an outage
and "the documents do not say" are different facts and a caller that cannot tell
them apart will retry the wrong one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import structlog

from production_rag.config_loader import GenerationConfig
from production_rag.generation import guardrails
from production_rag.generation.citations import Citation, CitationResult, extract_citations
from production_rag.generation.llm import LLM, LLMResponse
from production_rag.generation.prompts import RenderedPrompt, build_prompt
from production_rag.retrieval.hybrid import RetrievalHit

_log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """An answer and its evidence, or a refusal and its reason.

    ``refused`` is the field a caller branches on; ``refusal_reason`` is one of
    :data:`~production_rag.generation.guardrails.REFUSAL_REASONS` and is ``None``
    exactly when ``refused`` is false.

    ``uncited_claims`` is reported on served answers too. It is not a failure —
    a refusal on one uncited transition sentence would make the guardrail
    useless — but it is the citation-coverage signal, measured on every request
    rather than sampled offline.
    """

    answer: str
    citations: tuple[Citation, ...] = ()
    refused: bool = False
    refusal_reason: str | None = None
    model: str | None = None
    hits_used: int = 0
    dropped_hits: int = 0
    invalid_markers: tuple[int, ...] = ()
    uncited_claims: tuple[str, ...] = ()
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    def to_dict(self, *, include_citation_text: bool = True) -> dict[str, Any]:
        """JSON-serialisable form."""
        return {
            "answer": self.answer,
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
            "model": self.model,
            "hits_used": self.hits_used,
            "dropped_hits": self.dropped_hits,
            "invalid_markers": list(self.invalid_markers),
            "uncited_claims": list(self.uncited_claims),
            "citations": [
                citation.to_dict(include_text=include_citation_text) for citation in self.citations
            ],
        }


@dataclass(frozen=True, slots=True)
class Draft:
    """A raw completion next to the prompt that produced it.

    The prompt travels with the answer because citation mapping needs the
    numbered blocks, not the hits: those two lists differ exactly when
    truncation happened, and that is the case where using the wrong one
    produces a citation pointing at a passage the model never saw.
    """

    prompt: RenderedPrompt
    response: LLMResponse

    @property
    def text(self) -> str:
        """The model's raw text."""
        return self.response.text


def refuse(refusal: guardrails.Refusal, *, model: str | None = None) -> GenerationResult:
    """Package a guardrail decision as a result.

    A refusal carries no citations by construction: if there were evidence worth
    citing, the pipeline would not be refusing.
    """
    return GenerationResult(
        answer=refusal.message,
        refused=True,
        refusal_reason=refusal.reason,
        model=model,
    )


def draft_answer(
    query: str,
    hits: Sequence[RetrievalHit],
    *,
    llm: LLM,
    config: GenerationConfig | None = None,
    system_prompt: str | None = None,
) -> Draft:
    """Build the prompt and call the model once.

    Raises:
        LLMError: The provider failed. Not caught here — an outage must not be
            served as "the documents do not say".
    """
    settings = config or GenerationConfig()
    prompt = build_prompt(query, hits, config=settings, system_prompt=system_prompt)
    response = llm.complete(prompt.messages)
    _log.info(
        "generation_completed",
        model=response.model,
        hits_used=prompt.hits_used,
        dropped_hits=prompt.dropped_hits,
        estimated_context_tokens=prompt.estimated_context_tokens,
        finish_reason=response.finish_reason,
    )
    return Draft(prompt=prompt, response=response)


def resolve_citations(draft: Draft) -> CitationResult:
    """Map the draft's markers onto the blocks that were in its prompt."""
    return extract_citations(draft.text, draft.prompt.blocks)


def finalise(
    draft: Draft,
    cited: CitationResult,
    *,
    config: GenerationConfig | None = None,
) -> GenerationResult:
    """Apply the post-generation guardrail and package the result."""
    settings = config or GenerationConfig()
    refusal = guardrails.check_answer(cited.answer, cited.citations, config=settings.citations)
    if refusal is not None:
        return refuse(refusal, model=draft.response.model)
    return GenerationResult(
        answer=cited.answer,
        citations=cited.citations,
        refused=False,
        model=draft.response.model,
        hits_used=draft.prompt.hits_used,
        dropped_hits=draft.prompt.dropped_hits,
        invalid_markers=cited.invalid_markers,
        uncited_claims=guardrails.uncited_sentences(cited.answer),
        prompt_tokens=draft.response.prompt_tokens,
        completion_tokens=draft.response.completion_tokens,
    )


def generate_answer(
    query: str,
    hits: Sequence[RetrievalHit],
    *,
    llm: LLM,
    config: GenerationConfig | None = None,
    system_prompt: str | None = None,
) -> GenerationResult:
    """Generate a grounded answer for *query* over *hits*.

    Args:
        query: The user's question, verbatim.
        hits: Retrieved chunks, best first. Empty is a legitimate input and
            produces a refusal without calling the model.
        llm: The model to call. :class:`~production_rag.generation.llm.FakeLLM`
            keeps this offline.
        config: The ``generation`` block; documented defaults when omitted.
        system_prompt: Overrides the configured prompt file, for experiments.

    Returns:
        A :class:`GenerationResult` — an answer with citations, or a refusal
        with a reason.

    Raises:
        LLMError: The provider failed.
    """
    settings = config or GenerationConfig()
    refusal = guardrails.check_evidence(hits, config=settings.citations)
    if refusal is not None:
        return refuse(refusal)
    draft = draft_answer(query, hits, llm=llm, config=settings, system_prompt=system_prompt)
    return finalise(draft, resolve_citations(draft), config=settings)


# Re-exported for the graph nodes, which adapt these names one-for-one.
__all__ = [
    "Draft",
    "GenerationResult",
    "draft_answer",
    "finalise",
    "generate_answer",
    "refuse",
    "resolve_citations",
]
