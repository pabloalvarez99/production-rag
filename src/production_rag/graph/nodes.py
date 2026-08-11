"""The nodes. Adapters and a stopwatch — no business logic (ADR-0002, rule 1).

Every function here does the same three things: call one plain function from
:mod:`production_rag.retrieval` or :mod:`production_rag.generation`, time it,
and return a partial state update. Nothing here decides anything. If a rule
about refusals, truncation or citation mapping ever appears in this file, the
constraint that keeps LangGraph a scheduler rather than an architecture has been
broken, and a version bump will take the rules with it.

Nodes take ``(state, deps)`` and are bound to their dependencies in
:mod:`production_rag.graph.build`. LangGraph calls single-argument callables, so
the alternative would be reading a retriever out of the state — which puts live
clients into a serialisable object and makes the state untestable in isolation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import structlog

from production_rag.config_loader import GenerationConfig, RerankConfig
from production_rag.generation import guardrails
from production_rag.generation.generator import (
    draft_answer,
    finalise,
    refuse,
    resolve_citations,
)
from production_rag.generation.llm import LLM
from production_rag.graph.state import (
    CITE_NODE,
    FINALISE_NODE,
    GENERATE_NODE,
    GUARD_NODE,
    RERANK_NODE,
    RETRIEVE_NODE,
    QueryState,
)
from production_rag.observability.tracer import NullTracer, Tracer
from production_rag.retrieval.hybrid import Retriever
from production_rag.retrieval.rerank import Reranker, apply_rerank

_log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class QueryDeps:
    """Live collaborators the nodes need, resolved once per graph.

    Passed in rather than constructed here for the reason
    :meth:`~production_rag.retrieval.hybrid.Retriever.from_config` gives: building
    a provider may need a credential from the environment, and the query path
    should not be the place that reads secrets.
    """

    retriever: Retriever
    llm: LLM
    generation: GenerationConfig
    reranker: Reranker | None = None
    rerank_config: RerankConfig | None = None
    system_prompt: str | None = None
    tracer: Tracer = field(default_factory=NullTracer)
    """Where node spans go. The default records nothing and costs nothing.

    A dependency like the others, so tracing is chosen once at the edge and no
    node has to ask whether it is configured.
    """

    @property
    def rerank(self) -> RerankConfig:
        """The rerank block, with documented defaults when none was supplied."""
        return self.rerank_config or RerankConfig()


class _Stopwatch:
    """Wall-clock timer for one node. A context manager so an exception still times."""

    __slots__ = ("_start", "elapsed_ms")

    def __enter__(self) -> _Stopwatch:
        self._start = time.perf_counter()
        self.elapsed_ms = 0.0
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0


def retrieve_node(state: QueryState, deps: QueryDeps) -> dict[str, object]:
    """Run hybrid retrieval for the query."""
    with _Stopwatch() as timer:
        result = deps.retriever.retrieve(state.query, mode=state.mode, top_k=state.top_k)
    return {
        "retrieval": result,
        "hits": result.hits,
        "timings_ms": state.timed(RETRIEVE_NODE, timer.elapsed_ms),
    }


def rerank_node(state: QueryState, deps: QueryDeps) -> dict[str, object]:
    """Apply the optional second-stage reranker.

    A no-op in two cases, both of which are ordinary rather than exceptional:
    no reranker was configured (the default), or the retriever already reranked
    because one was wired into it. Reranking twice would be a second bill and a
    second forward pass for an ordering that is already final.
    """
    retrieval = state.retrieval_result
    already = retrieval is not None and retrieval.rerank is not None
    if deps.reranker is None or already:
        return {"timings_ms": state.timed(RERANK_NODE, 0.0)}

    settings = deps.rerank
    with _Stopwatch() as timer:
        outcome = apply_rerank(
            deps.reranker,
            state.query,
            state.hits,
            top_n=settings.top_k,
            fail_open=settings.fail_open,
        )
    updates: dict[str, object] = {
        "hits": outcome.hits,
        "timings_ms": state.timed(RERANK_NODE, timer.elapsed_ms),
    }
    if retrieval is not None:
        # The outcome belongs on the retrieval result too: "was this ranking
        # reranked?" must stay answerable from the object a caller already has.
        updates["retrieval"] = replace(retrieval, hits=outcome.hits, rerank=outcome)
    return updates


def guard_node(state: QueryState, deps: QueryDeps) -> dict[str, object]:
    """Pre-generation guardrail: is there anything to ground an answer in?"""
    with _Stopwatch() as timer:
        refusal = guardrails.check_evidence(state.hits, config=deps.generation.citations)
    updates: dict[str, object] = {"timings_ms": state.timed(GUARD_NODE, timer.elapsed_ms)}
    if refusal is None:
        return updates
    result = refuse(refusal)
    updates.update(
        {
            "generation": result,
            "answer": result.answer,
            "refused": True,
            "refusal_reason": result.refusal_reason,
        }
    )
    return updates


def should_generate(state: QueryState) -> str:
    """Conditional edge: generate, or stop at the refusal the guard wrote.

    Reads the state rather than re-running the check, so the decision is made
    once and the edge cannot disagree with the reason already recorded.
    """
    return "refused" if state.refused else "generate"


def generate_node(state: QueryState, deps: QueryDeps) -> dict[str, object]:
    """Build the prompt and call the model once."""
    with _Stopwatch() as timer:
        draft = draft_answer(
            state.query,
            state.hits,
            llm=deps.llm,
            config=deps.generation,
            system_prompt=deps.system_prompt,
        )
    return {
        "draft": draft,
        "model": draft.response.model,
        "timings_ms": state.timed(GENERATE_NODE, timer.elapsed_ms),
    }


def cite_node(state: QueryState, deps: QueryDeps) -> dict[str, object]:
    """Map the answer's markers onto the blocks that were in the prompt."""
    if state.draft is None:  # pragma: no cover - unreachable via the compiled graph
        raise RuntimeError("cite_node ran before generate_node produced a draft")
    with _Stopwatch() as timer:
        cited = resolve_citations(state.draft)
    return {"cited": cited, "timings_ms": state.timed(CITE_NODE, timer.elapsed_ms)}


def finalise_node(state: QueryState, deps: QueryDeps) -> dict[str, object]:
    """Post-generation guardrail, and the fields a caller actually reads."""
    if state.draft is None or state.cited is None:  # pragma: no cover - unreachable
        raise RuntimeError("finalise_node ran before the answer was drafted and cited")
    with _Stopwatch() as timer:
        result = finalise(state.draft, state.cited, config=deps.generation)
    _log.info(
        "query_finalised",
        refused=result.refused,
        refusal_reason=result.refusal_reason,
        citations=len(result.citations),
        uncited_claims=len(result.uncited_claims),
        invalid_markers=len(result.invalid_markers),
    )
    return {
        "generation": result,
        "answer": result.answer,
        "citations": result.citations,
        "refused": result.refused,
        "refusal_reason": result.refusal_reason,
        "model": result.model,
        "timings_ms": state.timed(FINALISE_NODE, timer.elapsed_ms),
    }
