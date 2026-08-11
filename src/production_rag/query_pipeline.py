"""The public query entry point: a question in, a cited answer or a refusal out.

    run_query(...) -> QueryResult

This is the only thing a caller — the HTTP layer, a CLI, an eval harness — needs
from the query path. Everything under :mod:`production_rag.graph` and
:mod:`production_rag.generation` is an implementation detail of this function,
and deliberately so: the orchestrator is allowed to change (ADR-0002 anticipates
a LangGraph migration touching only the wiring file) without every caller
learning about it.

Two shapes, for two lifetimes:

* :class:`QueryPipeline` — holds a compiled graph. Build one per process and
  call :meth:`QueryPipeline.run` per request. This is what a service wants;
  compiling per request is waste, and a graph that reaches for its clients at
  call time cannot be reasoned about.
* :func:`run_query` — compiles, runs once, discards. What a script, a test and
  a notebook want.

Synchronous on purpose. Every provider client in this project is the SDK's sync
client, and an async wrapper over a blocking call is a thread pool wearing a
disguise; FastAPI already runs a sync path handler in a worker thread, so the
HTTP layer loses nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog

from production_rag.config_loader import YamlConfig
from production_rag.generation.citations import Citation
from production_rag.generation.llm import LLM
from production_rag.graph.build import build_query_graph, run_graph
from production_rag.graph.nodes import QueryDeps
from production_rag.graph.state import QueryState
from production_rag.retrieval.hybrid import NO_RERANK_SUMMARY, Retriever
from production_rag.retrieval.rerank import Reranker

_log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class QueryResult:
    """The answer, its evidence, and what produced them.

    ``refused`` is the field to branch on. When it is true the ``answer`` is the
    configured refusal message, ``citations`` is empty, and ``refusal_reason``
    is one of
    :data:`~production_rag.generation.guardrails.REFUSAL_REASONS` — a closed set,
    so a caller can switch on it and an eval can group by it.

    ``timings_ms`` is per graph node, not one number for the request. "Which
    stage?" is the first question asked about a slow or wrong answer, and a
    single total cannot answer it.
    """

    query: str
    answer: str
    citations: tuple[Citation, ...] = ()
    refused: bool = False
    refusal_reason: str | None = None
    hits_used: int = 0
    hits_retrieved: int = 0
    model: str | None = None
    mode: str = ""
    collection: str = ""
    embedded_model: str = ""
    rerank: dict[str, Any] | None = None
    invalid_markers: tuple[int, ...] = ()
    uncited_claims: tuple[str, ...] = ()
    timings_ms: dict[str, float] | None = None

    @property
    def total_ms(self) -> float:
        """Sum of the per-node timings. Convenience, not a substitute for them."""
        return round(sum((self.timings_ms or {}).values()), 3)

    def to_dict(self, *, include_citation_text: bool = True) -> dict[str, Any]:
        """JSON-serialisable form. The shape an HTTP response renders."""
        return {
            "query": self.query,
            "answer": self.answer,
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
            "citations": [
                citation.to_dict(include_text=include_citation_text) for citation in self.citations
            ],
            "hits_used": self.hits_used,
            "hits_retrieved": self.hits_retrieved,
            "model": self.model,
            "mode": self.mode,
            "collection": self.collection,
            "embedded_model": self.embedded_model,
            "rerank": self.rerank if self.rerank is not None else NO_RERANK_SUMMARY,
            "invalid_markers": list(self.invalid_markers),
            "uncited_claims": list(self.uncited_claims),
            "latency_ms": dict(self.timings_ms or {}),
            "total_ms": self.total_ms,
        }


class QueryPipeline:
    """A compiled query graph plus its dependencies. Build once, call per request."""

    def __init__(self, deps: QueryDeps) -> None:
        """Compile the graph for *deps*."""
        self._deps = deps
        self._graph = build_query_graph(deps)

    @property
    def deps(self) -> QueryDeps:
        """The collaborators this pipeline was built against."""
        return self._deps

    def run(
        self,
        query: str,
        *,
        mode: str | None = None,
        top_k: int | None = None,
    ) -> QueryResult:
        """Answer *query*.

        Args:
            query: The user's question, verbatim.
            mode: ``dense``, ``sparse`` or ``hybrid``. Defaults to the configured
                mode, which is ``hybrid``.
            top_k: Chunks to retrieve. Defaults to ``retrieval.top_k``.

        Returns:
            A :class:`QueryResult`. An empty index, a stopword-only query and a
            model that abstains all produce a refusal, not an exception.

        Raises:
            RetrievalError: The query is empty, or the mode is unknown.
            EmbeddingError: The embedding provider failed.
            VectorStoreError: The store could not be queried.
            LLMError: The generation provider failed. Distinct from a refusal on
                purpose: an outage and "the documents do not say" are different
                facts, and a caller that cannot tell them apart retries the
                wrong one.
        """
        state = run_graph(self._graph, QueryState(query=query, mode=mode, top_k=top_k))
        _log.info("query_completed", **state.summary())
        return _to_result(state)


def build_query_pipeline(
    *,
    retriever: Retriever,
    llm: LLM,
    config: YamlConfig | None = None,
    reranker: Reranker | None = None,
    system_prompt: str | None = None,
) -> QueryPipeline:
    """Assemble the query pipeline from live collaborators.

    Args:
        retriever: A configured retriever. If it already carries a reranker, the
            graph's rerank node stands down rather than reranking twice.
        llm: The model. :class:`~production_rag.generation.llm.FakeLLM` keeps the
            whole path offline, which is what makes the demo runnable on a fresh
            clone and the unit suite runnable with no network.
        config: The whole YAML profile. Defaults apply when omitted.
        reranker: Reranker for the graph's own rerank node. Passed in rather
            than built here, because constructing one may need a credential from
            the environment and this is not the place that reads secrets.
        system_prompt: Overrides the configured prompt file, for experiments.
    """
    settings = config or YamlConfig()
    return QueryPipeline(
        QueryDeps(
            retriever=retriever,
            llm=llm,
            generation=settings.generation,
            reranker=reranker,
            rerank_config=settings.rerank,
            system_prompt=system_prompt,
        )
    )


def run_query(
    query: str,
    *,
    retriever: Retriever,
    llm: LLM,
    config: YamlConfig | None = None,
    mode: str | None = None,
    top_k: int | None = None,
    reranker: Reranker | None = None,
    system_prompt: str | None = None,
) -> QueryResult:
    """Answer *query* with a freshly compiled pipeline.

    The one-shot form of :class:`QueryPipeline`. A long-lived caller should build
    the pipeline once instead — see :func:`build_query_pipeline`.
    """
    pipeline = build_query_pipeline(
        retriever=retriever,
        llm=llm,
        config=config,
        reranker=reranker,
        system_prompt=system_prompt,
    )
    return pipeline.run(query, mode=mode, top_k=top_k)


def _to_result(state: QueryState) -> QueryResult:
    """Project the final graph state onto the caller-facing result."""
    retrieval = state.retrieval_result
    generation = state.generation
    return QueryResult(
        query=state.query,
        answer=state.answer,
        citations=state.citations,
        refused=state.refused,
        refusal_reason=state.refusal_reason,
        hits_used=generation.hits_used if generation else 0,
        hits_retrieved=len(state.hits),
        model=state.model,
        mode=retrieval.mode if retrieval else (state.mode or ""),
        collection=retrieval.collection if retrieval else "",
        embedded_model=retrieval.embedded_model if retrieval else "",
        rerank=retrieval.rerank.as_dict() if retrieval and retrieval.rerank else None,
        invalid_markers=generation.invalid_markers if generation else (),
        uncited_claims=generation.uncited_claims if generation else (),
        timings_ms=dict(state.timings_ms),
    )
