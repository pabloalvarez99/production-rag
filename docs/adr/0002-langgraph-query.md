# ADR 0002 — LangGraph for query-path orchestration

- **Status:** Proposed
- **Date:** 2026-08-10
- **Deciders:** production-rag maintainers
- **Supersedes:** —

## Context

The query path is a fixed sequence of stages — normalise, retrieve (dense +
sparse), fuse, rerank, generate, cite — but each stage has real operational
requirements that a straight-line function does not serve well:

- **Independent observability.** When a request is slow or wrong, the first
  question is always *which stage*. That requires per-stage timing and inputs
  and outputs, not one span around the whole call.
- **Independent testability.** Retrieval must be testable without an LLM, and
  citation mapping must be testable without a vector store.
- **Conditional edges.** Rerank is optional and `fail_open`. Generation is
  skipped entirely when no chunk clears the evidence threshold — the refusal
  path is a different edge, not an exception.
- **Room for cycles later.** Query rewriting on empty results, and self-critique
  loops on low-confidence answers, are known future stages. Both are cycles, and
  cycles are where hand-rolled pipelines turn into recursion with a depth guard
  and a comment apologising for it.

Alternatives considered: a plain async function chain; a LangChain LCEL chain;
a hand-written state machine.

## Decision

Orchestrate the query path as a **LangGraph** `StateGraph`. One node per stage,
a single typed state object threaded through, and explicit conditional edges for
the rerank bypass and the refusal path.

Constraints on the adoption, so the framework stays a scheduler and not an
architecture:

1. **Nodes contain no business logic.** Each node is a thin adapter over a
   plain, framework-free function in `src/production_rag/`. Every stage must be
   callable and unit-testable without importing LangGraph.
2. **The state object is a project-owned Pydantic model**, not a framework dict.
3. **LangGraph is confined to the query path.** Ingest stays a plain script —
   it is a batch loop with no branching and gains nothing from a graph.

LangChain's LCEL was rejected because it optimises for linear composition and
makes conditional and cyclic flows awkward. A hand-written state machine was
rejected on cost: it would end up reimplementing per-node tracing and
conditional dispatch, and worse.

## Consequences

**Positive**

- Per-node timing and state snapshots come for free, which is exactly the
  `latency_ms` breakdown the API returns and the runbook tells operators to
  read first.
- Adding query rewriting or self-critique later is a new node plus an edge, not
  a restructuring.
- Conditional edges make the refusal path a visible part of the design rather
  than an early `return` buried in a function.
- The bypass edge for a failed reranker is declarative, which makes the
  `fail_open` promise auditable.

**Negative**

- A dependency, and one in a fast-moving ecosystem. Breaking changes between
  minor versions are a real risk, which is why the version is pinned and the
  business logic is kept outside the nodes — a migration should touch only the
  graph wiring file.
- Extra indirection for what is, in M0, a nearly linear flow. The cost is paid
  up front and the benefit arrives at the first cycle.
- Debugging a graph is less obvious than stepping a function chain. Mitigated by
  the constraint that every node's inner function runs standalone.

**Neutral / follow-ups**

- `langgraph` sits in the `rag` optional-dependency extra, so the M0 API image
  does not carry it until the query path lands.
- If the graph never grows a cycle by the time the service is feature-complete,
  this decision should be revisited and probably reversed — the justification is
  branching and cycles, and it expires if they never arrive.
