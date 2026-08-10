# ADR 0002: LangGraph for the query pipeline

Status: Proposed
Date: 2026-08-10

## Context

The query path has several steps — embed, hybrid-retrieve, (later) rerank,
generate, cite — that we want to observe individually, test in isolation, and
extend (e.g. add a rerank node or a guardrail node) without rewriting the
endpoint each time. A hand-rolled chain of function calls makes each step
easy to write but hard to trace and easy to tangle as conditional logic
appears.

## Decision

Orchestrate the query pipeline as a LangGraph graph: each step is a node,
state is a typed object passed between nodes, and the FastAPI route only
invokes the compiled graph. FastAPI stays responsible for HTTP concerns
(validation, auth later, error mapping) and nothing else.

## Consequences

- Per-node tracing drops naturally out of the graph runtime, which feeds the
  eval harness (ADR 0003) and future observability.
- Nodes are plain functions: unit-testable without HTTP or Qdrant.
- Adding a step is a graph edit, not a route rewrite.
- Cost accepted: one more dependency and its learning curve; graph
  abstraction is overkill if the pipeline were to stay a straight line
  forever — we judge it will not.
