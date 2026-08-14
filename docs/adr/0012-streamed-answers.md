# ADR 0012 — Streamed answers are additive; FakeLLM chunks are the fixture

- **Status:** Accepted
- **Date:** 2026-08-14
- **Deciders:** production-rag maintainers
- **Relates to:** [ADR 0005](0005-grounded-generation.md) (citations and
  refusal decide the answer *after* generation),
  [ADR 0002](0002-langgraph-query.md) (one query path the stream must not fork)

## Context

A reviewer staring at a blank panel for several seconds cannot tell a working
system from a hung one. That is a product problem, not a retrieval one. The
temptation is to replace `POST /v1/query` with a stream, or to treat every
token the model emits as "the answer" as it arrives.

Neither is acceptable here. The citation contract and the refusal edge run
*after* the model finishes, against the full text mapped onto the prompt blocks
([ADR 0005](0005-grounded-generation.md)). Text that has not yet been through
that gate is provisional. Presenting it as an answer would force the UI to take
it back when the guardrail refuses — the exact failure mode this repository
refuses everywhere else.

A second temptation is to re-implement the query path for streaming so deltas
can be `yield`ed from inside generation. That fork would drift the moment a
guardrail rule changes: one path would refuse, the other would not.

## Decision

### 1. `POST /v1/query` stays the contract; stream is additive

`POST /v1/query/stream` is a second route that returns the **same**
`QueryResponse` body (or the same failure class) once generation finishes.
The JSON route does not change shape, status codes, or meaning. Clients that
already call `/v1/query` keep working without discovering streaming.

The stream always mounts. A route that appears and disappears with a profile is
a contract no client can code against. What the profile decides
(`generation.stream`) is only the demo form's default toggle.

### 2. Deltas are provisional; `result` is authoritative

Wire events, in order:

| Event | Meaning |
| --- | --- |
| `meta` | request id, before any work |
| `delta` | provisional model text — never the answer |
| `result` | the body `/v1/query` would have returned (grounded **or** refused) |
| `error` | the run failed — never a refusal |

A refusal after many deltas is the system working: it drafted, could not ground
what it drafted, and said so. A provider outage mid-stream is an `error` event
with `refused: false`, not a soft refusal. Those two facts must stay distinct.

The UI renders deltas as a visibly unverified draft and replaces them wholesale
when `result` arrives. There is one template for the terminal fragment on both
the swap path and the stream path, so markers and refusal chrome cannot drift.

### 3. One pipeline, teed — not a second graph

`StreamingTee` wraps the configured LLM and publishes each chunk to a sink while
still answering `complete()`. The graph, CLI, and evals call `complete()` as
they always have. The stream route starts a worker thread that runs the ordinary
executor with `on_delta`; it does not re-implement retrieval, fusion, or
guardrails. A thread is the cost of *not* forking the path.

### 4. FakeLLM chunks are a fixture, not a quality claim

`FakeLLM.stream` computes the whole answer, then splits it with
`fake_chunks` (one word per chunk, trailing spaces preserved so
`"".join(chunks) == text`). Streaming tests assert **exact SSE bytes** and the
final citations. A fixture that recomputes from a tokenizer would make those
assertions machine-dependent and worthless.

This is honest about what the free path is: a deterministic plumbing double.
It makes no claim that a real model composes answers word-by-word the same way.
What it guarantees is that a client sees the same bytes in the same order on
every run, on every machine — which is what makes an assertion about a stream
worth writing.

## Consequences

- OpenAPI and the README document both routes; the contract table still leads
  with `POST /v1/query`.
- Mid-stream failures after the first byte cannot recover a non-200 status, so
  pre-stream validation (empty question, filter outside the allowlist) stays a
  typed 422 before the stream opens.
- Capture regeneration for the UI is only required when chrome that appears in
  the committed UI captures (`ui-grounded.png`, `ui-refusal.png`,
  `ui-filtered.png`, and later `ui-stream.png`) changes. The stream toggle is
  additive chrome; the grounded / refusal / filtered outcomes remain the same
  pipeline results.
