# ADR 0013 — Filter-aware in-process query cache (not Redis)

- **Status:** Accepted
- **Date:** 2026-08-14
- **Deciders:** production-rag maintainers
- **Relates to:** [ADR 0011](0011-metadata-filters.md) (filters change the
  evidence set), [ADR 0012](0012-streamed-answers.md) (stream still runs the
  same executor)

## Context

A grounded answer is expensive enough to notice on a live demo: embed the
question, hybrid search, optional rerank, generate, validate citations. Asking
the same question twice in a row — common when a reviewer re-runs after
scrolling citations — pays that cost again for an identical result on the free
path, where FakeLLM and FakeEmbedding are deterministic.

A cache that keys only on the question text is a correctness bug the moment
filters exist. "How does filtering work?" unfiltered and the same question with
`title=Filtering` are different questions: they retrieve different passages and
may answer with different citations. Serving the unfiltered answer under the
filtered key (or the reverse) would look like a working system and would be
wrong in the direction that is hardest to spot in a demo.

Redis (or any external cache) would also solve latency, but it adds a service
to operate, a TTL and eviction policy to defend, and a multi-tenant keyspace
to get right — none of which the free path or a clone-and-run portfolio demo
needs.

## Decision

### 1. Optional in-process LRU, off by default

`cache.enabled` in the YAML profile defaults to **false**. Production-shaped
deployments leave it false. The local demo turns it on with
`CACHE_ENABLED=true` (compose) so a reviewer who re-asks sees the hit without
anyone inventing a hosted cache tier.

The map lives in the API process. Restart empties it. That is acceptable: the
demo cost of a cold start is one query, not a wrong answer.

### 2. The key includes filters and every ranking identity

Key material:

| Field | Why |
| --- | --- |
| collection identity | answers from `prag_demo` must not serve `production_rag` |
| query text | the question |
| canonical filters | field order and empty/omitted must not alias |
| embedder id | dense space must match the index |
| llm id | FakeLLM answers are not OpenAI answers |
| retrieval fingerprint | mode, top_k, dense/sparse k, rrf k, rerank override |

A filtered answer therefore **cannot** serve an unfiltered query: the filter
string differs, the digest differs, the lookup misses.

### 3. `cache` is debug-allowlist only

`cache: "hit" | "miss"` appears only when the request sets `debug: true`, on
the same allowlist as `timings_ms` and `invalid_markers`. A normal response
stays `answer` / `citations` / `refused` / `refusal_reason`. Cache status is a
diagnostic, not a product field clients should branch on for business logic.

Streaming still consults the cache inside the shared executor: a hit skips
generation, so the client may see zero deltas and a terminal `result` — which
is the honest description of "the answer was already known".

### 4. Why this is not Redis

- Single writer (the API process); no shared state across workers is promised.
- Free path has no Redis dependency and CI installs none.
- Correctness depends on key completeness, not on network consistency.
- When multi-worker production caching is needed, the key contract here is the
  part to keep; the transport can change without teaching filters to a new
  layer from scratch.

## Consequences

- Re-ingest or collection recreate should be followed by a process restart (or
  an explicit clear) so stale answers cannot outlive their evidence.
- Tests assert hit, miss, and filter-mismatch explicitly; a regression that
  drops filters from the key fails those tests rather than a silent demo.
