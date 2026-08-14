# ADR 0014 — Corpus hash belongs in the cache key; Qdrant stays local

- **Status:** Accepted
- **Date:** 2026-08-14
- **Deciders:** production-rag maintainers
- **Relates to:** [ADR 0013](0013-filter-aware-query-cache.md) (filter-aware
  in-process cache), [ADR 0001](0001-hybrid-qdrant.md) (local hybrid Qdrant)

## Context

v0.3.0 keyed the in-process query cache on collection name, query text, filters,
embedder id, llm id, and retrieval knobs. That prevents filtered answers from
serving unfiltered questions. It does **not** prevent two different corpora that
happen to share a collection *name* (or that a process re-points at after
re-ingest) from cross-hitting when only the name is compared.

Hosted Qdrant Cloud would give each deployment a URL and credentials. That is a
product choice that pulls secrets into CI narratives, adds a bill, and tempts
the portfolio to claim multi-tenant readiness it does not operate. The free path
is clone-local Qdrant for a reason.

## Decision

1. **Collection identity** is productized on `GET /v1/ready` (and a sidecar
   written at ingest): `embedder_id`, `chunker_version`, `doc_count`,
   `corpus_hash`, plus the collection name.
2. **Cache keys include corpus identity material** (`corpus_hash` and related
   fields). Same question + same filters + two corpora ⇒ two keys ⇒ no
   cross-hit.
3. **Wrong collection** is a typed client error (`error_type: wrong_collection`,
   HTTP 404 when the request names another collection than the process owns).
4. **Qdrant remains local** for the free path and for CI. No hosted Qdrant, no
   keys in GitHub Actions, no claim of multi-region search.

## Consequences

**Positive**

- A hiring manager can re-ask a question after swapping corpora and trust the
  answer is not a stale cache from the previous tree.
- `/ready` answers "which index do you think you have?" without dialing the
  vector store on every probe.
- Incremental ingest still no-ops unchanged docs; identity hash changes only
  when source bytes change.

**Negative**

- Identity computation walks the corpus root (cheap on sample; slower on the
  vendored docs tree). Sidecar avoids re-hashing on every request when present.
- Multi-worker deployments still need an external cache *transport* if they want
  shared answers; the key contract is the part that travels, not Redis itself.

**Neutral**

- Auth and rate limits remain platform concerns (P5). Identity is not auth.
