# production-rag v0.1.0 — free-path flagship

First tagged release of the flagship RAG service. A clean clone can exercise ingest,
retrieval, answering, the web UI, and both evaluation tiers on deterministic local
providers: no credential, no billed call, and no signup.

Target commit: `678c5543baf0f4a723dc823de1f19162ba54b4a9`

## What shipped

- **Hybrid retrieval with RRF:** dense and sparse/BM25 branches run in Qdrant and are
  fused by rank rather than by incomparable raw scores.
- **Optional fail-open reranking:** a cross-encoder can reorder the fused shortlist; a
  reranker failure is reported and falls back to fusion order instead of failing the query.
- **Grounded generation:** the LangGraph query path resolves `[n]` markers against the
  exact prompt context, removes invalid markers, and explicitly refuses when evidence is
  insufficient.
- **Two offline evaluation tiers:** retrieval and answer/citation behaviour are measured
  separately, with provenance, paired statistics, and generated documentation checks.
- **A reproducible free demo:** one script starts Qdrant, ingests the sample corpus, and
  serves the query UI with local providers pinned.
- **Credential-free CI:** lint, strict type checking, tests, ingest, both eval tiers, and
  an HTTP query run with provider keys set to empty values.
- **Production-facing seams:** request IDs, structured logs, readiness, timings, optional
  tracing, runbooks, ADRs, security guidance, and an explicit ship-status page.

## Honesty boundary

- The published scorecard uses deterministic fake providers. It validates plumbing,
  contracts, provenance, and publication; it does not measure hosted retrieval or answer
  quality.
- Authentication, authorization, rate limiting, a metrics endpoint, production retry and
  circuit-breaker policy, multi-tenancy, and load/concurrency claims are not included.
- There is no hosted demo and no attached binary. The supported demo is the reproducible
  local path documented in the README.

## Start here

- [Try it free](https://github.com/pabloalvarez99/production-rag#try-it-free-0-no-api-key)
- [Ship notes](https://github.com/pabloalvarez99/production-rag/blob/main/docs/SHIP.md)
- [Engineering case study](https://github.com/pabloalvarez99/production-rag/blob/main/docs/CASESTUDY.md)
- [Changelog](https://github.com/pabloalvarez99/production-rag/blob/main/CHANGELOG.md)
