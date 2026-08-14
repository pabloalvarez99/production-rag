# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-08-14

### Added

- `POST /v1/query/stream` — additive SSE stream of provisional model deltas ending in the
  same grounded or refused body as `POST /v1/query` (ADR-0012). FakeLLM chunks are a
  deterministic fixture for byte-level tests.
- UI stream toggle: draft is labelled unverified and replaced by the grounded fragment or
  a refusal.
- Optional filter-aware in-process query cache (ADR-0013). Key includes collection, query,
  filters, embedder id, llm id, and retrieval fingerprint. Off in production-shaped
  config; on in the local demo via `CACHE_ENABLED`. `cache=hit|miss` only on the debug
  allowlist.
- Human-readable free-path scorecard at `docs/assets/scorecard.html` and local `GET /evals`
  (contract/plumbing, `billed=false`, `n`, not SOTA).
- Optional OpenTelemetry console exporter behind `PRAG_OTEL_CONSOLE` (default NullTracer;
  CI empty).
- DEMO-DAY filter beat (`title=Filtering`) and stream beat; CASESTUDY section on fail-closed
  filters and why the sample corpus demo uses title rather than `source=sample`.

### Changed

- `generation.stream` now only defaults the demo form toggle; the stream route is always
  mounted.

## [0.1.0] - 2026-08-13

### Added

- Hybrid dense and sparse retrieval in Qdrant, fused with reciprocal rank fusion (RRF).
- Optional cross-encoder reranking with explicit fail-open degradation.
- A LangGraph query path with grounded generation, validated citations, and explicit
  refusal when evidence is insufficient.
- Deterministic offline providers and a credential-free demo covering ingest, API, CLI,
  web UI, and both evaluation tiers.
- Tier 1 retrieval and tier 2 answer evaluation, paired statistics, provenance, and a
  generated scorecard contract.
- Request IDs, structured logs, per-stage timings, readiness checks, and an optional
  tracing seam.
- CI that runs lint, strict type checking, tests, ingestion, evaluation, and an HTTP
  smoke query with provider keys set to empty values.
- Architecture, runbook, evaluation, security, contribution, ship-status, and ADR
  documentation.

### Security

- Provider credentials are optional, excluded from the free path, and empty in CI.
- Authentication, authorization, and rate limiting are not part of this release; do not
  expose the service to an untrusted network.

### Known limitations

- The published local-provider scorecard validates deterministic plumbing and contracts,
  not hosted retrieval or answer quality.
- There is no hosted-provider baseline, production retry or circuit-breaker policy,
  metrics endpoint, multi-tenancy, or load/concurrency claim.

[Unreleased]: https://github.com/pabloalvarez99/production-rag/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/pabloalvarez99/production-rag/releases/tag/v0.2.0
[0.1.0]: https://github.com/pabloalvarez99/production-rag/releases/tag/v0.1.0
