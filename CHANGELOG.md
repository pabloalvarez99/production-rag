# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- An end-to-end [engineering case study](docs/CASESTUDY.md) covering the constraints,
  architecture, trade-offs, failure behaviour, evaluation strategy, and next boundary.
- Copy-ready [GitHub profile README content](docs/PROFILE_README.md) for the five-system
  AI Engineering portfolio series.

### Changed

- Release and security documentation now reflects that `v0.1.0` is published while
  `main` continues to receive documentation improvements.

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

[Unreleased]: https://github.com/pabloalvarez99/production-rag/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/pabloalvarez99/production-rag/releases/tag/v0.1.0
