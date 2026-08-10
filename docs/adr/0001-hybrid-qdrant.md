# ADR 0001: Hybrid retrieval in Qdrant (dense + sparse, one collection)

Status: Proposed
Date: 2026-08-10

## Context

Pure dense-vector retrieval misses exact-keyword and out-of-vocabulary
matches (identifiers, error codes, product names); pure lexical retrieval
misses paraphrases and synonyms. We need both signals, and we want the
operational footprint to stay small — M0 runs exactly two containers.

## Decision

Use a single Qdrant collection holding both a named dense vector and a named
sparse (BM25-style) vector per chunk, and run hybrid queries through Qdrant's
query API with fusion (e.g. RRF) server-side. Qdrant is the only vector
store; there is no separate lexical index (no Elasticsearch/OpenSearch).

## Consequences

- One store to operate, back up, and upgrade; payloads and both vector kinds
  stay consistent by construction.
- Fusion happens server-side, so the API does not implement merge logic.
- Sparse-vector support requires Qdrant >= 1.10-ish features, hence the
  pinned `v1.13.2` tag; upgrades go through the runbook procedure.
- Reranking on top of the fused list is left to a later milestone (config
  shape already reserved).
- Trade-off accepted: less tuning control than a dedicated lexical engine.
