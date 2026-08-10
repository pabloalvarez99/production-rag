# Architecture

Status: M0 (milestone 0). Last updated: 2026-08-10.

## Overview

production-rag is a retrieval-augmented generation service built as a small,
production-shaped stack: a FastAPI application (`api`) backed by Qdrant as the
single vector + payload store. Everything else the system needs (embedding
model, LLM) is an outbound HTTPS call to a managed provider, so the runtime
footprint stays at two containers.

```
                  ┌──────────────────────────────────────────────┐
                  │ docker compose network (ragnet)              │
                  │                                              │
  client ──8000──▶│  api (FastAPI) ────6333 REST────▶  qdrant    │
                  │     │                            (vectors +  │
                  │     │ 6334 gRPC (ingest)          payloads)  │
                  │     │                                        │
                  └─────┼────────────────────────────────────────┘
                        ▼
              Managed providers (HTTPS):
              embeddings + LLM (e.g. OpenAI)
```

## Components

| Component | Role | Owned by |
|-----------|------|----------|
| `api` container | FastAPI app, `production_rag.main:app`. Serves `/health`, `/v1/*`. | K1 (code), K2 (image/compose) |
| `qdrant` container | Dense + sparse vectors and chunk payloads in one collection. Pinned to `qdrant/qdrant:v1.13.2`. | K2 |
| `configs/default.yaml` | Declarative runtime config: ingest, retrieval, rerank, generation, qdrant, evals, observability. | K2 |
| `data/` | `raw/` (corpus), `processed/` (derived chunk artifacts, gitignored), `eval/` (eval datasets). | K2 |

## Request flow (query path)

1. Client calls `POST /v1/query` with a natural-language question.
2. The query is embedded (dense) and tokenized (sparse, BM25-style).
3. Hybrid retrieval runs against Qdrant: dense vector search fused with
   sparse vector search (see [ADR 0001](adr/0001-hybrid-qdrant.md)).
4. Retrieved chunks are optionally reranked (declared in config; not live
   in M0).
5. A generation call answers with citations to the retrieved chunks.
6. The query pipeline is orchestrated as a LangGraph graph so steps are
   observable and individually testable (see [ADR 0002](adr/0002-langgraph-query.md)).

### Stage pipeline

```
  POST /v1/query
  { "question": … }
        │
        ▼
  ┌───────────────┐
  │ 1. NORMALISE  │  validate, attach request id, optional query rewrite
  └───────┬───────┘
          │ query text
   ┌──────┴──────────────────────────┐
   ▼                                 ▼
┌──────────────────────┐   ┌──────────────────────┐
│ 2a. DENSE RETRIEVAL  │   │ 2b. SPARSE RETRIEVAL │
│ embed(query) → kNN   │   │ BM25 over term vecs  │
│ Qdrant `dense`       │   │ Qdrant `sparse`      │
│ 40 candidates        │   │ 40 candidates        │
│ wins on paraphrase   │   │ wins on exact tokens │
└──────────┬───────────┘   └───────────┬──────────┘
           └───────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │ 3. FUSE — reciprocal rank    │  score = Σ 1/(k + rank), k = 60
        │    fusion, rank-based so the │  no score-scale calibration,
        │    two scales never need     │  nothing to drift as the corpus
        │    calibrating → top 12      │  grows
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │ 4. RERANK — cross-encoder    │  disabled in M0; fail-open, so a
        │    40 in → 6 out             │  reranker error degrades to fusion
        └──────────────┬───────────────┘  order instead of failing the call
                       ▼
        ┌──────────────────────────────┐
        │ 5. GENERATE                  │  6k-token context budget,
        │    temperature 0.1, stream   │  refuses when evidence is absent
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │ 6. CITE                      │  map [n] markers back to
        │                              │  chunk_id + source_path
        └──────────────┬───────────────┘
                       ▼
   { answer, citations[], usage, trace_id, latency_ms }
```

Each numbered stage is one LangGraph node with its own timing, which is what
populates the per-stage `latency_ms` breakdown in the response — the first
thing the [runbook](runbook.md) tells an operator to read when a query is slow.

### Why hybrid rather than dense-only

Dense retrieval fails predictably on rare literal tokens (SKUs, error codes,
function names); sparse retrieval fails just as predictably on paraphrase.
Running both costs one extra Qdrant query on the same round trip and removes an
entire class of "the document is right there and it didn't find it" bugs. Rank
fusion is used instead of weighted score blending because cosine similarity and
BM25 scores live on incomparable scales. See
[ADR 0001](adr/0001-hybrid-qdrant.md).

### Failure behaviour

| Failure | Behaviour | Rationale |
|---|---|---|
| Qdrant unreachable | `/v1/ready` reports not ready; `/health` still 200 | liveness must not depend on a dependency, or an orchestrator kills a healthy process |
| Reranker error or timeout | fall through to fusion order | availability over a few points of nDCG |
| Embedding provider 429 | bounded retry with backoff, then 503 | a silent empty result set is indistinguishable from "no matches" |
| No chunk clears the threshold | explicit refusal | an unsupported answer is worse than no answer |

## Ingest flow (offline path)

1. Markdown/text files under `data/raw/` are walked per
   `configs/default.yaml → ingest.include_extensions`.
2. Documents are split into chunks (recursive structural splitting).
3. Chunks are embedded (dense) and indexed (sparse) and upserted into the
   Qdrant collection with their payload (source path, heading, chunk index).
4. Derived artifacts may be cached under `data/processed/` (never committed,
   never baked into the image).

## What is live in M0 vs declared-only

`configs/default.yaml` is deliberately broader than what M0 consumes. Live in
M0: ingest, retrieval (hybrid), generation, qdrant. Declared for later
milestones (shape fixed, values may change): rerank, evals, observability.

## Deployment shape

Local development is the only target in M0 and it is `docker compose up -d
--build`. The compose file is written to be promotion-friendly: pinned image
tags, healthchecks on both services, named volume for the vector index, and
secrets injected exclusively via environment (never files baked into images).

## Non-goals (M0)

- No authentication/authorization on the API.
- No horizontal scaling, no API gateway.
- No GPU-bound local models in the runtime path.
- Reranker and full observability stack are config-shaped but not wired.
