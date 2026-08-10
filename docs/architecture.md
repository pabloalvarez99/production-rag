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
