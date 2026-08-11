# Architecture

Status: **M1 (ingest) in progress.** Last updated: 2026-08-10.

M0 shipped the walking skeleton: package, config, health and readiness probes,
container stack. M1 adds the offline ingest path only — walk, chunk, embed
(dense), upsert into Qdrant. The query path in this document is design, not
running code: nothing serves `POST /v1/query` yet, hybrid retrieval is not
wired, and no retrieval number quoted anywhere in this repo has been measured.

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
| `api` container | FastAPI app, `production_rag.main:app`. Serves `/health`, `/v1/*`. | A1 (code), A2 (image/compose) |
| `production_rag.ingest` | Offline ingest job: walk, chunk, embed, upsert. New in M1. | A1 |
| `qdrant` container | Dense (M1) and sparse (M2) vectors plus chunk payloads in one collection. Pinned to `qdrant/qdrant:v1.13.2`. | A2 |
| `configs/default.yaml` | Declarative runtime config: ingest, retrieval, rerank, generation, qdrant, evals, observability. | A2 |
| `data/` | `raw/` (corpus), `processed/` (derived chunk artifacts, gitignored), `eval/` (golden set). | A2 |
| `scripts/`, `Makefile` | Operator entrypoints: up, down, health, ingest. | A2 |
| `docs/`, ADRs | Architecture, data model, runbook, evaluation. | A2 |

Owners were `K1`/`K2` through M0; the same two seats are `A1`/`A2` from M1.

## Request flow (query path) — design only, not implemented

> Nothing in this section runs as of M1. It is the agreed target shape, kept
> here so that M2–M4 are implementation work rather than an architecture debate.
> Treat every stage below as unbuilt until the roadmap in the README marks its
> milestone done.

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

## Ingest flow (offline path) — M1

The ingest job is a batch process, not an endpoint. It is invoked from the
operator surface (`make ingest-fake`, `scripts/ingest.ps1`), runs to completion,
and reports counts. Nothing in the request path calls it.

```
  data/raw/  ← the corpus root; source_path is relative to it
        │
        ▼
  ┌────────────────────────────┐
  │ 1. WALK                    │  include_extensions allowlist, exclude_globs
  │    files → documents       │  a skipped file is logged, never silent
  └─────────────┬──────────────┘
                │ doc: source_path, title, tags, body
                ▼
  ┌────────────────────────────┐
  │ 2. PARSE                   │  YAML front matter → title/tags,
  │    front matter + headings │  else first H1 is the title
  └─────────────┬──────────────┘
                ▼
  ┌────────────────────────────┐
  │ 3. CHUNK                   │  recursive: "\n## " → "\n\n" → "\n" → ". "
  │    800 chars, 120 overlap  │  fragments < 120 chars dropped and counted
  └─────────────┬──────────────┘
                │ chunk: text, heading_path, chunk_index
                ▼
  ┌────────────────────────────┐
  │ 4. HASH                    │  sha256 over chunk text; unchanged hash
  │    incremental skip        │  skips step 5, the only paid step
  └─────────────┬──────────────┘
                ▼
  ┌────────────────────────────┐
  │ 5. EMBED (dense)           │  fake  → deterministic hash vectors, offline
  │    batch 128               │  openai → text-embedding-3-small, 1536 dims
  └─────────────┬──────────────┘
                │
                ▼
  ┌────────────────────────────┐
  │ 6. UPSERT                  │  named vector `dense`, payload alongside,
  │    Qdrant, wait=true       │  wait so a smoke test cannot read a
  └─────────────┬──────────────┘  half-built index
                ▼
   report: files, chunks, embedded, skipped, upserted, elapsed
```

Sparse vectors are **not** produced in M1. The collection is created with the
`sparse` named vector declared but unpopulated, so M2 adds a backfill rather
than a collection migration.

### Two embedders, one path

`--embedder fake` is a deterministic hash embedder: it maps text to a vector by
hashing, so the same text always yields the same vector, no API key is needed,
and nothing leaves the machine. It exists so the entire ingest path is
exercisable in CI and on a laptop with no credentials — a corpus can be walked,
chunked, upserted and counted for free.

It is worthless for retrieval quality, and that is deliberate. Its vectors carry
no semantics, so any similarity number measured against a fake-embedded
collection is noise. Use it to test plumbing; never to make a claim.

`--embedder openai` reads `OPENAI_API_KEY` from the environment. Ingest is the
only stage in the system that spends money per document, which is why the
content hash and the incremental skip exist before the embed call rather than
after it.

## What is live in M1 vs declared-only

`configs/default.yaml` is deliberately broader than what the code consumes.

| Config block | State after M1 |
|---|---|
| `ingest` (walk, chunking, embedding, incremental) | live |
| `qdrant` (collection, dense vector, payload indexes, write consistency) | live |
| `ingest.sparse` | declared; BM25 vectors land in M2 |
| `retrieval` (hybrid, fusion, thresholds) | declared only — no query path exists |
| `rerank` | declared only — M3 |
| `generation`, `citations` | declared only — M4 |
| `evals` | declared; the golden seed set exists, the harness does not |
| `observability.tracing` | declared only — M5 |

A key existing in that file is not a claim that the runtime reads it.

### Ingest failure behaviour

| Failure | Behaviour | Rationale |
|---|---|---|
| Unsupported extension under the corpus root | skipped and logged with a count | a silently ignored PDF is indistinguishable from an empty corpus |
| Document yields zero chunks after the minimum-size filter | warned, ingest continues | one malformed file must not abort a corpus run |
| Embedding provider 429 or timeout | bounded retry with backoff, then abort the run | a partial embed batch upserted as if complete is worse than a failed run |
| Qdrant unreachable at upsert | abort non-zero, nothing partially written | ingest is restartable; the content hash makes the retry cheap |
| Collection exists with a different vector size | abort with an explicit message | silently writing 1536-dim vectors into a 768-dim collection fails much later and much less legibly |

## Deployment shape

Local development is the only target through M1 and it is `docker compose up -d
--build`. The compose file is written to be promotion-friendly: pinned image
tags, healthchecks on both services, named volume for the vector index, and
secrets injected exclusively via environment (never files baked into images).

## Non-goals (M1)

- No query endpoint. Retrieval, reranking and generation are all unbuilt.
- No sparse/BM25 vectors yet; the named vector is declared, not populated.
- No authentication/authorization on the API.
- No horizontal scaling, no API gateway.
- No GPU-bound local models in the runtime path.
- Reranker and full observability stack are config-shaped but not wired.
