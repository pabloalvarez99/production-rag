# Architecture

Status: **M2 (hybrid retrieval) in progress.** Last updated: 2026-08-10.

M0 shipped the walking skeleton: package, config, health and readiness probes,
container stack. M1 added the offline ingest path — walk, chunk, embed (dense),
upsert into Qdrant. M2 adds **retrieval**: sparse/BM25 vectors written alongside
the dense ones at ingest time, a dense branch and a sparse branch queried
together, and reciprocal rank fusion over the two result lists.

What M2 does **not** add: generation. There is no `POST /v1/query`, no LLM call,
no answer, no citation rendering. Retrieval is exercised as a batch command that
prints ranked hits; the HTTP query surface and the generation stages below are
still design. No retrieval quality number quoted anywhere in this repo has been
measured against a semantically meaningful embedding — see
[the fake embedder](#two-embedders-one-path).

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
| `production_rag.ingest` | Offline ingest job: walk, chunk, embed, upsert. New in M1; writes sparse vectors too from M2. | A1 |
| `production_rag.retrieval` | Embedders, Qdrant store, and — new in M2 — the sparse encoder, the hybrid searcher and RRF fusion. | A1 |
| `qdrant` container | Dense and sparse vectors plus chunk payloads in one collection. Pinned to `qdrant/qdrant:v1.13.2`. | A2 |
| `configs/default.yaml` | Declarative runtime config: ingest, retrieval, rerank, generation, qdrant, evals, observability. | A2 |
| `data/` | `raw/` (corpus), `processed/` (derived chunk artifacts, gitignored), `eval/` (golden set). | A2 |
| `scripts/`, `Makefile` | Operator entrypoints: up, down, health, ingest. | A2 |
| `docs/`, ADRs | Architecture, data model, runbook, evaluation. | A2 |

Owners were `K1`/`K2` through M0; the same two seats are `A1`/`A2` from M1.

## Request flow (query path) — partly implemented

> Stages 1–3 (normalise, dense + sparse retrieval, RRF fusion) run as of M2, but
> as a **batch command**, not as an HTTP endpoint. Stages 4–6 (rerank, generate,
> cite) and the `POST /v1/query` surface itself are unbuilt. Treat every stage
> below as design until the roadmap in the README marks its milestone done.

1. Client calls `POST /v1/query` with a natural-language question. *(M4)*
2. The query is embedded (dense) and tokenized (sparse, BM25-style). *(live, M2)*
3. Hybrid retrieval runs against Qdrant: dense vector search fused with
   sparse vector search (see [ADR 0001](adr/0001-hybrid-qdrant.md)). *(live, M2)*
4. Retrieved chunks are optionally reranked (declared in config; not live). *(M3)*
5. A generation call answers with citations to the retrieved chunks. *(M4)*
6. The query pipeline is orchestrated as a LangGraph graph so steps are
   observable and individually testable (see [ADR 0002](adr/0002-langgraph-query.md)). *(M4)*

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

## Retrieval flow (M2) — live

This is the part of the query path that exists today. It is invoked as a batch
command (`python -m production_rag.retrieval`, wrapped by `make retrieve-fake`
and `scripts/retrieve.ps1`), takes a question string, and prints ranked hits.
Nothing HTTP-facing calls it yet.

```
  question text
        │
        ▼
  ┌──────────────────────────┐
  │ 0. NORMALISE             │  trim, reject empty, bind a run id
  └────────────┬─────────────┘
               │ query
    ┌──────────┴────────────────────────────┐
    ▼                                       ▼
┌────────────────────────┐      ┌────────────────────────────┐
│ 1a. DENSE BRANCH       │      │ 1b. SPARSE BRANCH          │
│ embed(query) → vector  │      │ tokenise → weight 1.0 per  │
│ kNN on named vector    │      │ distinct term → query on   │
│ `dense`, cosine        │      │ named vector `sparse`      │
│ dense_top_k = 40       │      │ dot product = BM25 score   │
│                        │      │ sparse_top_k = 40          │
│ wins on paraphrase     │      │ wins on exact tokens       │
└───────────┬────────────┘      └─────────────┬──────────────┘
            │  ranked list A                  │  ranked list B
            └──────────────┬──────────────────┘
                           ▼
            ┌──────────────────────────────────────┐
            │ 2. FUSE — reciprocal rank fusion     │
            │                                      │
            │   score(d) = Σ  w_branch / (k + rank)│
            │            branches                  │
            │                                      │
            │   k = 60, rank is 1-based, a         │
            │   document missing from a branch     │
            │   contributes nothing from it        │
            └──────────────┬───────────────────────┘
                           │ fused, deduplicated by point id
                           ▼
            ┌──────────────────────────────────────┐
            │ 3. CUT — score_threshold, then       │
            │    top_k = 12                        │
            └──────────────┬───────────────────────┘
                           ▼
   hits[]: { score, chunk_id, source_path, title, heading_path, text,
             ranks: {dense: 14, sparse: 1}, contributions: {…} }
```

Each fused hit carries `ranks` — the rank it held in every branch that returned
it — and `contributions`, that branch's share of the fused score. Without them a
hit's provenance is unrecoverable, and "did the sparse branch contribute anything
at all?" is the first question when a hybrid result looks wrong. "This chunk is
second because BM25 ranked it 1st and dense ranked it 14th" is a debuggable
statement; a bare fused score is not.

A hit whose `ranks` holds `sparse` and not `dense` is exactly the class of result
hybrid retrieval was adopted for.

### Fusion is rank-based, so nothing needs calibrating

RRF sums `1/(k + rank)` over the branches that returned a document. Cosine
similarity lives in `[-1, 1]`, BM25 scores are unbounded and corpus-dependent;
adding or averaging them requires a normalisation that has to be re-fitted every
time the corpus grows. Rank position has no scale, so there is nothing to drift.

`k = 60` is the constant from the original RRF paper. It damps the difference
between the top ranks: with `k = 60`, rank 1 scores `1/61` and rank 2 scores
`1/62` — close, deliberately. A branch that is confidently wrong at rank 1 does
not get to dominate.

The price is that RRF discards magnitude entirely. A document that is
overwhelmingly the best match contributes exactly what a merely-good one at the
same rank contributes. Recovering that ordering is what the M3 reranker is for.

### `score_threshold` on a fused score is not a relevance floor

`retrieval.score_threshold` is applied **after** fusion, to the RRF score. That
score is a function of rank positions, not of similarity, so its scale depends on
how many branches returned the document and on `k` — not on how relevant the
document is. With two branches and `k = 60`, the theoretical maximum is
`2/61 ≈ 0.0328`, and a document found by one branch at rank 1 scores `≈ 0.0164`.

Default is `0.0` (disabled) for that reason. Raising it filters by "found by both
branches, high up" rather than by relevance, which is a coarser thing than it
looks. Raise it only with eval evidence, and expect refusals to spike first.

### Migration: M2 needs a collection rebuild

The M1 collection was created with **one** named vector, `dense`
(`create_collection(vectors_config={"dense": …})` in
`production_rag.retrieval.store`). No sparse vector was declared, and no
`sparse_vectors_config` was passed. Earlier drafts of these documents claimed M1
had declared an empty `sparse` vector so that M2 would be a pure backfill; that
was never true of the shipped code, and the claim has been removed everywhere it
appeared.

So M2 is a **migration**, not a backfill:

```bash
make reingest-fake            # ingest --recreate-collection --embedder fake
```

That drops the collection and re-ingests, which re-embeds the whole corpus. On
the `fake` embedder it is free and takes seconds. On the `openai` embedder it is
a full re-embed of every chunk, billed — the incremental content-hash skip cannot
help, because the collection it would have compared against no longer exists.

There is no cheaper path. Qdrant cannot add a named vector to an existing
collection, so this is a migration by construction, not a choice between a
migration and a backfill. `QdrantStore._assert_sparse_declared` checks the live
collection and raises with the exact flag that performs the rebuild and what it
costs, rather than letting the first sparse upsert fail somewhere less legible.

The collection is also created with `sparse` declared **only when sparse weights
will actually be written**. Declaring it empty would make the sparse branch
return nothing on every query — which reads as a ranking bug and takes far
longer to diagnose than a missing named vector, precisely because the collection
looks correct.

Running M2 retrieval against a collection that predates it must fail loudly: the
sparse branch finds no `sparse` named vector, and the contract is to abort with
an explicit "recreate the collection" message rather than silently degrade to
dense-only. A hybrid system that quietly stops being hybrid is the failure mode
worth spending an error message on, and `retrieval.mode: dense` is how you ask
for dense-only on purpose.

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
  │ 5b. ENCODE (sparse)   M2   │  full BM25 weights, IDF folded in,
  │    corpus-wide statistics  │  no provider call, no spend
  └─────────────┬──────────────┘
                ▼
  ┌────────────────────────────┐
  │ 6. UPSERT                  │  named vectors `dense` + `sparse` in one
  │    Qdrant, wait=true       │  point, payload alongside; wait so a smoke
  └─────────────┬──────────────┘  test cannot read a half-built index
                ▼
   report: files, chunks, embedded, skipped, upserted, elapsed
```

Both vectors are written in the same upsert, so a point can never hold one and
not the other. That is the whole reason for keeping them in one collection.

Sparse encoding happens **after** the dense embed and before the upsert, and it
costs nothing per document — it is arithmetic over the corpus, not a provider
call. The incremental content-hash skip therefore still guards the only paid
stage.

Step 5b is new in M2, and it is the reason M1 collections have to be rebuilt:
they were created with `dense` as the only named vector. See
[Migration](#migration-m2-needs-a-collection-rebuild).

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

## What is live in M2 vs declared-only

`configs/default.yaml` is deliberately broader than what the code consumes.

| Config block | State after M2 |
|---|---|
| `ingest` (walk, chunking, embedding, incremental) | live |
| `ingest.sparse` (BM25 k1/b, lowercase, stopwords) | **live** — vectors are written at ingest |
| `qdrant` (collection, dense + sparse vectors, payload indexes, write consistency) | live |
| `retrieval` (mode, top-k per branch, RRF, threshold, payload fields) | **live** — read by the retrieve command |
| `retrieval.filters.allowed_fields` | declared only — the filter allowlist belongs to the HTTP surface (M4) |
| `rerank` | declared only — M3 |
| `generation`, `citations` | declared only — M4 |
| `evals` | thresholds declared; a source-level `hit@k` script exists, the Ragas harness does not — M6 |
| `observability.tracing` | declared only — M5 |

A key existing in that file is not a claim that the runtime reads it.

### Ingest failure behaviour

| Failure | Behaviour | Rationale |
|---|---|---|
| Collection exists without the `sparse` named vector (an M1 collection) | abort, telling the operator to re-run with `--recreate-collection` | writing dense-only points into a collection the retriever expects to be hybrid produces a silent recall hole |
| Unsupported extension under the corpus root | skipped and logged with a count | a silently ignored PDF is indistinguishable from an empty corpus |
| Document yields zero chunks after the minimum-size filter | warned, ingest continues | one malformed file must not abort a corpus run |
| Embedding provider 429 or timeout | bounded retry with backoff, then abort the run | a partial embed batch upserted as if complete is worse than a failed run |
| Qdrant unreachable at upsert | abort non-zero, nothing partially written | ingest is restartable; the content hash makes the retry cheap |
| Collection exists with a different vector size | abort with an explicit message | silently writing 1536-dim vectors into a 768-dim collection fails much later and much less legibly |

### Retrieval failure behaviour (M2)

| Failure | Behaviour | Rationale |
|---|---|---|
| Collection has no `sparse` named vector | abort with the recreate instruction | see [migration](#migration-m2-needs-a-collection-rebuild); degrading to dense-only silently would be worse |
| Collection missing entirely | abort, exit 2 | nothing has been ingested; retrying cannot help |
| Embedding provider 429 on the query embed | bounded retry with backoff, then fail the query | an empty result set is indistinguishable from "no matches" |
| Sparse branch returns nothing (all query terms out of vocabulary) | dense results are returned, and the empty branch is reported | legitimate: an all-stopword query has no lexical signal |
| No hit clears `score_threshold` | empty result set, reported as such | with the default `0.0` this cannot happen; it is a config decision, not a fault |

The asymmetry is deliberate. A *missing capability* (no sparse vector on the
collection) aborts; an *empty result from a working capability* is data.

## Deployment shape

Local development is the only target through M2 and it is `docker compose up -d
--build`. The compose file is written to be promotion-friendly: pinned image
tags, healthchecks on both services, named volume for the vector index, and
secrets injected exclusively via environment (never files baked into images).

## Non-goals (M2)

- No query **endpoint**. Retrieval runs as a batch command; `POST /v1/query` is M4.
- No reranking (M3) and no generation, answers or citations (M4).
- No measured retrieval quality number. The `hit@k` script reports what the
  current corpus and embedder produce; on the `fake` embedder that number is
  plumbing, not quality.
- No authentication/authorization on the API.
- No horizontal scaling, no API gateway.
- No GPU-bound local models in the runtime path.
- Reranker and full observability stack are config-shaped but not wired.
