# Architecture

Status: **M6 (evaluation).** Last updated: 2026-08-11.

M0 shipped the walking skeleton: package, config, health and readiness probes,
container stack. M1 added the offline ingest path — walk, chunk, embed (dense),
upsert into Qdrant. M2 added **retrieval**: sparse/BM25 vectors written alongside
the dense ones at ingest time, a dense branch and a sparse branch queried
together, and reciprocal rank fusion over the two result lists. M3 added
**reranking**: a cross-encoder pass over the fused candidates, opt-in, fail-open,
with a deterministic offline provider so the stage stays runnable with no
credentials.

M4 closes the path. The stages above are now orchestrated as a **LangGraph
graph** ([ADR 0002](adr/0002-langgraph-query.md), Accepted) behind
**`POST /v1/query`**, and the graph ends in **grounded generation**: an answer
whose every claim carries an inline `[n]` marker resolved back to a retrieved
chunk, or — when no chunk clears the evidence bar — an explicit **refusal with
no LLM call at all** ([ADR 0005](adr/0005-grounded-generation.md)). Retrieval
returns passages; M4 is the first milestone that returns prose, which is why the
citation contract and the refusal edge are part of the design rather than a
prompt detail.

M5 adds no stage. It makes the stages that exist **legible**: per-node timings
that were already on the graph state become the operator's first read, the
caller-facing `debug` flag gets a defined and deliberately narrow contract, and
tracing becomes an opt-in export that the system is complete without
([ADR 0006](adr/0006-observability.md)). The rule that shapes all of it is that
the observability layer is the most likely place in this system for corpus text
to leak, because copying internal state elsewhere is its entire job — so prompts
and passages are never a signal, at any level. See
[Observability](#observability-m5).

M6 adds **offline evaluation**, both tiers, behind one runner
(`production_rag.evals.run`): retrieval metrics over the golden set and
answer-side metrics over the real `run_query` path, in one process and one
versioned report. Its defaults are offline — fake embedder, fake model, fake
judge — so the whole thing is free, deterministic and CI-runnable, and the report
states `offline_defaults` so nobody mistakes a plumbing check for a quality
number. See [Evaluation](#evaluation-m6) and
[ADR 0003](adr/0003-eval-strategy.md).

Ownership note: this document describes the contract the M4 query path
implements. The library (`production_rag.generation`, the graph) is A1's; the
HTTP surface is A3's; this file, the runbook, the data model and the ADRs are
A2's.

What M6 does **not** add: a defensible *quality* number, or an armed merge gate.
The judge is uncalibrated and offline by default, `citation_precision` checks the
cited document rather than the cited passage, and the only gate is an opt-in flag
that defaults to reporting. Nor has any retrieval number here been measured
against a semantically meaningful embedding: see
[the fake embedder](#two-embedders-one-path), [the fake reranker](#three-rerank-providers-one-interface)
and, one stage later again, [the fake generator](#two-generation-providers-one-contract).

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
| `production_rag.retrieval.rerank` | New in M3: the cross-encoder rerank stage and its three providers (`fake`, `local`, `cohere`), plus the fail-open wrapper. | A1 |
| `production_rag.generation` | New in M4: prompt assembly under a context budget, the LLM providers (`fake`, `openai`), `[n]` marker resolution into `Citation` objects, and the refusal path. | A1 |
| `production_rag.graph` | New in M4: the LangGraph `StateGraph` that wires normalise → retrieve → fuse → rerank → generate → cite, the rerank bypass edge and the refusal edge. Nodes are adapters; no business logic. | A1 |
| `production_rag.query_pipeline` | New in M4: the one public entry point — `run_query()` for a one-shot call, `QueryPipeline` for a process that serves many. Everything under `graph/` and `generation/` is an implementation detail of it. | A1 |
| `production_rag.evals` | New in M6: the offline two-tier evaluation. `run` is the unified entry point; `tier1_retrieval` derives source-level `hit`/`recall`/`mrr`/`ndcg` from `source_hit`; `tier2_answer` scores answers from the real `run_query` path; `judges` supplies the optional `faithfulness`/`relevance` columns; `ablation` compares branches. | A1 |
| `production_rag.query` | New in M4: the batch CLI over that entry point, with the same stdout-JSON and graded-exit-code contract as ingest and retrieve. | A3 |
| `POST /v1/query` | New in M4: the HTTP surface — request validation, provider selection, correlation id, and a narrow response projection. | A3 |
| `qdrant` container | Dense and sparse vectors plus chunk payloads in one collection. Pinned to `qdrant/qdrant:v1.13.2`. | A2 |
| `configs/default.yaml` | Declarative runtime config: ingest, retrieval, rerank, generation, qdrant, evals, observability. | A2 |
| `data/` | `raw/` (corpus), `processed/` (derived chunk artifacts, gitignored), `eval/` (golden set). | A2 |
| `scripts/`, `Makefile` | Operator entrypoints: up, down, health, ingest. | A2 |
| `docs/`, ADRs | Architecture, data model, runbook, evaluation. | A2 |

Owners were `K1`/`K2` through M0; the same two seats are `A1`/`A2` from M1.

## Request flow (query path) — live as of M4

1. Client calls `POST /v1/query` with a natural-language question. *(live, M4)*
2. The query is embedded (dense) and tokenized (sparse, BM25-style). *(live, M2)*
3. Hybrid retrieval runs against Qdrant: dense vector search fused with
   sparse vector search (see [ADR 0001](adr/0001-hybrid-qdrant.md)). *(live, M2)*
4. Fused candidates are optionally reranked by a cross-encoder, fail-open (see
   [ADR 0004](adr/0004-rerank-cross-encoder.md)). *(live, M3 — opt-in)*
5. A generation call answers with inline `[n]` citations to the retrieved
   chunks, or the request refuses without calling the model when nothing clears
   the evidence bar (see [ADR 0005](adr/0005-grounded-generation.md)). *(live, M4)*
6. The pipeline is orchestrated as a LangGraph graph so stages are observable
   and individually testable (see [ADR 0002](adr/0002-langgraph-query.md)). *(live, M4)*

The batch retrieve command from M2/M3 has not gone anywhere: it is still the way
to inspect ranked passages without spending a generation call, and it is still
the surface the eval script drives.

### The query graph (M4)

One node per stage, one typed state object threaded through, and two conditional
edges. The edges are the whole reason this is a graph rather than a function
chain — a bypass when reranking does not happen, and a refusal that leaves the
graph before the LLM is ever constructed.

```
                    POST /v1/query  { question, mode?, rerank?, llm?, debug? }
                              │
                              ▼
                    ┌───────────────────────┐
                    │ normalise             │  trim, reject empty, bind
                    │                       │  request id into the log context
                    └──────────┬────────────┘
                               ▼
                    ┌───────────────────────┐
                    │ retrieve              │  dense + sparse branches, M2
                    │                       │  mode override honoured here
                    └──────────┬────────────┘
                               ▼
                    ┌───────────────────────┐
                    │ fuse                  │  RRF, k = 60, rank-based
                    └──────────┬────────────┘
                               │
                 rerank off ───┴─── rerank on
                  (bypass edge)      │
                        │            ▼
                        │  ┌───────────────────────┐
                        │  │ rerank                │  cross-encoder, M3
                        │  │                       │  fail-open → bypass edge
                        │  └──────────┬────────────┘
                        └──────┬──────┘
                               ▼
                    ┌───────────────────────┐
                    │ gate — is there        │  evidence check, no LLM yet
                    │ supporting evidence?   │
                    └───┬───────────────┬────┘
                        │ no            │ yes
        (refusal edge)  │               ▼
                        │    ┌───────────────────────┐
                        │    │ generate              │  budgeted context,
                        │    │                       │  temperature 0.1,
                        │    │                       │  [n] markers required
                        │    └──────────┬────────────┘
                        │               ▼
                        │    ┌───────────────────────┐
                        │    │ cite                  │  resolve [n] → chunk;
                        │    │                       │  strip + record the rest
                        │    └──────────┬────────────┘
                        │               │
                        │        nothing citable? ──┐  (second refusal check)
                        ▼               ▼           │
              ┌─────────────────────────────────────┴─┐
              │ respond                               │
              └──────────────────┬────────────────────┘
                                 ▼
   HTTP  { answer, refused, refusal_reason, citations[] }
   lib   + hits_used, hits_retrieved, model, mode, collection, embedded_model,
           rerank{…}, invalid_markers[], uncited_claims[], latency_ms{per node}
```

The refusal edge is the design point worth reading twice: **it skips the
generate node entirely.** The model is not asked to decide whether it has
evidence — the system decides, before spending a token. See
[abstention](#abstention--the-refusal-edge-never-reaches-the-model).

There are **two** refusal points, not one: before the call (no evidence to
ground anything in) and after it (nothing in the answer resolved to a chunk, or
the model emitted the abstention sentinel). Both produce a code from a closed
set, which is what lets an operator alert on one and an eval group by it.

Node timings are fields on the state object, which is what populates the
per-node `latency_ms` breakdown on the library result — the first thing the
[runbook](runbook.md) tells an operator to read when a query is slow.

The HTTP response is deliberately a **narrower projection** of that result:
answer, citations, `refused`, `refusal_reason`, and nothing else. Internal
counts, timings, the collection name and the embedding model stay on the library
result and in the logs, where they are keyed by the request id. A public
endpoint that reports its own collection name and stage latencies is describing
its interior to anyone who asks.

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
        │ 4. RERANK — cross-encoder    │  opt-in; fail-open, so a reranker
        │    40 in → 6 out             │  error degrades to fusion order
        └──────────────┬───────────────┘  instead of failing the call
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
   { answer, refused, refusal_reason, citations[] }
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

## Retrieval flow (M2 + M3) — live

This is the part of the query path that exists today. It is invoked as a batch
command (`python -m production_rag.retrieve`, wrapped by `make retrieve-fake`
and `scripts/retrieve.ps1`), takes a question string, and prints ranked hits.
Nothing HTTP-facing calls it yet.

Stages 0–3 are M2 and always run. Stage 4, rerank, is M3 and runs only when it
is switched on (`rerank.enabled`, or `--rerank` on the command).

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
            │    top_k = 12  (input_top_k = 40     │
            │    when rerank is on)                │
            └──────────────┬───────────────────────┘
                           │
              rerank off ──┴── rerank on (M3, opt-in)
                    │                │
                    │                ▼
                    │  ┌──────────────────────────────────────┐
                    │  │ 4. RERANK — cross-encoder scores     │
                    │  │    each (query, passage) pair with   │
                    │  │    full attention across both.       │
                    │  │    40 candidates in -> 6 kept.       │
                    │  │    fake | local | cohere.            │
                    │  │    On error: fusion order, reported. │
                    │  └──────────────┬───────────────────────┘
                    └────────┬────────┘
                             ▼
   hits[]: { rank, score, chunk_id, source_path, title, heading_path, text,
             branch_ranks: {dense: 14, sparse: 1}, branch_scores: {…},
             pre_rerank_rank: 4, rerank_score: 0.87 }   ← last two only when
                                                          rerank ran
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

### Why rerank runs after RRF and not instead of it

RRF orders by rank position, so it never has to calibrate cosine against BM25 —
and never sees magnitude. Both branches also score a passage *without ever
reading it next to the query*: the dense branch compares two independently
produced vectors, the sparse branch sums per-term weights. The predictable result
is a top-12 that is uniformly on-topic with the actual answer at position four.

A cross-encoder closes exactly that gap: one model, one forward pass per
`(query, passage)` pair, attention across both texts at once. It is accurate
precisely because it cannot be precomputed — there is no passage vector to index,
the query has to be present — which is why it is a reranking stage over a short
list rather than a retriever.

So the two stages own different metrics. **Retrieval owns recall** (is the
supporting chunk in the candidate list at all?); **rerank owns precision at the
top** (is it first?). Rerank cannot fix a recall failure: it never queries
Qdrant and never introduces a document fusion did not return.

### `input_top_k` (40) is larger than `top_k` (6) on purpose

The reranker is fed more candidates than it keeps. Handing it exactly the number
that survives makes it a no-op sorter of an already-final list; handing it 40 is
what lets it lift a passage fusion buried at rank 30 into position 2 — the case
the stage was added for.

Cost is linear in `input_top_k`: 40 candidates is 40 forward passes locally, or
40 passages on the wire for a hosted provider. It is the stage's price dial.
Raising it above what fusion returned (`dense_top_k + sparse_top_k`) buys
nothing — fusion can only pass on what the branches produced.

The flip side is a hard ceiling: with rerank on, a relevant chunk outside
`input_top_k` is unreachable by construction. The reported counts therefore
include how many candidates the stage actually saw, so "was it even a candidate?"
is answerable before "was the ranking wrong?".

### Three rerank providers, one interface

| `--rerank` | `rerank.provider` | Model | Needs | What it measures |
|---|---|---|---|---|
| `fake` | `fake` | none — query-term overlap, pure Python | nothing: no key, no download, no network | **nothing.** Plumbing only |
| `local` | `local-cross-encoder` | `BAAI/bge-reranker-base` | `sentence-transformers` (ships in the `rag` extra), ~1.1 GB model download, CPU | real relevance, no per-query spend, corpus stays local |
| `cohere` | `cohere` | `rerank-english-v3.0` | the `cohere` package, `COHERE_API_KEY`, HTTPS per query | real relevance, hosted, billed per search |

Both real providers import lazily, so `--rerank off` and `--rerank fake` never
pay for torch or a Cohere client. A missing package fails with a message naming
what to install, not an `ImportError` from three frames down.

`--rerank` also takes `off` (default: M2 behaviour) and `auto` (read
`rerank.enabled` and `rerank.provider` from the YAML). The flag switches one run;
`auto` plus the config file switches a deployment.

`fake` is the rerank stage's counterpart to the fake embedder, with the same
warning attached. It scores a candidate by the share of distinct query terms its
text contains — deterministic, so tests can assert on it — which is a cruder
version of what BM25 already did one stage earlier. The flag path, the candidate
arithmetic, the `fail_open` branch, the emitted hit fields and the JSON contract
are all genuinely exercised under it. Its *ordering* is not a quality signal and
no number measured with it is reported as one.

Stated plainly: **rerank plumbing is live everywhere; rerank quality is live only
on `local` or `cohere`.**

`local` is the project's default choice — a CPU cross-encoder makes reranking a
fixed infrastructure cost instead of a per-query bill, and no passage leaves the
machine. `cohere` is a supported swap for deployments that cannot host a model:
same interface, different latency and cost profile. See
[ADR 0004](adr/0004-rerank-cross-encoder.md).

### `fail_open` — the reranker degrades ordering, never availability

If the reranker raises, times out, or returns something malformed, the stage logs
it and returns **fusion order**, unchanged; the query succeeds. Availability beats
a few points of nDCG, because the un-reranked result is still correct in kind —
it is merely ordered worse.

That is deliberately the opposite of how a *missing capability* is treated: a
collection with no `sparse` named vector aborts, because the system would
otherwise be silently unhybrid. A missing improvement is not a missing capability.

The degradation is never silent. Every result carries a `rerank` object —
`{applied, reranker, candidates, error}` — present even when nothing reranked,
and a failure is logged at warning level. So "ordering got worse last Tuesday" is
answerable from the response and the logs instead of from a bisect. A reranked
hit additionally carries `pre_rerank_rank` and `rerank_score`, which is what
makes "the cross-encoder pulled this from rank 27 to rank 2" a measurable
statement rather than an impression.

`fail_open: false` exists for a deployment that would
rather fail the request than serve un-reranked hits; choosing it means accepting
that a provider outage is an outage.

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

## Generation flow (M4) — live

Everything above produces *passages*. This is the stage that produces *prose*,
and it is the first place the system can be confidently wrong. The decision and
its alternatives are in [ADR 0005](adr/0005-grounded-generation.md); the
mechanics are here.

```
  ranked hits (rerank order if it ran, else fusion order)
        │
        ▼
  ┌──────────────────────────────────────────────┐
  │ 1. BUDGET — take hits in order until either  │  max_chunks_in_prompt = 8
  │    max_chunks_in_prompt or                   │  max_context_tokens = 6000
  │    max_context_tokens is hit; the tail is    │  retrieval order IS
  │    dropped and the count reported            │  truncation order
  └───────────────────┬──────────────────────────┘
                      ▼
  ┌──────────────────────────────────────────────┐
  │ 2. RENDER — number the survivors [1]…[n] in  │  heading_path included
  │    that same order, prepend the system       │  when configured; nothing
  │    prompt from configs/prompts/system.md     │  else from the payload
  └───────────────────┬──────────────────────────┘
                      ▼
  ┌──────────────────────────────────────────────┐
  │ 3. GENERATE — temperature 0.1, bounded       │  fake | openai
  │    output, bounded retries, timeout          │  extraction, not authoring
  └───────────────────┬──────────────────────────┘
                      ▼
  ┌──────────────────────────────────────────────┐
  │ 4. RESOLVE — each [n] in the answer maps to  │  out-of-range marker:
  │    the nth PROMPT BLOCK → Citation objects,  │  stripped from the text,
  │    in first-appearance order                 │  recorded in
  └───────────────────┬──────────────────────────┘  invalid_markers
                      ▼
  ┌──────────────────────────────────────────────┐
  │ 5. CHECK — nothing citable, or the abstain   │  refusal_reason:
  │    sentinel, or an empty answer → refuse     │  no_citations |
  └───────────────────┬──────────────────────────┘  model_abstained |
                      ▼                             empty_answer
   { answer, refused, refusal_reason, citations[…] }
```

### The `[n]` citation contract

The number in `[3]` is an **ordinal into the context this request assembled** —
the third passage the prompt contained — and nothing else. Not a chunk id, not a
rank in the collection, not a stable identifier across requests. Two consequences
follow, and both are load-bearing:

- **Resolution is a lookup, not a heuristic.** The request knows exactly which
  chunks it sent and in what order, so `[3]` resolves deterministically to one
  `chunk_id`. No similarity matching between a generated sentence and a passage
  is involved anywhere — that would attribute fluency rather than provenance.
- **Markers are meaningless outside their response.** `[3]` from yesterday's
  answer does not identify anything today. What is durable is the resolved
  `Citation`: `chunk_id`, `source_path`, `title`, `heading_path`, and the quoted
  text. Clients store citations, never markers.

Resolution runs against the **numbered blocks of the rendered prompt**, not
against the retrieval result. Those two lists differ whenever the context budget
truncated, and mapping against the longer one would shift every marker by however
many chunks did not fit — silently, and in the direction that makes citations
look fine.

Each resolved marker becomes one `Citation` in `citations[]`, ordered by first
appearance in the answer, deduplicated — a passage cited three times appears
once. The exact field list is in the [data model](data-model.md#citation).

**Out-of-range markers are stripped from the text and recorded.** If six blocks
were sent and the model writes `[7]`, the marker is removed from the answer and
listed in `invalid_markers` on the library result. Leaving it in would show a
reader a footnote that goes nowhere, which reads as *more* grounded than an
uncited sentence rather than less. Recording it is what makes "this model invents
citations" a measurable claim rather than an impression.

**Surviving markers are not renumbered.** An answer that cites only `[3]` keeps
`[3]` in its text and carries marker 3 in its citation list. Compacting to `[1]`
would make the answer stop matching the prompt that produced it, and lining those
two up is where every debugging session on this path starts.

`uncited_claims` reports the other direction: sentences long enough to be a claim
(24 characters or more) carrying no marker at all. **Reported, never fatal** —
refusing an answer because one transition sentence lacks a marker produces a
guardrail with a false-positive rate high enough that someone switches it off,
which leaves the system with none. It is citation coverage measured on every
request rather than sampled by an offline judge.

A `require_citation: true` deployment does treat an answer with **no** resolvable
citation at all as unsupported: it is refused rather than served. An entirely
uncited answer from a grounded system is either the model ignoring its context or
a claim the context does not support, and neither is worth serving.

### Abstention — the refusal edge never reaches the model

When no retrieved chunk clears the evidence bar, the graph takes the refusal
edge: **the generate node is not entered and no provider call is made.** The
response is the configured `refusal_message` with `refused: true`,
`citations: []`, and a `refusal_reason` from a closed set.

The design point is *who decides*. Asking the model to refuse when appropriate
delegates the judgement to the component whose documented failure mode is
exactly that judgement — a fluent, plausible answer built from parametric memory
and the topical vocabulary of whatever passages it was handed. Deciding in the
graph makes the refusal deterministic, free, and testable without a provider.

There are two checks, and only the first one saves a call:

| Situation | `refusal_reason` | LLM called |
|---|---|---|
| Retrieval returns nothing at all | `no_evidence` | no |
| Every hit is filtered out by `retrieval.score_threshold` | `no_evidence` | no |
| The model emits the `INSUFFICIENT_CONTEXT` sentinel, anywhere in its text | `model_abstained` | yes |
| Nothing in the answer resolves to a block, with `require_citation: true` | `no_citations` | yes |
| The model returns only whitespace | `empty_answer` | yes |
| Hits exist and the answer carries at least one resolvable marker | — | yes, answer served |
| Some markers valid, some out of range | — | yes, answer served; invalid ones stripped and recorded |

The sentinel check matches the token **anywhere** in the text, not only as the
whole message: models routinely wrap a sentinel in a polite sentence, and
treating that as an answer serves the user a refusal dressed as a result.

The reason codes are a closed set (`no_evidence`, `model_abstained`,
`no_citations`, `empty_answer`), which is what lets an operator alert on one and
an eval group by them. The user-facing message is config and can change without
invalidating a dashboard — clients branch on `refused` and `refusal_reason`,
never on the message string.

`refuse_without_evidence: false` disables only the *pre-call* check. It is an
escape hatch for a deployment that wants a general-knowledge fallback, and taking
it gives up the central promise of the system, which is why it is on by default.

A refusal is a **200 with `refused: true`**, not an error status. Nothing failed:
the system was asked something the corpus does not cover and said so. Encoding
that as 4xx/5xx would make a correct outcome indistinguishable from an outage in
every dashboard the service ever gets.

The cost of this design is stated plainly in
[ADR 0005](adr/0005-grounded-generation.md): a question whose supporting chunk
was never retrieved now yields a refusal, so **answer-path recall is capped by
retrieval recall**. That is the correct failure — but it means a spike in
refusals is a retrieval investigation, not a prompt investigation.

### Two generation providers, one contract

| `--llm` / the `llm` request field | Model | Needs | What it measures |
|---|---|---|---|
| `fake` *(default)* | none — deterministic extractive stitching over the supplied passages, pure Python | nothing: no key, no network, no spend | **nothing about answer quality.** The contract only |
| `openai` | `generation.model` (`gpt-4o-mini`) | `OPENAI_API_KEY`, HTTPS per query, billed per token | real answer quality on this corpus |

The default is `fake` on **both** the endpoint and the CLI, deliberately: a caller
that forgets to choose gets the offline path rather than a bill. The corollary is
that an answer is plumbing unless someone asked for `openai` — a demo that forgets
the flag is demonstrating the schema.

Retries, timeouts and status-code classification on the hosted path are the
OpenAI SDK's job (`max_retries`, `timeout_seconds` are handed to the client
rather than reimplemented around it); the SDK already knows which failures are
worth repeating.

`fake` is the third instance of the same pattern in this repository, and it
carries the same warning as the fake embedder and the fake reranker. It composes
an answer from the top passages and emits `[n]` markers that resolve, so the
whole contract is genuinely exercised offline: budgeting, prompt rendering,
marker resolution, invalid-marker stripping, both refusal checks, the response
schema, the per-node timings, the HTTP surface. What it does not do is *reason*. It
cannot synthesise across two passages, it cannot decline a question its passages
technically mention, and its prose is not prose anyone should read.

Stated plainly: **the generation contract is live everywhere; answer quality
exists only on `openai`.** No answer produced by `fake` is evidence of anything
except that the wiring holds, and no faithfulness claim appears in this
repository at all — that needs the M6 judge (see [evaluation](evaluation.md)).

### Context budget: retrieval order is truncation order

`generation.max_chunks_in_prompt` (8) and `generation.max_context_tokens` (6000)
both cut from the **tail**, in the order the retrieval path produced. Whichever
binds first wins. Two things follow:

- The reranker's job is now doubly load-bearing. It does not merely order what
  the model reads; it decides what the model reads *at all*, because a passage
  pushed past the budget is not in the prompt. A rerank that ran on a 40-hit
  candidate window and kept 6 has already made the truncation decision.
- `max_chunks_in_prompt` (8) sits deliberately above `rerank.top_k` (6) so the
  rerank cut is the binding one when the stage is on, and the budget is the
  binding one when it is off (fusion returns `retrieval.top_k`, 12).

Token counts use the same `len(text) / 4` estimate the ingest payload carries;
no tokeniser is loaded. It is a budget guard, not an accounting figure — a few
percent off costs a little unused headroom, while importing a tokeniser to be
exact costs a dependency and a model download.

One block always survives. The budget check is skipped for the first block, so a
single oversized chunk is sent rather than producing an empty context and a
refusal that blames retrieval for a chunking decision. Truncation is logged
(`context_truncated`, with kept and dropped counts) and the library result
carries `hits_used` alongside `hits_retrieved`.

### Request id — one id, every stage, every log line

The correlation id bound by the middleware (`X-Request-ID`, replaced with a
fresh UUID4 when absent or malformed) is bound into the structlog contextvars at
the `normalise` node and is therefore on **every** line the request emits: the
Qdrant queries, the rerank call, the generation call, the citation resolution,
and the access log. It comes back on the `X-Request-ID` response header.

It is deliberately **not** in the response body. The body is the answer and its
evidence; a caller already holds the id on the response it received, and every
piece of interior detail kept out of the payload is one that cannot be read off a
public endpoint.

That single id is what makes an incident tractable in M4 specifically. Before
generation, one question was one Qdrant round trip; now one question fans out
into an embedding call, two searches, a rerank provider call and an LLM call,
each with its own latency and its own way of failing. Without one id on every
line, "why was this answer bad" is a guess. Tracing proper — spans, a Langfuse
or OTel backend — is M5, opt-in and off by default; the id is the seam it
attaches to. See [Observability](#observability-m5) and
[ADR 0006](adr/0006-observability.md).

### No secret ever reaches a response or a log

Three rules, all of them already enforced by existing machinery and all of them
newly relevant now that a provider call sits on the request path:

1. **Keys are read from the environment by name only.** `configs/default.yaml`
   holds `api_key_env: OPENAI_API_KEY` — the *name* of a variable, never a
   value. `Settings.safe_dump()` is the only sanctioned way to render settings
   and it masks; a test asserts the masking.
2. **Prompt and passage text are not logged at INFO.**
   `observability.logging.log_prompts` and `log_retrieved_text` are both `false`
   by default. The prompt now contains customer corpus text verbatim, so a log
   aggregator with these switched on becomes a copy of the corpus with none of
   its access controls. Turn them on for a local debugging session; never in a
   deployment.
3. **Provider errors are summarised, not echoed.** An upstream error body can
   contain the request that produced it, which for a generation call is the
   whole prompt. What reaches the client is a status and a stable message; the
   detail stays server-side, keyed by the request id.

A refusal and its reason are reported in the response; truncation, invalid
markers and uncited claims are reported on the library result and in the logs.
Credentials, prompts and raw provider payloads are reported nowhere. The CLI
follows the same rule — a failure prints the exception *type*, never the provider
message, because an SDK error can carry the request that caused it.

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

## What is live vs declared-only

`configs/default.yaml` is deliberately broader than what the code consumes.

| Config block | State after M6 |
|---|---|
| `ingest` (walk, chunking, embedding, incremental) | live |
| `ingest.sparse` (BM25 k1/b, lowercase, stopwords) | **live** — vectors are written at ingest |
| `qdrant` (collection, dense + sparse vectors, payload indexes, write consistency) | live |
| `retrieval` (mode, top-k per branch, RRF, threshold, payload fields) | **live** — read by the retrieve command and by the query graph |
| `retrieval.filters.allowed_fields` | declared only — `POST /v1/query` accepts no `filters` field yet, so there is nothing to enforce the allowlist against. The keys stay because the payload indexes that would make a filter cheap already exist |
| `rerank` (enabled, provider, input_top_k, top_k, timeout, fail_open) | **live** — read by the retrieve command and by the query graph. Ordering quality is real only on `local` / `cohere`; the `fake` provider exercises the stage, not relevance |
| `generation` (provider, model, temperature, budgets, timeout, retries, stream) | **live** — read by the query path. Answer quality is real only on `openai`; the `fake` provider exercises the contract, not the answer |
| `generation.citations` (style, require_citation, refuse_without_evidence, refusal_message) | **live** — `[n]` markers are resolved and the refusal edge is taken from these keys |
| `generation.prompt` (system_path, include_heading_path, max_chunks_in_prompt) | **live** — the system prompt is a file under `configs/prompts/`, not a string in code |
| `evals.dataset_path` | **live** — the golden set is the default input of both tier-1 commands (`--golden` overrides it) |
| `evals.retrieval.metrics`, `evals.answer.*` | declared only — the runner takes tier, k, sample, embedder, model and judge from CLI flags, not from this block. `recall@k`/`ndcg@k` as *specified* need chunk-level labels that do not exist; what runs is the source-level equivalent |
| `evals.thresholds` | **read by nothing**, and not baseline-derived. The only gate is the runner's `--fail-under-hit` flag on `source_hit_at_k`, default `0.0`. See [evaluation](evaluation.md#thresholds) |
| `observability.logging` (level, format, `include_request_id`) | **live** — structured JSON logs with a request id on every line |
| `observability.logging.log_prompts` / `log_retrieved_text` | **live as a guard**: both default `false` and nothing logs prompt or passage text. Local debugging only, never a deployment — see [ADR 0006](adr/0006-observability.md) |
| `observability.tracing` (Langfuse, keys by env-var name, `sample_rate`) | **opt-in** — off by default, and the whole system runs offline without it; a trace failure never fails a request |
| `observability.metrics` (`/metrics`, latency buckets) | declared only — the endpoint is not wired |
| `observability.health` | live since M0 — liveness never checks a dependency, readiness does |

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

### Retrieval failure behaviour (M2 + M3)

| Failure | Behaviour | Rationale |
|---|---|---|
| Collection has no `sparse` named vector | abort with the recreate instruction | see [migration](#migration-m2-needs-a-collection-rebuild); degrading to dense-only silently would be worse |
| Collection missing entirely | abort, exit 2 | nothing has been ingested; retrying cannot help |
| Embedding provider 429 on the query embed | bounded retry with backoff, then fail the query | an empty result set is indistinguishable from "no matches" |
| Sparse branch returns nothing (all query terms out of vocabulary) | dense results are returned, and the empty branch is reported | legitimate: an all-stopword query has no lexical signal |
| No hit clears `score_threshold` | empty result set, reported as such | with the default `0.0` this cannot happen; it is a config decision, not a fault |
| Reranker errors, times out, or returns a malformed response | fusion order is returned and the degradation is reported (`fail_open: true`, default) | the un-reranked result is correct in kind, only ordered worse; availability beats a few points of nDCG |
| Same, with `fail_open: false` | the query fails | a deployment that would rather 5xx than serve un-reranked hits; a provider outage becomes an outage |
| `local` reranker requested but the extra is not installed, or `cohere` without `COHERE_API_KEY` | abort, exit 2, before any query runs | a missing provider is a bad invocation, not a runtime fault — and silently falling back to `fake` would fabricate a quality claim |

The asymmetry is deliberate, twice over. A *missing capability* (no sparse vector
on the collection, or a reranker that was never installed) aborts; an *empty
result from a working capability* is data; a *failed improvement* over a result
that already exists degrades and says so.

### Query-path failure behaviour (M4)

Everything in the retrieval table above still applies — the query path runs those
stages. What follows is what the HTTP surface and the generation stage add.

| Failure | Behaviour | Rationale |
|---|---|---|
| Empty or whitespace-only question | 422, no retrieval, no LLM call | a bad request is not an outage; fail before spending anything. Strings are stripped first, so `"   "` is empty |
| Unknown field in the request body | 422 | `extra="forbid"`: a misspelled control must not silently fall back to a default and answer a different question than the one asked |
| Nothing clears the evidence bar | 200, `refused: true`, `refusal_reason: no_evidence`, no LLM call | the corpus not covering a question is a correct outcome, not an error status |
| Model emits an out-of-range `[n]` | marker stripped from the answer, recorded in `invalid_markers`, answer served | a dead citation link is worse than no marker; recording it makes it measurable |
| Model emits no resolvable citation at all, `require_citation: true` | 200, `refused: true`, `refusal_reason: no_citations` | an uncited answer from a grounded system is either an ignored context or an unsupported claim |
| Generation provider 429 / timeout | the SDK's bounded retry (`max_retries`, `timeout_seconds`), then the request fails | unlike the reranker there is nothing to fall back to: every available fallback is an ungrounded answer |
| Generation provider returns an error body | the failure surfaces as a status and a stable message; the provider detail stays in the server logs, and the CLI prints only the exception type | an upstream error body can quote the request that caused it, which here is the entire prompt |
| Qdrant unreachable during a query | the request fails; no LLM call | answering with no context is exactly the failure mode ADR 0005 exists to prevent |
| `openai` selected with no `OPENAI_API_KEY` | `LLMError` before any query runs | a missing credential is a configuration error; falling back to `fake` would fabricate an answer |
| The query pipeline module is absent from the checkout | 503 from the endpoint | a split-milestone checkout fails honestly instead of growing a second implementation inside the route |

The generation stage is deliberately **not** fail-open, and that is the one place
M4 departs from the reranker's posture. A failed reranker leaves a result that is
correct in kind and merely ordered worse. A failed generator leaves nothing that
can be substituted — every available fallback is an answer without grounding,
which is the thing the milestone is built to refuse.

## Observability (M5)

Everything above describes what the system *does*. This section describes how
anyone finds out what it did. The decision and its alternatives are in
[ADR 0006](adr/0006-observability.md); the mechanics are here.

The problem M5 answers is specific to what M4 built. Through M3 a question was
one Qdrant round trip, so "it was slow" had one suspect. A question is now an
embedding call, two searches, an optional rerank provider call, an LLM call and
two guardrail checks — and the interesting failures in that path do not raise:
an answer is thin because the context budget truncated, the reranker has been
failing open for a week, the model is emitting markers that get stripped before
anyone sees them. Those are numbers the system either records or does not.

```
   request ──▶ ┌────────────────────────────────────────────┐
               │ RequestContextMiddleware                   │
               │  X-Request-ID in (or a fresh UUID4)        │
               │  bound into structlog contextvars          │
               └───────────────────┬────────────────────────┘
                                   │ every line below carries request_id
                                   ▼
               ┌────────────────────────────────────────────┐
               │ query graph — one stopwatch per node       │
               │  retrieve · rerank · guard · generate ·    │
               │  cite · finalise                           │
               │            ↓ writes                        │
               │  QueryState.timings_ms{node: ms}           │
               └───────────────────┬────────────────────────┘
                                   │
        ┌──────────────────────────┼───────────────────────────────┐
        ▼                          ▼                               ▼
 ┌───────────────┐   ┌──────────────────────────┐   ┌──────────────────────────┐
 │ HTTP response │   │ library result           │   │ structured JSON logs     │
 │ answer,       │   │ QueryResult.timings_ms   │   │ request_completed        │
 │ citations,    │   │  → to_dict() latency_ms  │   │ query_completed          │
 │ refused,      │   │ hits_used/hits_retrieved │   │ query_finalised          │
 │ refusal_reason│   │ invalid_markers          │   │ context_truncated        │
 │               │   │ uncited_claims           │   │ rerank_failed_open       │
 │ + diagnostics │   │ rerank{applied,          │   │                          │
 │   iff debug   │   │        candidates, error}│   │ never: prompts, passages │
 └───────┬───────┘   └────────────┬─────────────┘   └────────────┬─────────────┘
         │ X-Request-ID           │                              │
         │ X-Response-Time-ms     │                              │
         ▼                        ▼                              ▼
     the caller            eval harness (tier 2)          log aggregator
                                                                 │
                                      opt-in, off by default ────┤
                                                                 ▼
                                                    Langfuse (traces)
                                          enabled + 3 env vars, fail-open
```

Three sinks, three audiences, one join key. The response serves the caller, the
library result serves a harness or a CLI, the logs serve an operator — and the
request id is on all of them, which is what makes an incident a query rather than
a guess.

### Timings are always collected; the response is not always allowed to show them

`timings_ms` is populated on **every** request, with no flag and no sampling. The
cost is one `perf_counter()` pair per node, and the alternative is asking an
operator to reproduce a slow request that already happened. Because the graph
nodes are adapters around exactly one stage each ([ADR 0002](adr/0002-langgraph-query.md)),
stage latency and node latency are the same number by construction; the node
names are constants in `production_rag.graph.state`, so a rename cannot silently
break a dashboard keyed on them.

Where that dict is readable differs by surface, and the difference is a security
boundary rather than an oversight:

| Surface | Timings | Why |
|---|---|---|
| Library result (`run_query(...)`) | always — `timings_ms`, and `latency_ms` + `total_ms` via `to_dict()` | the caller is the process itself |
| CLI (`python -m production_rag.query --debug`) | on request | already inside the trust boundary; the operator ran it |
| `POST /v1/query` | only with `debug: true`, and only the safe subset | the caller is not necessarily trusted |
| Logs | always, in `query_completed` | keyed by request id, behind whatever guards the aggregator has |

The library and log columns are live as of M4 — `timings_ms` has been on the
state object and in `query_completed` since the graph landed. What M5 defines is
the **`debug` projection**: the request field has been accepted and validated
since M4 but widened nothing, and the contract above is what the M5 query and
API code implements against. A response with no `diagnostics` object under
`debug: true` is a checkout that predates it, not a different contract.

**`debug` is caller-controlled**, and that single fact defines its contract.
Anyone who can reach the endpoint can set it, so it is not an authenticated
diagnostic channel and may only widen the response to things that would be safe
to publish:

| Exposed under `debug: true` | Withheld regardless |
|---|---|
| per-node `timings_ms` and the total | prompt text, system prompt, rendered blocks |
| `hits_retrieved` / `hits_used` | passage text beyond the citations already returned |
| `rerank` summary — `applied`, `candidates`, `error` | collection name, embedder model, provider identity |
| `invalid_markers` | credentials, in any form, at any level |

The subset arrives as one optional `diagnostics` object rather than as fields
spliced into the body, so the four stable response fields never change shape
depending on a flag and a client that ignores diagnostics needs no conditional.

`debug` answers *what the system did*, never *what the system knows*. The
withheld column is not a to-do list: those fields stay on the library result and
in the logs on purpose, because a public endpoint that reports its own collection
name and embedding model is describing its interior to anyone who asks.

### The three ops signals that need no judge

Answer-quality measurement is offline, sampled, and judged
([evaluation](evaluation.md)). These three are none of those things — they are produced by the request path
itself, on every request, for free, and they are what an operator watches:

| Signal | Where | What a change in it means |
|---|---|---|
| `timings_ms` per node | library result, logs, `debug` response | which stage got slow. `generate` dominating is normal; `rerank` dominating is `input_top_k`; `retrieve` dominating points at Qdrant |
| `invalid_markers` | library result, `query_finalised` log | the model is emitting citations that resolve to nothing. They are stripped from the answer, so without this field the misbehaviour is invisible |
| `hits_used` vs `hits_retrieved` | library result | the context budget truncated the tail, so retrieval order was truncation order and the reranker decided what the model could cite |

Two more are degradation reporters rather than steady signals: `rerank.error`
with `rerank.applied: false` (the reranker failed open — one is noise, a steady
stream is an ordering incident), and `refusal_reason` (`no_evidence` is a
retrieval miss, the other three are the model).

None of these is a quality metric. `invalid_markers` at zero says nothing about
whether the markers that *did* resolve support their sentences — that needs a
judge. The distinction is drawn in full in
[evaluation](evaluation.md#ops-signals-are-not-eval-metrics).

### Tracing is an export, never a dependency

`observability.tracing` configures Langfuse, off by default, reading
`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` and `LANGFUSE_HOST` **by name**.
Three properties hold, and they are what makes it an export rather than an
integration:

- **Offline works.** Every command, the endpoint, the CLI, the tests and the eval
  script run with nothing configured and no network to a trace backend. A laptop
  with no credentials is this project's reference environment.
- **A trace failure is not a request failure.** The exporter is fail-open in the
  same sense as the reranker: backend down, slow or misconfigured, the request
  still answers and the degradation is logged. Losing a diagnostic is not losing
  the product.
- **No vendor inside the library.** Tracing attaches at the pipeline boundary —
  the seam the request id already occupies — not inside
  `production_rag.retrieval` or `production_rag.generation`. A vendor client
  called from a retrieval function makes that function untestable without the
  vendor.

What it costs when it is on: a generation trace *is* the prompt and the answer,
sent to a third party. `sample_rate` bounds volume, not sensitivity. Turning it
on is a decision about where corpus text may go, not a toggle.

### Logs: what is recorded, and the one field that is user data

Structured JSON via structlog, `request_id` bound once in the middleware and
therefore present on every line the request emits. The events worth knowing:

| Event | Emitted by | Carries |
|---|---|---|
| `request_completed` | middleware | method, path, status, `duration_ms` — server-side handling time, also returned as `X-Response-Time-ms` |
| `request_failed` | middleware | the same, plus the traceback; the exception is re-raised, never swallowed |
| `query_completed` | pipeline | mode, hit count, citation count, `refused`, `refusal_reason`, model, `timings_ms` — **and the question text** |
| `query_finalised` | finalise node | `refused`, `refusal_reason`, citation count, `uncited_claims`, `invalid_markers` |
| `context_truncated` | generation | kept and dropped chunk counts |
| `rerank_failed_open` | rerank stage | provider and error — a degradation that would otherwise be silent |

`query_completed` logging the **question** is a deliberate exception to "text is
not a signal", and it is called out because it is one. The question is
user-supplied rather than corpus content, and it is what makes a report
correlatable at all; a deployment handling sensitive questions should treat that
line as personal data and drop the field at the aggregator. What is *never*
logged is the prompt, the retrieved passages, provider error bodies, and
credentials in any form — see
[No secret ever reaches a response or a log](#no-secret-ever-reaches-a-response-or-a-log)
and the standing warning on `log_prompts` in the [runbook](runbook.md#never-turn-on-log_prompts-in-a-deployment).

### What M5 does not add

- **No metrics endpoint.** `observability.metrics` describes a Prometheus
  exposition on `/metrics` with buckets shaped for a RAG request; it is not
  wired. The config keys record the intended shape, and the file says so.
- **No cost or token accounting.** The provider returns token counts on the
  generation call; nothing aggregates them and the result does not carry them.
  Latency is not spend.
- **No per-branch timing inside `retrieve`.** The stopwatch is around the node,
  so the dense and sparse branches report as one number. Attributing a slow
  retrieve to a branch still means running the retrieve command per mode.
- **No behaviour under load.** Per-request timings say nothing about concurrency;
  load testing waits for a real deployment target.

## Evaluation (M6)

Evaluation is an **offline path**, structurally separate from the request path.
It reuses the retriever and `run_query` rather than the endpoint: an eval that
drives HTTP would measure serialization, and one that reimplements the pipeline
would measure the reimplementation — and the divergence would show up as a
passing eval over a broken service.

```
data/eval/golden.jsonl                     configs/default.yaml
  17 items, document-level labels            evals.dataset_path (default only)
  answerable: true|false                     evals.thresholds ── read by nothing
         │                                            │
         ▼                                            ▼
┌───────────────────────── evals.run ─────────────────────────┐
│  --tier 1 | 2 | all   --embedder --llm --judge --sample     │
│  offline defaults: fake embedder, fake model, fake judge    │
└───────────┬─────────────────────────────┬───────────────────┘
            ▼                             ▼
┌───────────────────────────┐ ┌───────────────────────────────┐
│ TIER 1 — tier1_retrieval  │ │ TIER 2 — tier2_answer         │
│ built on source_hit       │ │ answers via run_query         │
│                           │ │                               │
│ source_hit_at_k           │ │ judge-free:                   │
│ source_recall_at_k        │ │   citation_precision          │
│ mrr, ndcg_at_k            │ │   invalid_marker_rate         │
│ by_category               │ │   refusal_accuracy            │
│                           │ │ judged (AnswerJudge):         │
│ no judge, no key,         │ │   faithfulness, relevance     │
│ deterministic             │ │ hosted judge: --judge openai  │
│                           │ │   + RUN_LLM_EVALS=1 + key     │
└───────────┬───────────────┘ └───────────────┬───────────────┘
            └──────────────┬──────────────────┘
                           ▼
        one versioned JSON report (report_version: 1)
        offline_defaults · embedder/llm/rerank · gate{...}
                           │
                           ▼
              --fail-under-hit (default 0.0 = report only)
              exit 1 when source_hit_at_k falls below it
```

Five properties of that picture are load-bearing:

- **One runner, one report.** ADR-0003's stated cost of a two-tier split was
  "two commands and two sets of results to reconcile". Both tiers run in one
  process over the same sample and the same collection, so the numbers are
  comparable by construction.
- **The tier-1 aggregate denominator is 13, not 17.** The four
  `answerable: false` items carry no `expected_source_paths`, and
  `GoldenCase.is_scorable` is exactly `bool(expected_source_paths)`. They are
  excluded and reported as `unscored_cases`; their correct outcome is a refusal,
  which is tier 2's `refusal_accuracy`.
- **Everything is source-level.** `source_hit_at_k` and `source_recall_at_k` are
  named that way because the labels are documents, not chunks;
  `citation_precision` inherits the same limit — right document, not necessarily
  supporting passage.
- **Cost is opt-in twice.** A hosted judge needs the flag, the
  `RUN_LLM_EVALS=1` environment variable and a credential. The default run is
  free, offline and deterministic, and the report says so in `offline_defaults`
  rather than leaving it to be inferred from the score.
- **One gate, off by default.** `--fail-under-hit` is the only thing that can
  fail a run, it scores the deterministic metric, and `configs/default.yaml`
  thresholds are read by nothing. See
  [evaluation](evaluation.md#thresholds).

The eval path bypasses HTTP and tracing. Tier 2 still supplies a deterministic
`eval-<case-id>` request id because it deliberately reuses the real query
pipeline. Conversely the ops signals the request path emits —
`timings_ms`, `invalid_markers`, `hits_used` — are not eval metrics, and the
distinction is spelled out in
[evaluation](evaluation.md#ops-signals-are-not-eval-metrics).

## Deployment shape

Local development is the only documented target through M6 and it is `docker compose up -d
--build`. The compose file is written to be promotion-friendly: pinned image
tags, healthchecks on both services, named volume for the vector index, and
secrets injected exclusively via environment (never files baked into images).

One M3 note for any non-local target: the `local` reranker needs its model
weights (~1.1 GB) present before the first query. Bake them into the image or
mount a warm cache — a cold container that downloads a cross-encoder on its first
request is a latency incident, not a cold start. See the
[runbook](runbook.md#the-local-reranker-downloads-a-model).

Two M4 notes. The API container now makes an outbound LLM call on the request
path, so `OPENAI_API_KEY` has to reach it through the environment — never a file
baked into an image, never a command-line flag that lands in `docker inspect`.
And a request without authentication can now spend money per call, which is the
reason rate limiting moves from "nice to have" to M7's headline: an unprotected
`POST /v1/query` on a public address is a billing incident waiting for a crawler.

## Non-goals (M4 + M5 + M6)

- No defensible **answer-quality** number. M6 scores answers, but the default
  judge is lexical overlap, no judge has been calibrated against hand labels, and
  `citation_precision` checks the cited *document* rather than the cited passage.
  The citation mechanics make an unfaithful answer easy to *check*; nothing here
  measures a faithfulness rate that is worth quoting.
- No armed merge gate. The mechanism exists (`--fail-under-hit` on
  `source_hit_at_k`) and defaults to reporting; the baseline run that would set
  its value has not been performed, and 17 golden items cannot carry a
  threshold.
- No measured retrieval quality number either, before or after rerank. The
  `hit@k` script reports what the current corpus, embedder and reranker produce;
  on the `fake` embedder and the `fake` reranker that number is plumbing.
- No query rewriting, no self-critique loop, no multi-hop. The graph has the
  shape that would host them; it has no cycle today (see
  [ADR 0002](adr/0002-langgraph-query.md)).
- No conversational memory. Every query is independent; there is no session, no
  history and no follow-up resolution.
- No authentication/authorization on the API — and it now guards a paid path.
- No horizontal scaling, no API gateway.
- No GPU in the runtime path. The `local` cross-encoder is CPU-only by design.
- No metrics endpoint, no cost accounting, no load-tested latency. M5 makes one
  request legible; it does not make a fleet legible. See
  [what M5 does not add](#what-m5-does-not-add).
