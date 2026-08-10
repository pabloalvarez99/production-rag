# ADR 0001 — Hybrid retrieval on a single Qdrant collection

- **Status:** Proposed
- **Date:** 2026-08-10
- **Deciders:** production-rag maintainers
- **Supersedes:** —

## Context

The service must answer questions over a heterogeneous document corpus. Two
retrieval families are available, and they fail on disjoint inputs:

- **Dense (embedding kNN)** generalises over paraphrase and synonymy, and fails
  on rare literal tokens the embedding model never saw — error codes, SKUs,
  function names, internal acronyms. These get mapped near the centroid of
  "meaningless string" and rank arbitrarily.
- **Sparse (BM25)** nails exact tokens and rare-term weighting, and fails
  whenever the question and the document share no surface vocabulary.

A corpus of technical documentation contains both query shapes in roughly equal
measure, so choosing one family means accepting a known, permanent recall hole.

Secondary constraints:

- One storage system, not two. Running an embedding store next to a separate
  lexical engine (Elasticsearch, OpenSearch) doubles the operational surface and
  introduces a consistency problem — two indexes that can disagree about which
  documents exist.
- Retrieval must be reproducible offline for evaluation, so the fusion step
  cannot depend on a hosted black-box ranker.
- The local stack must fit in a laptop-sized `docker compose`.

## Decision

Use **Qdrant** as the single store and run **hybrid retrieval** against **one
collection carrying two named vectors** per point:

- `dense` — 1536-d, cosine, from `text-embedding-3-small`.
- `sparse` — BM25 term weights computed at ingest, stored as a sparse vector
  with an IDF modifier.

Both branches are queried in the same request. Results are combined with
**Reciprocal Rank Fusion** (`score = Σ 1/(k + rank)`, `k = 60`) rather than
weighted score blending.

RRF is chosen specifically because cosine similarity and BM25 scores live on
incomparable scales. Normalising them requires corpus-dependent calibration that
drifts as the corpus grows; rank position does not drift. RRF has one constant,
it is interpretable, and it is trivially reproducible in an offline eval run.

Both branches pull 40 candidates and fusion emits 12. Over-retrieving is
deliberate: fusion can only reorder what it was given.

## Consequences

**Positive**

- The largest class of "the answer was right there and it didn't find it" bugs
  is removed rather than tuned around.
- One container, one collection, one backup unit, one consistency domain. A
  document either exists in the collection or it does not.
- Named vectors mean dense and sparse can never drift out of sync — they are
  written in the same upsert.
- RRF adds one integer constant to the config surface, not a calibration job.
- Both branches are individually measurable, so evaluation can attribute a
  regression to a specific retriever (see [ADR-0003](0003-eval-strategy.md)).

**Negative**

- BM25 statistics are computed by the ingest job, not by the store. A corpus
  that grows substantially shifts IDF, and the sparse index becomes mildly stale
  until the next full ingest. Acceptable at this corpus size; it becomes a real
  problem in the tens of millions of chunks.
- Two vector searches per query instead of one. Measured in tens of
  milliseconds, so it is dominated by generation latency, but it is not free.
- RRF discards score magnitude. A chunk that is overwhelmingly the best match
  contributes exactly the same as a merely-good one at the same rank. This is
  the price of scale-independence, and it is partly what the rerank stage is
  there to recover.
- Qdrant's storage format is version-sensitive, which is why the image tag is
  pinned and upgrades require a backup step (see the runbook).

**Neutral / follow-ups**

- Fusion weights are equal until evaluation says otherwise. Do not tune them by
  intuition — per-branch retrieval metrics exist for exactly this.
- If the corpus later becomes single-domain and purely conceptual, the sparse
  branch could be disabled via `retrieval.mode: dense` without a schema change.
