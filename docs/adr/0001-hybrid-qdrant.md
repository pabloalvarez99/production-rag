# ADR 0001 — Hybrid retrieval on a single Qdrant collection

- **Status:** Accepted — implemented in M2
- **Date:** 2026-08-10 (proposed), 2026-08-10 (accepted on M2 landing)
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
- `sparse` — full BM25 term weights computed at ingest, IDF folded in, stored as
  a sparse vector with Qdrant's own `idf` modifier **off**.

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
- Named vectors mean dense and sparse can never drift out of sync *within a
  point* — both are written in the same upsert, so a point carries both or is
  not written. (Corpus-wide IDF is a different kind of staleness; see below.)
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

## Status change — accepted on M2 (2026-08-10)

Implemented as specified. Both branches are queried against one collection, RRF
with `k = 60` fuses them, and every fused hit carries the rank it held in each
branch plus that branch's contribution to its score, so a result's position is
explainable rather than merely reported. Three things learned in the build are
worth recording; the first two contradict something written while this ADR was
still *Proposed*.

### 1. M1 did not pre-declare the `sparse` named vector

The plan assumed M1 would create the collection with an empty `sparse` vector so
that M2 would be a backfill. **The shipped M1 code did not do this** — it called
`create_collection` with `vectors_config={"dense": …}` and no
`sparse_vectors_config`. The assumption reached `configs/default.yaml`,
`docs/architecture.md` and `docs/data-model.md` as a statement of fact and has
now been corrected in all three.

M2 is therefore a **migration**: `--recreate-collection` plus a full re-ingest,
free on the `fake` embedder and a full billed re-embed on `openai`. There was no
cheaper option to weigh — Qdrant cannot add a named vector to an existing
collection at all, so the choice the *Proposed* text thought it was making
(backfill versus migration) did not exist.

The general lesson is cheaper than the migration: a schema commitment written in
an ADR is not a schema commitment made in code. If a milestone's design depends
on a previous milestone having declared something, assert it against the running
collection rather than against the document that promised it.

### 2. The sparse branch is real even on the `fake` embedder

BM25 weights are computed from the chunk text, not from the embedding model, so a
collection built with `--embedder fake` still has a genuinely lexical sparse
side. Dense scores in that collection are hash noise; sparse scores are not.

This makes the offline path more useful than expected — exact-token retrieval can
be exercised and even measured with no API key — and more dangerous to quote: a
`hit@k` from a `fake` collection reflects lexical matching only and must always
be reported with the embedder that produced it. See
[evaluation.md](../evaluation.md#reading-a-hitk-number-honestly).

### 3. The IDF modifier is off, and that is not an oversight

This ADR originally specified the sparse vector as carrying BM25 weights "with
an IDF modifier". Both halves cannot be true at once. Ingest stores the
**complete** BM25 weight with IDF already folded in, which is what lets a query
vector carry weight `1.0` per distinct term and makes the dot product exactly
the BM25 score — so querying needs the tokenizer alone, with no fitted state and
no warm-up. Enabling Qdrant's `idf` modifier on top would apply IDF a second
time and square the term, letting rare tokens dominate well past what BM25 says.

`configs/default.yaml` therefore sets `modifier: none`, and the loader keeps the
field visible rather than dropping it, because a declared knob that is silently
ignored is worse than one that is explained.

Handing IDF to Qdrant instead is a coherent alternative — it would fix the drift
in point 1's neighbourhood by computing IDF from live collection statistics — but
it is an either/or, not a layering. Revisit it if drift ever becomes measurable.
