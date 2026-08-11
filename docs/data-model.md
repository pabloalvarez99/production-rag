# Data Model

How data is shaped across the corpus, the chunk artifacts, the Qdrant collection,
and the hits the retrieve command emits. Scope: **M2 + M3** — both named vectors,
dense and sparse, are written by the ingest job and read by the retriever, and a
hit can now also carry what the rerank stage did to it. Nothing here is
aspirational except where a field is explicitly marked as belonging to a later
milestone.

## Corpus documents (`data/raw/`)

Plain Markdown or text files. The file itself is the unit of provenance:

- `source_path` — path relative to `data/raw/`, e.g. `sample/00-intro.md`.
- `source` — the first path segment (`sample`), used as the filter dimension.
- Optional YAML front matter supplies `title` and `tags`. Without it, the first
  H1 becomes the title and `tags` is empty.

Only extensions listed in `configs/default.yaml → ingest.include_extensions`
are ingested; anything else is skipped loudly.

## Chunks

Documents are split with a recursive strategy: structural boundaries first
(headings, then paragraphs, then sentences), falling back to a hard character
cut. Chunk size and overlap are set in `configs/default.yaml → ingest.chunking`.

### Chunk identity is derived, never random

`chunk_id` is `<doc_id>:<chunk_index>`, where `doc_id` is a hash of
`source_path`. Nothing in it is random, so re-ingesting an unchanged document
produces the same ids and a stored citation still resolves.

Two consequences follow, and the second is the nastier one:

- **Renaming a file changes every id in it,** because `doc_id` hashes the path.
  The old points are orphaned until a full re-ingest — nothing links the new
  path to the old one.
- **Changing `chunk_size` or `chunk_overlap` keeps the ids and moves the text
  underneath them.** `chunk_id` is `<doc_id>:<index>`, so `…:0003` still exists
  after a re-chunk; it just holds a different passage. Eval labels and stored
  citations therefore do not break loudly, they silently point somewhere else.
  Re-chunking is a re-labelling event, and the labels have to be re-checked by
  hand rather than by a missing-key error.

## Qdrant point — the exact shape written by M2

One collection (default name `production_rag`, overridable via
`QDRANT_COLLECTION`) holds everything. One point per chunk:

```jsonc
{
  // Point id: UUID5 over "<source_path>::<chunk_index>::<content_sha256>".
  "id": "6f3a1c2e-8b47-5d90-a1f2-0c9d7e4b3a15",
  "vector": {
    // Two named vectors, written in the same upsert. A point produced by M1
    // carries only `dense` — see "Migration from an M1 collection" below.
    "dense": [0.0123, -0.0456, /* … 1536 floats … */],
    // Sparse vectors are index/value pairs, not a dense array: only the terms
    // present in the chunk appear. Indices are hashed term ids (31 bits of
    // SHA-1), values are full BM25 weights with IDF already applied.
    "sparse": {"indices": [1174, 20388, 44901], "values": [1.82, 0.94, 2.31]}
  },
  "payload": {
    "chunk_id": "9f2c1a7b3d4e5f60:0003",
    "text": "Three ideas, each doing one job: …",
    "source_path": "sample/08-bm25-vs-dense.md",
    "source": "sample",
    "title": "BM25 versus dense embeddings — the mechanics behind the trade-off",
    "heading": "What BM25 actually computes",
    "heading_path": "BM25 versus dense embeddings > What BM25 actually computes",
    "chunk_index": 3,
    "content_sha256": "1b9d6bcd4621d373cade4e832627b4f6…",
    "doc_id": "9f2c1a7b3d4e5f60",
    "token_count_est": 194,
    "tags": ["bm25", "sparse", "dense", "embeddings", "tokenisation"],
    "ingest_run_id": "2026-08-10T14:02:11Z/01J…",
    "embedded_model": "text-embedding-3-small"
  }
}
```

This is `Chunk.to_payload()` in `production_rag.ingest.models`, field for field.
When the two disagree, the code is right and this document is stale.

### Payload fields

| Field | Type | Source | Why it exists |
|---|---|---|---|
| `chunk_id` | string | `<doc_id>:<chunk_index:04d>` | the citation target; what eval labels will name. Zero-padded so it sorts lexicographically |
| `text` | string | the chunk body | returned with the hit, so no second store is needed |
| `source_path` | string | path relative to the corpus root, forward slashes | provenance shown to the user |
| `source` | string | first path segment, or `"root"` for a file in the corpus root | the filter dimension; keyword-indexed |
| `title` | string \| null | front matter `title`, else first H1, else the file stem | prefixed to the embedded text |
| `heading` | string \| null | nearest enclosing heading | shown with a citation |
| `heading_path` | string \| null | heading ancestry joined with `" > "` | a string, not an array — it is rendered, not iterated |
| `chunk_index` | int | position within the document | ordering and id derivation |
| `content_sha256` | string | SHA-256 of `text`, NFC-normalised first | drives the incremental skip and the point id |
| `doc_id` | string | first 16 hex chars of SHA-256 over `source_path` | groups the chunks of one document; keyword-indexed |
| `token_count_est` | int | `len(text) / 4`, rounded up | context budgeting; an estimate by design, no tokeniser is loaded |
| `tags` | string[] | front matter `tags`, else `[]` | filterable slicing; keyword-indexed |
| `ingest_run_id` | string | stamped per run | "which run wrote this?" — the first question when quality moves |
| `embedded_model` | string | the model that produced the vector | a collection with two models mixed in is otherwise undetectable |

`embed_text` is deliberately **not** in the payload: it is derivable from `text`
plus the heading fields, and storing it would double the payload size of every
point in the collection.

### Three ids, three questions

| Id | Answers | Form |
|---|---|---|
| `doc_id` | which document is this? | 16 hex chars of `sha256(source_path)` |
| `chunk_id` | which chunk of it? | `<doc_id>:0003` — human-readable, citable, sortable |
| point id | which row in Qdrant? | UUID5 over `<source_path>::<chunk_index>::<content_sha256>` |

Qdrant accepts only an unsigned integer or a UUID as a point id, so the citable
`chunk_id` cannot be the point id directly.

The point id is **content-addressed**, which has a consequence worth stating
plainly: re-ingesting unchanged content upserts the same point ids and is a
no-op, but editing a paragraph produces a *new* point id. The old point is not
overwritten — it is still there, holding the previous text under the same
`chunk_id`. Until the ingest job deletes stale points for a re-ingested document,
a collection can hold two generations of the same chunk.

Nothing about the hashing is process-dependent: Python's built-in `hash()` is
salted per process and would produce different ids on every run, so every
identifier goes through SHA-256. Paths are normalised to forward slashes first,
so the same document ingested on Windows and inside the Linux container gets the
same `doc_id`.

### The embedded text is not the stored text

`payload.text` is the clean chunk body — what gets cited and what a reader sees.
What the embedding model receives is `embed_text`: the same body, optionally
prefixed with the title and heading path when
`ingest.chunking.prepend_heading_context` is on. The prefix costs a few tokens
and measurably helps short queries; letting it reach the payload would put it in
every generation prompt for no benefit and make citations read oddly.

### Vectors

| Named vector | State after M2 | Spec |
|---|---|---|
| `dense` | **written** | size `1536`, distance `Cosine`, from `text-embedding-3-small`. Must match `ingest.embedding.dimensions`; a mismatch aborts the run rather than writing vectors that fail much later. |
| `sparse` | **written** | Full BM25 term weights, `modifier: none`, no fixed dimensionality. Parameters in `configs/default.yaml → ingest.sparse` (`k1`, `b`, `lowercase`, `stopwords`). See [ADR 0001](adr/0001-hybrid-qdrant.md). |

Term identity is a **hash, not a vocabulary**: the sparse index of a term is the
first 31 bits of its SHA-1. No vocabulary file to ship, version or keep in sync,
and a query can weight a term the ingest run never saw. Two distinct terms can
collide; at 2³¹ slots and this corpus size that degrades one ranking slightly
rather than corrupting anything.

The two are written in a single upsert, so a point cannot carry one without the
other. That invariant is what makes "the sparse index is stale relative to the
dense one" impossible by construction rather than by discipline.

#### Migration from an M1 collection

M1 created the collection with `dense` as its **only** named vector. It did not
declare an empty `sparse` vector — earlier revisions of this document and of
`configs/default.yaml` said it did, and that was wrong about the shipped code.

Consequence: an M1 collection cannot serve hybrid retrieval and cannot be
upgraded in place by the M2 ingest job. Rebuild it:

```bash
make reingest-fake     # or: scripts/ingest.ps1 -Recreate
```

Free on the `fake` embedder. On `openai` it is a **full re-embed of every
chunk**, billed, because the collection the content-hash skip would compare
against is the one being dropped. Budget it with `make ingest-dry` first.

#### Why sparse weights are computed at ingest, not at query time

BM25 needs corpus statistics — document frequency per term, average document
length — and those are properties of the whole corpus, not of one chunk. The
ingest job has the corpus in hand; the retriever has one question.

The stored value is the **complete** BM25 weight, IDF included:

```
weight(term, chunk) = idf(term) · (tf · (k1 + 1)) / (tf + k1 · (1 − b + b · dl/avgdl))
idf(term)           = ln(1 + (N − df + 0.5) / (df + 0.5))
```

A query vector then carries weight `1.0` per distinct query term, so the dot
product of query and document vector *is* the BM25 score. Two things follow:
querying needs the tokenizer only — no fitted statistics, no warm-up, no
corpus-state dependency at query time — and Qdrant's `modifier: idf` must stay
**off** on this vector, or IDF is applied twice and rare tokens dominate far
beyond what BM25 says. `configs/default.yaml` sets `modifier: none` for exactly
that reason.

The cost is **IDF drift**: term weights stored in the index reflect the corpus as
it stood at the last full ingest. Add a batch of documents about one topic and
the true IDF of its vocabulary drops, while the stored weights keep the old,
higher values — that topic's chunks stay slightly over-ranked on the sparse
branch until the next full re-ingest. Incremental ingest makes this worse, not
better: it skips unchanged chunks, so their weights are exactly the ones that do
not get refreshed.

At this corpus size the effect is not measurable. It becomes real at a scale
where a full re-ingest stops being a coffee break, and the fix at that point is a
periodic full re-encode of the sparse side, not a smarter query. Handing IDF to
Qdrant's query-time `idf` modifier would remove the drift, but only by moving
the whole weighting there — it cannot be layered on top of weights that already
contain it.

The `fake` embedder writes dense vectors of the same declared dimensionality, so
the collection shape is identical whichever embedder ran. Only the dense numbers
are meaningless — the sparse vectors are computed from the text either way, which
is why a `fake`-embedded collection still produces a genuinely lexical sparse
branch.

### Payload indexes

Keyword indexes on `doc_id`, `source`, and `tags`
(`configs/default.yaml → qdrant.payload_indexes`). Those three are the fields
the query API will be allowed to filter on; an unindexed filter degrades to a
payload scan.

There is deliberately no separate metadata database: Qdrant payloads cover every
field the query path will read.

## Retrieval hit — the shape the retrieve command emits

Nothing below is stored: this is the read-side shape, produced per query by
`production_rag.retrieval.hybrid` and printed as the last line of stdout by the
retrieve command. It is `RetrievalHit.to_dict()`, field for field.

```jsonc
{
  "rank": 2,                       // final position, after rerank if it ran
  "score": 0.032258,               // fused RRF score (not a similarity)
  "chunk_id": "9f2c1a7b3d4e5f60:0003",
  "source_path": "sample/08-bm25-vs-dense.md",
  "title": "BM25 versus dense embeddings",
  "heading": "What BM25 actually computes",
  "heading_path": "BM25 versus dense embeddings > What BM25 actually computes",
  "point_id": "6f3a1c2e-8b47-5d90-a1f2-0c9d7e4b3a15",
  "branches": ["dense", "sparse"], // which branches returned it
  "branch_ranks":  {"dense": 14, "sparse": 1},
  "branch_scores": {"dense": 0.7412, "sparse": 11.83},
  "pre_rerank_rank": 27,           // M3, present only when a reranker ran
  "rerank_score": 0.8713,          // M3, present only when a reranker ran
  "text": "Three ideas, each doing one job: …"
}
```

### The two M3 fields, and why they are optional keys

| Field | Type | Set when | Why it exists |
|---|---|---|---|
| `pre_rerank_rank` | int | a reranker ran | the position fusion gave the hit. Without it, "the reranker moved this to the top" and "fusion had it at the top all along" are the same observation, and the stage's contribution is unmeasurable |
| `rerank_score` | float | a reranker ran | the cross-encoder's score for this `(query, passage)` pair. Provider-specific scale — comparable within one run, never across providers |

Both keys are **absent**, not null, when no reranker ran. An M2-era consumer of
this JSON therefore keeps parsing M3 output unchanged, and the presence of the
key is itself the signal that the stage ran on this hit.

`pre_rerank_rank` is set once and never overwritten: it holds the fusion
position, not the position before the most recent reordering. `rank` is always
the final position.

`score` remains the **fused RRF score** after reranking — it is not replaced by
`rerank_score`. The two live on unrelated scales, and overwriting one with the
other would make the fused score unrecoverable and every historical comparison
wrong. So after a reranked run, `rank` and `score` can disagree in direction:
rank 1 need not hold the highest `score`, and that is the visible evidence that
reranking did something.

### The run-level `rerank` summary

Alongside the hits, the result carries one object describing the stage — present
on every run, including runs where nothing reranked:

```jsonc
"rerank": {
  "applied": false,        // did reranking actually reorder these hits?
  "reranker": "fake",      // provider name or model id; null when off
  "candidates": 40,        // how many hits the stage was given
  "error": null            // the failure text when fail_open swallowed one
}
```

`candidates` is the honest answer to "was the right chunk even eligible?" — with
rerank on, anything outside `input_top_k` is unreachable by construction.
`applied: false` with a non-null `error` is a fail-open degradation; `applied:
false` with `reranker: null` is simply a run with reranking off. Reporting the
distinction is the whole point: a rerank that quietly stopped happening looks
exactly like a rerank that is not helping. See
[ADR 0004](adr/0004-rerank-cross-encoder.md).

## Derived artifacts (`data/processed/`)

Chunking/embedding caches the ingest job may write to avoid recomputation.
Gitignored, excluded from the Docker build context, and safe to delete at any
time — the ingest job rebuilds them from `data/raw/`.

## Eval datasets (`data/eval/`)

`golden.jsonl`, read by the evaluation harness (see
[evaluation.md](evaluation.md)). Committed, versioned with the repo, and never
written by the runtime.

The committed seed set labels **document paths**, not chunk ids:

```jsonc
{"id": "q-0006", "question": "…", "expected_source_paths": ["sample/08-bm25-vs-dense.md"], "category": "exact_token"}
```

Document-level labels are a weaker signal than `relevant_chunk_ids` — they
support `hit@k`, not `recall@k` — and they are the only labels that survive a
re-chunk, which is why they exist first. Field spec in
[`data/eval/README.md`](../data/eval/README.md).
