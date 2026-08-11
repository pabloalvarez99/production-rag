# Data Model

How data is shaped across the corpus, the chunk artifacts, the Qdrant collection,
the hits the retrieve command emits, and — new in M4 — the answer, citations and
response body `POST /v1/query` returns. Scope: **M2 + M3 + M4**. Both named
vectors, dense and sparse, are written by the ingest job and read by the
retriever; a hit can carry what the rerank stage did to it; and a query response
carries an answer with `Citation` objects resolved from its `[n]` markers, or a
refusal. Nothing here is aspirational except where a field is explicitly marked
as belonging to a later milestone.

Three shapes, three lifetimes, and the difference matters:

| Shape | Lives | Stable across requests |
|---|---|---|
| Qdrant point | in the collection, until the next ingest | yes — `chunk_id` is the durable citation target |
| `RetrievalHit` | one retrieve command / one graph run | no — `rank` and `score` are per query |
| `QueryResponse` / `Citation` | one HTTP response | the `[n]` marker: **no**. The `chunk_id` inside the citation: yes |

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

## Citation

New in M4. A `Citation` is what an inline `[n]` marker in an answer resolves to:
the passage the model was shown, named in a way a reader can act on. It is
`Citation.to_dict()` in `production_rag.generation.citations`, field for field.

```jsonc
{
  "marker": 2,                     // the [n] this resolves; 1-based, ordinal into
                                   // the PROMPT BLOCKS this request rendered
  "chunk_id": "9f2c1a7b3d4e5f60:0003",
  "source_path": "sample/08-bm25-vs-dense.md",
  "title": "BM25 versus dense embeddings",
  "heading_path": "BM25 versus dense embeddings > What BM25 actually computes",
  "point_id": "6f3a1c2e-8b47-5d90-a1f2-0c9d7e4b3a15",
  "rank": 2,                       // its position in the retrieved ranking
  "score": 0.032258,               // the fused RRF score of that hit
  "text": "Three ideas, each doing one job: …"   // omit with include_text=False
}
```

| Field | Type | Why it exists |
|---|---|---|
| `marker` | int | binds the citation to the `[n]` in the answer text. **Request-scoped**: `[2]` identifies nothing outside this response |
| `chunk_id` | string | the durable citation target. This is what a client stores, re-resolves, or reports as a broken link after a re-chunk |
| `source_path` | string | the provenance a human reads — the file the claim came from |
| `title`, `heading_path` | string \| null | where in that file. `heading_path` is a rendered string, not an array |
| `point_id` | string | the Qdrant row, for anyone who needs to fetch the point itself |
| `rank` | int | position in the retrieved ranking, so "the model cited the 6th passage and ignored the 1st" is visible |
| `score` | float | the fused RRF score of that hit — a rank-derived number, not a similarity. See [`score_threshold`](architecture.md#score_threshold-on-a-fused-score-is-not-a-relevance-floor) |
| `text` | string | the passage exactly as the model saw it, so verification needs no second fetch. Droppable (`include_text=False`) for a caller that already holds the hits |

Ordering and duplication rules, because they are contract, not incidental:

- Citations appear in **first-appearance order in the answer**, not in retrieval
  order. Reading the answer top to bottom walks `citations[]` in order.
- A passage cited three times appears **once**. `marker` holds its number; the
  answer text keeps all three occurrences.
- A passage that was sent but never cited is **not** in `citations[]`. The array
  is what the answer used, not what retrieval found.
- A marker outside the range of rendered blocks is **stripped from the answer
  text** and listed in `invalid_markers`. It never becomes a `Citation`, and it
  is never left in place as a footnote that goes nowhere. See
  [ADR 0005](adr/0005-grounded-generation.md).
- Surviving markers are **not renumbered**. An answer citing only `[3]` keeps
  `[3]`, and its citation carries marker 3 — so the answer still lines up with
  the prompt that produced it.

### Markers index the prompt, not the retrieval result

`[3]` means "the third **block in the rendered prompt**", which is not the same
as the third retrieved hit whenever the context budget truncated. Resolution is
done against the blocks for exactly that reason: mapping against the retrieval
list would shift every marker by however many chunks did not fit, silently, and
in the direction that makes the citations look correct.

### `chunk_id` is durable; `[n]` is not

Change `top_k`, toggle rerank, or re-run the same question an hour later, and
`[3]` is a different passage. `chunk_id` survives all of that — it survives a
full re-ingest of unchanged content, because point ids and chunk ids are derived,
never random.

What it does **not** survive is a re-chunk. `chunk_id` is `<doc_id>:<index>`, so
changing `chunk_size` keeps the id and moves the text underneath it: a stored
citation then resolves successfully to different text, silently. That is the same
trap the eval labels face, described in
[chunk identity](#chunk-identity-is-derived-never-random), and it is why a client
archiving citations should archive `text` alongside `chunk_id`.

## QueryResponse — the shape `POST /v1/query` returns

New in M4. Four fields, whether the request produced an answer or a refusal:

```jsonc
{
  "answer": "RRF sums 1/(k + rank) over the branches that returned a document [1], which is why cosine and BM25 scores never have to be calibrated against each other [2].",
  "citations": [ /* Citation objects, first-appearance order */ ],
  "refused": false,
  "refusal_reason": null
}
```

A refusal is the same object, and it arrives with **HTTP 200**:

```jsonc
{
  "answer": "I could not find support for that in the indexed documents.",
  "citations": [],
  "refused": true,
  "refusal_reason": "no_evidence"
}
```

| Field | Note |
|---|---|
| `answer` | prose with `[n]` markers, or the configured refusal message. Markers resolve against `citations[]` and against nothing global |
| `citations` | `Citation` objects, minus nothing — the HTTP model carries `marker`, `chunk_id`, `source_path`, `text`, `rank`, `title`, `heading_path` |
| `refused` | the field a client branches on. Never string-match the message: it is config (`generation.citations.refusal_message`) and it is meant to change per deployment |
| `refusal_reason` | one of `no_evidence`, `model_abstained`, `no_citations`, `empty_answer`; `null` on a served answer. A **closed set**, so an operator can alert on one and an eval can group by them |

### Why 200, and why the body carries nothing else

**200 with `refused: true`.** Nothing failed — the corpus does not cover the
question and the system said so. Encoding that as 4xx or 5xx would make a correct
outcome indistinguishable from an outage in every dashboard the service ever
gets.

**No timings, no counts, no collection name, no token usage.** All of that exists
— on the library result, below — and is deliberately not projected onto the
endpoint. The body is the answer and its evidence; a public surface that reports
its own collection name, per-stage latencies and retrieval internals is
describing its interior to anyone who asks. The correlation id comes back on the
`X-Request-ID` header, which is what ties a response to the logs that hold the
rest.

### The request

```jsonc
{
  "question": "how does reciprocal rank fusion work",   // required, 1–8000 chars, stripped
  "mode": "hybrid",      // dense | sparse | hybrid — omitted uses the config default
  "rerank": "local",     // off | auto | fake | local | cohere — omitted uses the default
  "llm": "fake",         // fake | openai — DEFAULT fake: an unasked-for answer is never billed
  "debug": false         // ask the pipeline for diagnostics; the response shape is unchanged
}
```

Unknown fields are **rejected** (422), not ignored: a misspelled control that
silently falls back to a default answers a different question than the one asked,
and does it invisibly. A whitespace-only question is empty after stripping and is
rejected the same way — before retrieval, before any provider call.

## QueryResult — the library shape

`production_rag.query_pipeline.QueryResult.to_dict()` is what `run_query()`
returns and what the CLI prints. It is a superset of the HTTP response, and the
extra fields are the diagnostics:

```jsonc
{
  "query": "how does reciprocal rank fusion work",
  "answer": "…[1]…[2]",
  "refused": false,
  "refusal_reason": null,
  "citations": [ /* … */ ],
  "hits_used": 6,             // blocks that survived the context budget
  "hits_retrieved": 12,       // what retrieval handed over
  "model": "gpt-4o-mini",     // or the fake model id
  "mode": "hybrid",
  "collection": "production_rag",
  "embedded_model": "text-embedding-3-small",
  "rerank": {"applied": true, "reranker": "local", "candidates": 40, "error": null},
  "invalid_markers": [9],     // out-of-range [n] the model emitted; stripped from `answer`
  "uncited_claims": ["Both branches contribute independently."],
  "latency_ms": {"retrieve": 38, "rerank": 291, "generate": 1840, "cite": 3},
  "total_ms": 2172
}
```

| Field | What it answers |
|---|---|
| `hits_used` vs `hits_retrieved` | did the context budget truncate? Retrieval order is truncation order, so the dropped ones are the tail |
| `rerank` | byte-for-byte the M3 summary object, so a consumer that already parses retrieve-command output parses this unchanged |
| `invalid_markers` | is the model inventing citations? A steady non-empty list is a model or prompt problem, visible without reading answers |
| `uncited_claims` | citation coverage on **every** request rather than sampled by a judge. Reported, never fatal — see [ADR 0005](adr/0005-grounded-generation.md) |
| `latency_ms` | per graph node, not one number. "Which stage?" is the first question about a slow answer and a total cannot answer it. `total_ms` is their sum, a convenience and not a substitute |
| `model`, `collection`, `embedded_model` | which model answered, out of which collection, embedded by what — the three facts that make a result reproducible a week later |

Nothing in either shape is a credential, a prompt, or a raw provider payload. The
system prompt is not echoed, the API key is never serialised
(`Settings.safe_dump()` masks and a test asserts it), and an upstream error body —
which can quote the entire prompt back — stays in the server logs behind the
request id. The CLI's failure line prints the exception *type*, not its message,
for the same reason.

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
