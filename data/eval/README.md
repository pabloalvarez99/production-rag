# `data/eval/` — golden evaluation dataset

Hand-labelled questions used to gate changes. Committed and versioned with the
code: an eval dataset that lives outside the repo drifts away from the system it
measures. See [`docs/evaluation.md`](../../docs/evaluation.md) for how it is
scored and [ADR-0003](../../docs/adr/0003-eval-strategy.md) for why the strategy
is split in two tiers.

## `golden.jsonl`

One JSON object per line. JSONL rather than a single JSON array so a malformed
entry breaks one line instead of the whole file, and diffs stay readable.

The file is committed and currently holds the **M1 seed set**: 12 items labelled
at *document* granularity. The chunk-level schema below is the M2 target, not
what is in the file today — see [M1 seed schema](#m1-seed-schema-current) first.

```jsonc
{
  "id": "q-0001",
  "question": "How does hybrid retrieval combine dense and sparse results?",
  "relevant_chunk_ids": ["9f2c1a7b3d4e5f60:0003", "9f2c1a7b3d4e5f60:0004"],
  "relevance": {"9f2c1a7b3d4e5f60:0003": 2, "9f2c1a7b3d4e5f60:0004": 1},
  "expected_answer": "Both branches are queried, then fused with reciprocal rank fusion.",
  "answerable": true,
  "category": "conceptual",
  "tags": ["retrieval", "fusion"],
  "notes": "Baseline case: the answer sits in one chunk."
}
```

| Field | Type | Required | Meaning |
|---|---|---|---|
| `id` | string | yes | stable identifier; never reused after deletion |
| `question` | string | yes | the query as a user would phrase it |
| `relevant_chunk_ids` | string[] | yes when `answerable` | ground truth for retrieval metrics; `[]` for unanswerable items |
| `relevance` | object | no | graded relevance per chunk (`2` = fully answers, `1` = partial). Needed for `ndcg`; without it every listed chunk is treated as `1` |
| `expected_answer` | string | no | reference answer shown to the LLM judge |
| `answerable` | bool | yes | `false` means the correct behaviour is a refusal |
| `category` | enum | yes | `conceptual` \| `exact_token` \| `multi_hop` \| `unanswerable` |
| `tags` | string[] | no | free-form slicing for per-topic reports |
| `notes` | string | no | why this item exists; read by whoever debugs its failure |

## M1 seed schema (current)

M1 ingests the corpus; it does not serve queries. Chunk ids therefore exist but
nothing has ever retrieved one, and labelling against them now would produce
labels invalidated by the first chunking change (see
[`sample/05-chunking-pitfalls.md`](../raw/sample/05-chunking-pitfalls.md)).

So the seed set labels the **source document** instead. It is a weaker signal —
document-level `hit@k`, not chunk-level `recall@k` — and it is a signal that
survives re-chunking, which is what makes it worth writing before the retriever
exists.

```jsonc
{
  "id": "q-0006",
  "question": "What do the k1 and b parameters control in the BM25 scoring formula?",
  "expected_source_paths": ["sample/08-bm25-vs-dense.md"],
  "category": "exact_token",
  "notes": "Single-character identifiers. Dense retrieval is expected to lose this one badly."
}
```

| Field | Type | Required | Meaning |
|---|---|---|---|
| `id` | string | yes | stable identifier; never reused after deletion |
| `question` | string | yes | the query as a user would phrase it |
| `expected_source_paths` | string[] | yes | paths relative to `data/raw/`; `[]` for unanswerable items |
| `category` | enum | yes | `conceptual` \| `exact_token` \| `multi_hop` \| `unanswerable` |
| `notes` | string | no | why this item exists; read by whoever debugs its failure |

Every `expected_source_paths` entry must resolve to a committed file under
`data/raw/`. A path that does not exist scores as a permanent miss and looks
exactly like a retrieval regression.

Migrating to the chunk-level schema means adding `relevant_chunk_ids` alongside
`expected_source_paths`, not replacing it: the document-level labels stay useful
as the coarse check that survives the next re-chunk.

## Composition targets

| Category | Share | What it catches |
|---|---|---|
| `conceptual` | 40% | paraphrase; the case dense retrieval should win |
| `exact_token` | 25% | IDs, error codes, function names; the case sparse should win |
| `multi_hop` | 20% | answers needing 2+ chunks; exposes context-budget truncation |
| `unanswerable` | 15% | the refusal path |

Minimum useful size is 50 items. Below that, one item moves `recall@5` by two
points and the metric stops being a signal.

The committed M1 seed set is 12 items and is therefore **not** a merge gate. It
exists to pin the schema, to prove the corpus is reachable end to end, and to be
the thing that gets extended rather than invented under deadline. Treat its
numbers as a smoke test, never as a quality claim.

## Authoring rules

- **Label chunks, not documents.** `chunk_id` labels are what make `recall@k`
  meaningful. They are also coupled to the current chunking config — a chunk
  size change invalidates them.
- **Write the question first, find the chunk second.** Reading a chunk and then
  writing a question about it produces questions that use the chunk's own
  vocabulary, which flatters sparse retrieval and measures nothing.
- **Never include a real secret, customer name, or credential.** This file is
  committed.
- **One behaviour per item.** An item that fails should point at one cause.

## Results

Run outputs go to `data/processed/eval-runs/<timestamp>/` (gitignored), never
here. This directory is read-only at runtime.
