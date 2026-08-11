# `data/eval/` — golden evaluation dataset

Hand-labelled questions used to gate changes. Committed and versioned with the
code: an eval dataset that lives outside the repo drifts away from the system it
measures. See [`docs/evaluation.md`](../../docs/evaluation.md) for how it is
scored and [ADR-0003](../../docs/adr/0003-eval-strategy.md) for why the strategy
is split in two tiers.

## `golden.jsonl`

One JSON object per line. JSONL rather than a single JSON array so a malformed
entry breaks one line instead of the whole file, and diffs stay readable.

Use `golden.jsonl` with the nine-document `data/raw/sample/` corpus for fast,
offline CI and plumbing checks. Its 17 seed items pin the schema and exercise
the evaluator; they are not a retrieval-quality benchmark.

## `golden-corpus.jsonl`

Use `golden-corpus.jsonl` with an ingest rooted at `data/corpus/`. It contains
60 source-labelled items over the vendored Qdrant documentation, exactly ten in
each adversarial category: `lexical_only`, `paraphrase_only`, `multi_source`,
`distractor`, `near_miss_unanswerable`, and `deep_rank`. The original
`corpus-dist-009` was removed because its supposed wrong document also contained
the answer; `corpus-dist-011` replaces it with a measured rank-1/rank-2 genuine
competition rather than reusing the retired id or padding with a weak case.

The larger corpus exists to make source-level `hit@5` discriminating. With nine
sources, five random distinct results hit a labelled source about 56% of the
time; with 3,067 source documents the same random coverage is about 0.16%.
Always pair the golden file with its intended ingest root because source-path
matching is exact:

| Golden file | Ingest root | Purpose |
|---|---|---|
| `golden.jsonl` | `data/raw/` | deterministic CI fixture and smoke test |
| `golden-corpus.jsonl` | `data/corpus/` | retrieval measurement and slice analysis |

The corpus golden set does not imply a quality result by itself. No paid
embedding or hosted judge baseline was run when it was authored.

`paraphrase_only` has a mechanical invariant: after the production BM25
tokenizer lowercases and removes English stopwords, a query shares zero terms
with every full document named by `expected_source_paths`. Checking the whole
target document is deliberately stronger than checking only its answering
chunk, and `scripts/check_golden_integrity.py` enforces it.

The file is committed and currently holds the **seed set**: 17 items labelled at
*document* granularity. The chunk-level schema below is the target, not what is
in the file today — see [seed schema](#seed-schema-current) first.

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

## Seed schema (current)

M2 serves retrieval, but chunk-level labels still wait: nothing has yet been
tuned against a measurement, so chunk size and overlap remain the settings most
likely to move, and labelling against chunk ids now would produce labels
invalidated by the first chunking change (see
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
  "answerable": true,
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
| `answerable` | bool | no (written on every item) | `false` means the correct behaviour is a refusal |
| `notes` | string | no | why this item exists; read by whoever debugs its failure |

### Optional fields, and which reader actually consumes them

Three fields in the target schema are optional in the sense that the tier-1
retrieval scorer runs without them. They are not decorative — each one is the
sole input to a metric that cannot be computed otherwise, which is why the table
below names the consumer rather than saying "future use":

| Field | Consumed by | What is lost without it |
|---|---|---|
| `answerable` | the refusal slice of tier 2 (`refusal_accuracy`) | nothing today; the tier-1 scorer derives scorability from an empty `expected_source_paths` instead — see below |
| `relevance` | `ndcg@k` | grading collapses: every listed chunk is treated as relevance `1`, so a run cannot distinguish "the fully answering chunk is at rank 1" from "the partial one is" |
| `expected_answer` | the tier-2 judge | the judge scores `faithfulness` against the cited passages alone, with no reference answer to compare against |

`relevance` and `expected_answer` belong to the chunk-level schema and are absent
from every item in the file today. `answerable` **is** written on every item, and
the reason it is still marked optional is worth stating precisely, because the
two facts look contradictory:

- **`production_rag.evals.source_hit` never reads it.** `GoldenCase.from_json`
  parses `id`, `question`, `expected_source_paths` and `category`; `is_scorable`
  returns `bool(self.expected_source_paths)`. Unanswerable items are excluded
  from the aggregate because their label list is empty, not because a flag says
  so. Adding the field changed no number — it was verified against the loader,
  which still reports 13 scorable and 4 unscored.
- **It is written anyway, on every item, because the invariant is the point.**
  `answerable == bool(expected_source_paths)` is now checkable by inspection
  rather than inferable. An item labelled `unanswerable` that carries a source
  path, or an answerable item whose labels were dropped in a bad merge, reads as
  a contradiction on the line instead of silently leaving the aggregate.

So the field is redundant for tier 1 by design and load-bearing for tier 2: an
`unanswerable` item is the only kind whose *correct* outcome is a refusal, and
`refusal_accuracy` needs to know which items those are without inferring it from
the absence of something.

Every `expected_source_paths` entry must resolve to a committed file under
`data/raw/`. A path that does not exist scores as a permanent miss and looks
exactly like a retrieval regression. `scripts/eval_hit.py` reports unresolvable
labels separately from misses for that reason — one is a dataset bug, the other
is a result.

### Path matching is exact, and the ingest root decides it

`scripts/eval_hit.py` compares `expected_source_paths` against the `source_path`
field of each returned hit as **exact strings**. Both sides are relative to the
corpus root that ingest walked, which makes the ingest `SOURCE` argument part of
this dataset's contract rather than an operator preference:

| Ingested with | Stored `source_path` | Labels here | Result |
|---|---|---|---|
| `SOURCE=data/raw` (default) | `sample/08-bm25-vs-dense.md` | `sample/08-bm25-vs-dense.md` | matches |
| `SOURCE=data/raw/sample` | `08-bm25-vs-dense.md` | `sample/08-bm25-vs-dense.md` | every label misses |

The second row scores `hit@k = 0.00` at every k — shaped exactly like total
retrieval failure, caused by nothing but a path prefix. It is the first thing to
check when the number is a flat zero.

Paths use forward slashes on every platform, including Windows: ingest
normalises separators before hashing, so a document ingested on the host and
inside the Linux container produces the same `doc_id` and the same
`source_path`. Write labels with `/`.

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

The committed seed set is 17 items and is therefore **not** a merge gate. One
item moves `hit@5` by roughly eight points. It exists to pin the schema, to prove
the corpus is reachable end to end, and to be the thing that gets extended rather
than invented under deadline. Treat its numbers as a smoke test, never as a
quality claim.

### Where the seed set stands against those targets

Measured on the committed file, not aspired to:

| Category | Target | Seed set | Reading |
|---|---|---|---|
| `conceptual` | 40% | 7 / 17 (41%) | on target |
| `exact_token` | 25% | 4 / 17 (24%) | on target |
| `multi_hop` | 20% | 2 / 17 (12%) | **under** — the context-budget slice is the thin one, and it is the one that catches truncation |
| `unanswerable` | 15% | 4 / 17 (24%) | deliberately over, see below |

The two deviations have different causes and only one of them is a decision.

**`unanswerable` is over target on purpose.** Fifteen percent of a 17-item set is
two and a half items. A refusal slice of two cannot distinguish "the system
refuses correctly" from "the system refused twice" — a single flip is fifty
percent of the slice. M6 raised it to four (`q-0012`, `q-0015`, `q-0016`,
`q-0017`) so that the slice has enough items to show a pattern. That inflates the
share now and self-corrects as the set grows toward 50, at which point the target
share is what should be honoured. Four items is still too few to quote a
`refusal_accuracy` figure from; it is enough to see the refusal path fire, or
fail to.

**`multi_hop` being under target is a gap, not a choice.** It is recorded here so
the next person extending the set knows which slice to write first.

### The unanswerable items are adjacent, not off-topic

The tempting way to write an unanswerable item is to ask about something the
corpus has never heard of. Such an item measures almost nothing: retrieval
returns low-scoring junk, `score_threshold` filters it, and the system refuses
via `no_evidence` without the model ever being called. That tests a config
constant.

The four committed items are all *topically adjacent* instead — they use the
corpus's own vocabulary and retrieve confident, well-scoring, genuinely relevant
chunks that simply do not contain the fact asked for:

| Item | Asks for | Why retrieval succeeds and the answer still must not exist |
|---|---|---|
| `q-0012` | a concurrency limit | the corpus discusses latency and cost profiles; it states no capacity figure |
| `q-0015` | a dollar cost per month | `08-bm25-vs-dense.md` compares cost profiles at length and `00-intro.md` names embedding as the paid step — no document carries a currency figure |
| `q-0016` | a Qdrant version number | `Qdrant` and `sparse vectors` are corpus vocabulary, so the sparse branch ranks `06-qdrant-payloads.md` high; no document states a version |
| `q-0017` | a multilingual model recommendation | embeddings are discussed throughout, language coverage nowhere |

`q-0017` differs in mechanism from the other three and that is why it exists: the
first three ask for a *fact the corpus omits*, it asks for a *recommendation the
corpus never makes*. The second is easier to hallucinate, because no specific
missing token gives the gap away.

The failure these catch is the one that matters: a model answering from
pretraining once retrieval has handed it a plausible chunk to cite. On these
items a refusal is the correct output and a fluent, cited answer is the defect.

M2 added two `exact_token` items (`q-0013`, `q-0014`) whose questions carry rare
literal strings that appear verbatim in exactly one corpus document. They are
the sparse branch's job description: the dense branch has nothing useful to say
about a hyphenated model name or a config key, so a hit ranked by `sparse` and
not by `dense` on those items is the hybrid claim being demonstrated rather than
asserted. They are also the two items that stay meaningful on a `fake`-embedded
collection, since BM25 weights are computed from the text and never touch the
embedder.

Both tokens (`bge-reranker-base`, `full_scan_threshold`) survive the tokenizer
intact — it keeps internal `-`, `_` and `.` joined rather than splitting them
into fragments, which is the whole point of having a lexical branch.

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
