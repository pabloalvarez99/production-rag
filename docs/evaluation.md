# Evaluation

A RAG system without offline evaluation is a system where every change is a
guess. This document defines what is measured, how, and what number blocks a
merge. The strategy rationale is in [ADR-0003](adr/0003-eval-strategy.md).

## Status after M2

M2 landed a retriever, so the first measurement in this repository is now
possible: **source-level `hit@k`**. Everything else is still design.

| Piece | State |
|---|---|
| `data/eval/golden.jsonl` | **committed** — 14-item seed set, document-level labels |
| Retrieval metrics (`hit@k` over source paths) | **runnable** — `scripts/eval_hit.py`, `make eval-hit-fake` |
| Retrieval metrics (`recall@k`, `mrr`, `ndcg@k` over chunk ids) | needs chunk-level labels; not written yet |
| Ragas / LLM-judge answer metrics | needs generated answers; M4 at the earliest |
| `production_rag.evals` harness | M6 |
| Merge gate in CI | M6 |

`scripts/eval_hit.py` is deliberately a **script, not a harness**: stdlib only,
one metric, no thresholds, no gate. It exists so M2 can be checked end to end
today without pre-building M6's abstractions around a metric that may not survive
the move to chunk-level labels.

### Reading a `hit@k` number honestly

`hit@k` asks one question: *did any chunk from a labelled document appear in the
top k?* It is coarse by construction — a document can be retrieved for the wrong
reason and still count.

What the number means depends entirely on which embedder built the collection,
and the two cases are not comparable:

| Collection built with | What `hit@k` measures | What it does **not** measure |
|---|---|---|
| `--embedder fake` | that the plumbing works end to end: the corpus is indexed, both branches return, fusion orders, payload paths match the labels. The sparse branch is genuinely lexical, so exact-token items can legitimately hit. | anything about dense retrieval quality. The dense branch contributes hash noise, and a conceptual/paraphrase item hitting is luck. |
| `--embedder openai` | retrieval quality on this corpus, at this chunking, with this fusion config — a real number for a 14-item set, which is to say a smoke test with error bars measured in whole documents. | statistical significance. One item moves `hit@5` by seven points. |

So: a `fake`-path `hit@k` is a **plumbing assertion**, and the only honest way to
report it is alongside the embedder that produced it. Neither number belongs in a
README as a quality claim. See [ADR-0003](adr/0003-eval-strategy.md).

The per-branch breakdown is worth more than the headline at this size. `hit@k`
on the sparse branch alone, over the `exact_token` slice, is a direct test of the
thing M2 was built for — and it is meaningful even on a `fake` collection.

## What the seed set is for

Fourteen items is far too few to gate a merge, and it is not trying to. It
exists to do three things now, so that none of them has to be invented later
under deadline:

1. **Pin the schema.** Downstream code can be written against a real file.
2. **Prove the corpus is reachable.** Every `expected_source_paths` entry
   resolves to a committed file under `data/raw/`; a path that does not exist
   scores as a permanent miss and reads exactly like a retrieval regression.
3. **Write the questions before the retriever exists.** Reading a chunk and then
   writing a question about it produces questions phrased in the chunk's own
   vocabulary, which flatters sparse retrieval and measures nothing. Authoring
   ahead of the retriever removes that temptation structurally.

### Document labels first, chunk labels later

The seed set names source paths, not chunk ids. Chunk ids exist after ingest,
but a chunk id is `<doc_id>:<index>` — so changing `chunk_size` or
`chunk_overlap` does not invalidate a label, it quietly repoints it at a
different passage. A label that breaks loudly is recoverable; one that keeps
scoring against the wrong text is not. M1 is exactly the milestone in which
those chunking values are most likely to move.

Document-level labels support `hit@k` — did any chunk from the right document
reach the top k — which is a coarser metric than `recall@k` and a real one. When
chunking settles, `relevant_chunk_ids` is added *alongside*
`expected_source_paths`, not instead of it: the coarse labels stay useful as the
check that survives the next re-chunk. M2 did not settle it — chunk size and
overlap are unchanged, but nothing has yet been tuned against a measurement, so
the chunk-level labels still wait.

### Labels match on `source_path`, exactly

The comparison `scripts/eval_hit.py` performs is a string match between
`expected_source_paths` and the `source_path` field of each returned hit. Both
sides must be relative to the same corpus root.

That makes the ingest `SOURCE` argument part of the eval contract, not an
operator preference. Ingesting `data/raw` stores `sample/08-bm25-vs-dense.md`,
which is what the labels say. Ingesting `data/raw/sample` stores
`08-bm25-vs-dense.md`, every label misses, and `hit@k` reads `0.00` — identical
in shape to a total retrieval failure and caused by nothing but a path prefix.

`scripts/eval_hit.py` reports unmatched-label paths separately from misses for
exactly this reason: a label whose file does not exist under `data/raw/` is a
dataset bug, not a retrieval result.

## Running `hit@k` today

```bash
make reingest-fake      # rebuild the collection with sparse vectors (M2 needs this)
make eval-hit-fake      # source-level hit@k over data/eval/golden.jsonl
```

```powershell
.\scripts\eval_hit.ps1                    # same, Windows without make
.\scripts\eval_hit.ps1 -Embedder openai   # real embeddings; costs money
```

Output is a per-k table plus a per-category and per-branch breakdown, and a JSON
object on the last line of stdout for anything that wants to parse it. There is
no threshold and no non-zero exit on a low score — this script reports, it does
not gate. Gating is M6, and gating a 14-item set would be theatre.

## Target design (M6 onward)

## Principle: separate the two failure modes

A wrong answer has exactly two causes, and they need different fixes:

1. **Retrieval failed** — the supporting chunk was never handed to the model.
   No prompt change fixes this.
2. **Generation failed** — the chunk was present and the model ignored,
   misread, or embellished it.

Measuring only end-to-end answer quality collapses these into one number and
sends you tuning prompts when the real defect is chunk size. So retrieval is
scored first, independently, and answer quality is scored only over queries
whose retrieval succeeded.

## Golden dataset

`data/eval/golden.jsonl` — one JSON object per line. Field spec in
[`data/eval/README.md`](../data/eval/README.md). Minimum viable set is 50
queries; below that, a single item moves recall@5 by two points and the metric
stops being a signal. The committed seed set is 14 items and is explicitly not a
gate — see [Status after M2](#status-after-m2).

Composition targets:

| Slice | Share | Why it exists |
|---|---|---|
| Paraphrase / conceptual | 40% | the case dense retrieval is supposed to win |
| Exact token (IDs, names, codes) | 25% | the case sparse retrieval is supposed to win |
| Multi-hop (needs 2+ chunks) | 20% | exposes context-budget truncation |
| Unanswerable | 15% | the refusal path; without these, a model that never refuses scores well |

The unanswerable slice is the one most often skipped and the most diagnostic:
it is the only measurement of whether the system hallucinates under pressure.

## Retrieval metrics

Computed by comparing returned `chunk_id`s against the golden
`relevant_chunk_ids`. No LLM involved, so runs are free, fast, and
deterministic.

| Metric | Question it answers |
|---|---|
| `recall@k` | did the supporting chunk make it into the top k at all? |
| `mrr` | how far down the list was the first good chunk? |
| `ndcg@k` | is the ordering good, weighting graded relevance? |

`recall@5` is the headline. If a chunk is not in the top 5, no reranker or
prompt recovers it, and everything downstream is capped by that number.

Report retrieval metrics per branch as well as fused — `dense`, `sparse`,
`fused`. A fused score that never beats its best branch means the fusion
constants need work, not the retriever.

## Answer metrics

Scored by an LLM judge (`gpt-4o`), sampled to bound cost. Judged only where
retrieval succeeded.

| Metric | Definition |
|---|---|
| `faithfulness` | every claim in the answer is supported by a cited chunk |
| `answer_relevance` | the answer addresses the question actually asked |
| `citation_precision` | cited chunks genuinely support the sentences citing them |
| `refusal_accuracy` | refuses on unanswerable items, does not refuse on answerable ones |

Faithfulness is the metric that matters most: an unfaithful answer with
confident citations is worse than no answer, because it survives review.

## Thresholds

From `evals.thresholds` in `configs/default.yaml`. A run fails if any is missed:

| Metric | Threshold |
|---|---|
| `recall@5` | ≥ 0.80 |
| `mrr` | ≥ 0.65 |
| `faithfulness` | ≥ 0.90 |

These are starting values, set from the first full baseline run rather than
from ambition. Raise them as the system improves; lowering one to make a build
pass is a decision that belongs in an ADR, not in a commit.

## Running an evaluation

> `production_rag.evals` does not exist yet. The commands below are the agreed
> interface for M6, recorded here so the harness is written to a stated contract
> rather than discovered. Running them today fails with a module-not-found
> error, which is the correct outcome.

```bash
# retrieval only — free, deterministic, run on every change
docker compose exec api python -m production_rag.evals --suite retrieval

# with the LLM judge — costs money, run before a release
docker compose exec api python -m production_rag.evals --suite all --sample 50
```

Results land in `data/processed/eval-runs/<timestamp>/` (gitignored) as
`summary.json` plus a per-query `details.jsonl`. `seed: 42` in config keeps two
runs over the same dataset comparable.

## Interpreting a regression

| Symptom | Likely cause | Where to look |
|---|---|---|
| `hit@k` is exactly `0.00` at every k | labels and payload paths disagree | the ingest `SOURCE` root — see [labels match on `source_path`](#labels-match-on-source_path-exactly) |
| `recall@5` drops, `mrr` stable | fewer relevant chunks retrieved overall | chunking size/overlap, `dense_top_k` |
| `recall@5` stable, `mrr` drops | ordering degraded | fusion weights, `rrf.k` |
| Sparse branch collapses | tokenisation or stopwords changed | `ingest.sparse` block |
| Sparse branch quietly returns nothing | collection predates M2, no `sparse` vector | rebuild: `make reingest-fake` |
| Exact-token items regress after adding documents | IDF drift since the last full ingest | full re-ingest, then re-measure |
| `faithfulness` drops, retrieval flat | prompt or model change | `generation.prompt`, model version |
| Refusals spike | `score_threshold` raised too far | `retrieval.score_threshold` |

Change one variable per run. Two changes and a moved metric produce a story,
not a finding.

## What is deliberately not measured yet

- **Latency under concurrency.** Single-request latency is reported per stage
  in every response; load testing waits until the deployment target is real.
- **Cost per query.** Token usage is in the response payload; aggregating it
  into a per-query cost dashboard is a later milestone.
- **Human preference.** The LLM judge is a proxy. It is calibrated against a
  20-item hand-labelled subset once, and re-calibrated when the judge model
  changes — an uncalibrated judge is a number, not a measurement.
