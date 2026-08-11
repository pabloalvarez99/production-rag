# Evaluation

A RAG system without offline evaluation is a system where every change is a
guess. This document defines what is measured, how, and what number blocks a
merge. The strategy rationale is in [ADR-0003](adr/0003-eval-strategy.md).

## Status after M1

Nothing here runs yet. M1 built the ingest path; there is no query path, so
there is nothing to score.

| Piece | State |
|---|---|
| `data/eval/golden.jsonl` | **committed** — 12-item seed set, document-level labels |
| Retrieval metrics (`hit@k` over source paths) | possible once M2 lands a retriever |
| Retrieval metrics (`recall@k` over chunk ids) | needs chunk-level labels; M2 |
| Ragas / LLM-judge answer metrics | needs generated answers; M4 at the earliest |
| Merge gate in CI | M6 |

Everything below the next section is the target design. Nothing in this repo has
produced a measured retrieval or answer number, and no such number should be
quoted until the harness exists.

## What the M1 seed set is for

Twelve items is far too few to gate a merge, and it is not trying to. It exists
to do three things now, so that none of them has to be invented later under
deadline:

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
chunking settles in M2, `relevant_chunk_ids` is added *alongside*
`expected_source_paths`, not instead of it: the coarse labels stay useful as the
check that survives the next re-chunk.

## Target design (M2 onward)

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
stops being a signal. The committed seed set is 12 items and is explicitly not a
gate — see [Status after M1](#status-after-m1).

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
| `recall@5` drops, `mrr` stable | fewer relevant chunks retrieved overall | chunking size/overlap, `dense_top_k` |
| `recall@5` stable, `mrr` drops | ordering degraded | fusion weights, `rrf.k` |
| Sparse branch collapses | tokenisation or stopwords changed | `ingest.sparse` block |
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
