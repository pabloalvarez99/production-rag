# Evaluation

A RAG system without offline evaluation is a system where every change is a
guess. This document defines what is measured, how, and what number blocks a
merge. The strategy rationale is in [ADR-0003](adr/0003-eval-strategy.md).

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
stops being a signal.

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
