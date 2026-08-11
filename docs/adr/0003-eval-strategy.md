# ADR 0003 — Two-tier evaluation: free retrieval gate, sampled answer judge

- **Status:** Accepted — both tiers implemented in M6, at source granularity
- **Date:** 2026-08-10 (proposed), 2026-08-11 (accepted on M6 landing)
- **Deciders:** production-rag maintainers
- **Supersedes:** —
- **Relates to:** [ADR 0001](0001-hybrid-qdrant.md) (the fusion constants tier 1
  turns into an empirical question), [ADR 0004](0004-rerank-cross-encoder.md)
  (the stage whose value is a delta, so it needs a before and an after),
  [ADR 0005](0005-grounded-generation.md) (the refusal behaviour the
  `unanswerable` slice measures), [ADR 0006](0006-observability.md) (the ops
  signals this ADR insists are not eval metrics)

## Context

Every meaningful change to a RAG system — chunk size, embedding model, fusion
constants, prompt wording — moves quality in a direction nobody can predict by
reading the diff. Without measurement, the project degrades into taste.

The obvious approach, an end-to-end LLM-judged answer score, has three defects:

1. **It costs money per run**, so it gets run rarely, so regressions are found
   late and in batches where they cannot be attributed to a single change.
2. **It is non-deterministic**, so a two-point move is indistinguishable from
   judge noise.
3. **It collapses the two failure modes.** A wrong answer caused by a missing
   chunk and a wrong answer caused by a model ignoring a present chunk get the
   same score, and the score points at the prompt either way — which is the
   wrong place to look half the time.

## Decision

Adopt a **two-tier evaluation strategy** over one golden dataset
(`data/eval/golden.jsonl`).

**Tier 1 — retrieval, deterministic and free.** Compare returned `source_path`s
against hand-labelled `expected_source_paths`. Metrics: `source_hit@k`,
`source_recall@k`, `mrr`, and binary-gain `ndcg@k`, reported over the selected
retrieval mode; branch ablation compares `dense`, `sparse`, `fused`, and
reranked runs separately. No LLM is in the loop, so the arithmetic is fast and
repeatable. The implemented `--fail-under-hit` gate is opt-in and defaults to
reporting only. A chunk-level gate waits for chunk-level labels and a measured
baseline.

**Tier 2 — answer behaviour, sampled and optionally judged.** Every selected
case runs through `run_query`. Citation precision at document granularity,
invalid-marker rate, and refusal accuracy are deterministic and judge-free;
an `AnswerJudge` adds faithfulness and relevance. The offline default is a
lexical `FakeJudge`, while the hosted judge is explicitly gated and remains
uncalibrated. Tier 2 reports every selected case, including retrieval misses,
and has no armed gate.

Supporting decisions:

<!-- provenance-allow: historical-measurement: target mix fixed when ADR-0003 was accepted, not a current result -->
- **The dataset is stratified**, not just collected: 40% paraphrase, 25% exact
token, 20% multi-hop, 15% unanswerable. The unanswerable slice is mandatory —
  it is the only measurement of whether the system hallucinates under pressure,
  and a system that never refuses scores perfectly without it.
- **Fixed seed (42)** and versioned dataset, so two runs are comparable.
- **The judge is calibrated once** against a 20-item hand-labelled subset, and
  re-calibrated whenever the judge model changes. An uncalibrated judge produces
  a number, not a measurement.
- **Thresholds are set from the first baseline run**, not from ambition.
  Lowering one to make a build pass requires a new ADR.

## Consequences

**Positive**

- The cheap tier runs on every change, so a regression is attributable to one
  commit instead of to a week.
- Retrieval and generation regressions are distinguishable, which means the fix
  gets attempted in the right subsystem.
- Per-branch retrieval metrics turn ADR-0001's fusion constants into an
  empirical question rather than an argument.
- The refusal path has a number attached to it, so "safe by default" is a
  measured property.

**Negative**

- Source-level labels are coarser than passage relevance: the right document can
  be retrieved or cited for the wrong reason. Adding `relevant_chunk_ids` later
  is real work, and those labels will be coupled to the chunking strategy.
- 50 golden queries is small. It is enough to catch a collapse and not enough to
  resolve a two-point difference; treat small moves as noise until the dataset
  grows.
- The judge is a proxy for human preference and inherits the judge model's
  biases, notably a preference for verbosity.
- Two tiers means two commands and two sets of results to reconcile.

**Neutral / follow-ups**

- Cost per query and latency under concurrency are explicitly deferred; see
  [evaluation.md](../evaluation.md).
- If the golden set reaches ~200 items, revisit whether the judge sample can
  shrink further without losing the ability to detect a real regression.

## Implementation status at acceptance (M6)

This ADR is accepted as the strategy, and M6 implemented both tiers behind one
runner (`python -m production_rag.evals.run`). Where the implementation departs
from the Decision above, it is recorded here rather than discovered later by
someone who trusted that section:

<!-- provenance-allow: historical-measurement: milestone implementation record preserved with ADR-0003 -->
| Decided above | What M6 actually landed |
|---|---|
| Tier 1 metrics `recall@k`, `mrr`, `ndcg@k` over `relevant_chunk_ids` | **the same arithmetic over `expected_source_paths`** — `source_hit_at_k`, `source_recall_at_k`, `mrr`, `ndcg_at_k` in `evals.tier1_retrieval`. Document granularity, and the names say `source_` for that reason. `ndcg` uses binary gain: the graded `relevance` field is on no item |
| Per-branch reporting (`dense`, `sparse`, `fused`) | **landed** — `evals.ablation`, which also runs the fused set through the fake reranker |
| Tier 2: `faithfulness`, `answer_relevance`, `citation_precision`, `refusal_accuracy`, judged | **landed, and split by cost.** `citation_precision`, `invalid_marker_rate` and `refusal_accuracy` are judge-free and deterministic; `faithfulness` and `relevance` come from an `AnswerJudge` whose default is offline and lexical. `citation_precision` is document-level, so it checks the citation's source and not its support |
| Judged only where retrieval succeeded | **not implemented.** `evaluate_tier2` answers every case in the sample. Acceptable while the per-case rows are read individually; not acceptable once an aggregate is quoted, because a `faithfulness` mean over items whose document was never retrieved grades the retriever |
| Sampled (default 50), fixed seed | **landed** — `--sample` / `--seed 42`, both echoed in the report. The default is *every* case, because 17 < 50 |
| Judge calibrated once against 20 hand-labelled items | **not done.** The hosted judge exists behind `--judge openai`, `RUN_LLM_EVALS=1` and a credential; no calibration subset has been labelled, so its output is a number and not a measurement |
| Stratified dataset, 15% unanswerable | **partly.** 17 items: 41% conceptual, 24% exact-token, 12% multi-hop (under target), 24% unanswerable (over target on purpose — 15% of 17 is two items, too few for the slice to show a pattern) |
| `recall@5 ≥ 0.80` and `faithfulness ≥ 0.90` as gates | **replaced by one opt-in flag.** `evals.thresholds` in `configs/default.yaml` is read by nothing. The only gate is `--fail-under-hit`, on `source_hit_at_k`, default `0.0` |

Four corrections follow, and they are binding on whoever arms the gate:

1. **The thresholds in `configs/default.yaml` are placeholders.** The Decision
   above says thresholds come from the first baseline run. No baseline run has
   been performed, so the committed values satisfy the letter of nothing. They
   are replaced by measured numbers, not adjusted toward them.
2. **The gate is deliberately on the deterministic metric.** `--fail-under-hit`
   scores `source_hit_at_k` and nothing else. Gating a judged column would make a
   build outcome depend on a proxy this ADR requires to be calibrated first.
3. **A gate on 17 items would fire on noise.** One item is roughly eight points
   of source `hit@k`. The dataset reaching ~50 items is a precondition for the
   gate, not a nice-to-have alongside it.
4. **The retrieval-succeeded filter is outstanding work, not a nuance.** Until it
   lands, tier 2 aggregates are read against tier 1's `misses` from the same run
   — which is why both tiers report into one file.

Chunk-level labels remain the blocker for the tier-1 metrics as specified. They
wait on chunking settling, because a `relevant_chunk_ids` label survives a
chunk-size change by silently repointing at different text — the failure mode
that is worse than breaking loudly. The metric functions do not change when the
labels arrive; the names lose their `source_` prefix.
