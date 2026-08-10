# ADR 0003 — Two-tier evaluation: free retrieval gate, sampled answer judge

- **Status:** Proposed
- **Date:** 2026-08-10
- **Deciders:** production-rag maintainers
- **Supersedes:** —

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

**Tier 1 — retrieval, deterministic and free.** Compare returned `chunk_id`s
against hand-labelled `relevant_chunk_ids`. Metrics: `recall@k`, `mrr`,
`ndcg@k` at k ∈ {1, 3, 5, 10}, reported **per branch** (`dense`, `sparse`,
`fused`) as well as overall. No LLM in the loop, so it is fast, repeatable, and
runnable on every change. `recall@5 ≥ 0.80` is the headline gate: a chunk not in
the top 5 cannot be recovered by any downstream stage.

**Tier 2 — answer quality, sampled and judged.** An LLM judge (`gpt-4o`) scores
`faithfulness`, `answer_relevance`, `citation_precision`, and
`refusal_accuracy`, over a bounded sample (default 50) and **only on queries
whose retrieval succeeded** — otherwise the judge is grading the retriever. Gate:
`faithfulness ≥ 0.90`.

Supporting decisions:

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

- Hand-labelling `relevant_chunk_ids` is real work, and the labels are coupled
  to the current chunking strategy — a chunk-size change invalidates chunk-level
  labels and requires relabelling or a document-level fallback metric.
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
