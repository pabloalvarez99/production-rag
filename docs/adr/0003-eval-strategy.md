# ADR 0003: Evaluation strategy — versioned dataset, layered metrics, baseline deltas

Status: Proposed
Date: 2026-08-10

## Context

RAG systems regress silently: a chunking tweak, an embedding model bump, or a
prompt edit can degrade answers while every unit test stays green. We need a
way to detect that before merge. Ad-hoc "looks good to me" checks against a
couple of questions do not scale and are not repeatable.

## Decision

Adopt three rules, detailed in [evaluation.md](../evaluation.md):

1. **Versioned eval dataset** in `data/eval/`, committed and reviewed like
   code; small and stable beats large and churny.
2. **Layered metrics**: retrieval (Recall@k, MRR) measured separately from
   generation (faithfulness, correctness), so a regression is attributable
   to a layer.
3. **Baseline deltas**: the first stable run on a dataset version is the
   baseline; every later run is reported as a delta, with pass/fail
   thresholds declared in `configs/default.yaml → evals`.

## Consequences

- Any change to chunking, embeddings, retrieval params, or prompts requires
  an eval run before merge — a process cost we accept deliberately.
- The eval config keys exist from M0 even though the harness lands later, so
  automation changes values, not shape.
- LLM-graded generation metrics introduce grader noise; mitigated by keeping
  retrieval metrics deterministic and treating LLM grades as secondary
  signals with human spot-checks.
