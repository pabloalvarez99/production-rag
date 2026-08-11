---
title: Evaluating a RAG system — retrieval metrics, judges, and Ragas
tags: [rag, evaluation, metrics, ragas]
---

# Evaluating a RAG system — retrieval metrics, judges, and Ragas

Without offline evaluation, every change to a RAG system is a guess dressed as
an improvement. The point of an eval suite is not a leaderboard number; it is
the ability to say which of two configurations is better, and to notice when a
change made things worse.

## Split the measurement before you take it

A wrong answer has two possible causes, and they need opposite fixes: either the
supporting passage never reached the model, or it reached the model and the
model mishandled it. A single end-to-end score collapses both into one number
and sends you tuning prompts when the real defect is chunk size.

So retrieval is scored first and independently, and answer quality is scored
only over the queries whose retrieval succeeded.

## Retrieval metrics

These compare the returned chunk ids against hand-labelled relevant ids. No
model is involved, so runs are free, fast, and deterministic — which is what
makes them safe to run on every commit.

| Metric | The question it answers |
|---|---|
| `recall@k` | did the supporting chunk reach the top k at all? |
| `mrr` | how far down the list was the first good chunk? |
| `ndcg@k` | is the ordering good, weighting graded relevance? |

`recall@5` is the headline number because it is a hard ceiling: a chunk outside
the top 5 is not recovered by a reranker, a better prompt, or a larger model.

Report each branch separately — dense, sparse, fused. A fused score that never
beats its best individual branch is evidence that the fusion constants are
wrong, not that the retrievers are.

## Answer metrics and the LLM judge

Answer quality needs a grader that can read. In practice that is a strong model
prompted to score one dimension at a time: faithfulness, answer relevance,
citation precision, refusal accuracy.

Ragas packages these as reference-free metrics, meaning most of them need only
the question, the retrieved contexts, and the answer — not a hand-written gold
answer for every item. That is what makes a judge-based suite affordable to
maintain as the corpus grows.

Judge runs cost money and are non-deterministic, so they are sampled and run
before a release rather than on every commit.

## Calibrate the judge or do not quote it

An LLM judge is a proxy for human preference, and an uncalibrated proxy is a
number, not a measurement. Calibration is unglamorous: hand-label 20 items,
compare the judge's verdicts against them, and record the agreement rate. Repeat
whenever the judge model version changes.

A judge that agrees with humans 70% of the time can still rank two systems
correctly; a judge nobody has ever checked cannot be defended when the number
disagrees with what users report.

## Composition of the golden set

Coverage matters more than size once the set is large enough to be stable.
Roughly 40% paraphrase questions (where dense should win), 25% exact-token
questions with identifiers and error codes (where sparse should win), 20%
multi-hop questions needing two or more chunks, and 15% deliberately
unanswerable questions.

The unanswerable slice is the one most often skipped and the most diagnostic: it
is the only measurement of whether the system hallucinates under pressure. A
system that never refuses will score perfectly on the other three slices.

## Thresholds are gates, not aspirations

Thresholds should be set from the first real baseline run, not from ambition. A
gate that fails on day one gets disabled by day three and never re-enabled.
Raising a threshold as the system improves is routine; lowering one to make a
build pass is a decision that belongs in a written record, not in a commit
message.
