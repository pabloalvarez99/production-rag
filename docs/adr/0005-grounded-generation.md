# ADR 0005 — Grounded generation: mandatory citations, refusal over invention

- **Status:** Accepted
- **Date:** 2026-08-10
- **Deciders:** production-rag maintainers
- **Relates to:** [ADR 0001](0001-hybrid-qdrant.md) (hybrid retrieval),
  [ADR 0002](0002-langgraph-query.md) (the graph that hosts this stage),
  [ADR 0004](0004-rerank-cross-encoder.md) (the stage that orders the evidence),
  [ADR 0003](0003-eval-strategy.md) (what will eventually measure this)

## Context

M1–M3 built the evidence path: chunk, embed, index both a dense and a sparse
vector, fuse the two branches, rerank the fused candidates. Every stage so far
returns *passages*. M4 is the first stage that returns **prose**, and prose is
the point at which a retrieval system acquires the ability to be confidently
wrong.

The failure is specific and well documented. Given a set of passages and a
question they do not answer, an instruction-tuned model will usually produce a
fluent, plausible answer anyway — assembled from parametric memory, from the
topical vocabulary of the passages, or from both. The output is
indistinguishable in tone from a grounded one. It survives review precisely
because it reads well, and a reviewer who has to open the corpus to check every
sentence gets no value from the system at all.

Two design questions therefore have to be answered before the first token is
generated, and neither is a prompt-tuning detail:

1. **What does the system do when the retrieved context does not support an
   answer?**
2. **How does a reader verify any individual sentence without re-reading the
   corpus?**

Answering them late is not an option: both change the response schema, the
graph topology, and what the eval harness has to score.

Alternatives considered for grounding: a free-form answer with a "sources"
list appended; post-hoc attribution (generate first, then attach the passages
that look most similar to each sentence); a second LLM pass that verifies the
first; and inline citation markers emitted by the generating model itself.

Alternatives considered for the no-evidence case: answer anyway with a hedging
preamble ("based on general knowledge…"); answer with a confidence score;
return an empty answer with the retrieved passages; and an explicit refusal.

## Decision

**Generation is grounded by construction, and refusal is a first-class
outcome.** Concretely:

### 1. The model may only use the supplied context

The system prompt states that the passages are the entire world for this
request, that outside knowledge is not to be used, and that a claim not
supported by a passage must not be made. The passages are presented numbered,
`[1]`…`[n]`, in the order the retrieval path produced (rerank order when the
stage ran, fusion order otherwise).

### 2. Every claim carries an inline `[n]` marker

Citations are **inline markers referencing an ordinal in the supplied context**
(`citations.style: bracketed_index`), not a bibliography at the end. The marker
binds a specific sentence to a specific passage, which is the only form that
lets a reader check one claim without re-reading everything. A trailing source
list makes "which of these five sources backs *this* sentence?" unanswerable —
exactly the question review consists of.

Post-hoc attribution was rejected on honesty grounds: matching a generated
sentence to the passage it most resembles produces a citation that looks
identical whether the model used the passage or not. It attributes fluency, not
provenance.

### 3. Markers are resolved against the actual retrieved set, and unresolvable
   ones are dropped

`[n]` is an ordinal into the context the request assembled, so resolution is a
lookup, not a heuristic. Every resolved marker becomes a `Citation` object
carrying `chunk_id`, `source_path`, `title`, `heading_path` and the quoted
text — the reader gets the passage, not a number.

A marker outside the range of supplied passages (`[9]` when eight were sent) is
a model error. It is **dropped from the answer text and counted**, never
rendered as a dead link. The count is reported so "the model invents citations"
is a measurable statement rather than a suspicion.

### 4. No evidence means refusal, not a hedged answer

When retrieval returns nothing that clears the evidence bar, **generation is
skipped entirely** — the graph takes a different edge, the LLM is never called,
and the response carries the configured refusal message with `citations: []`
and an explicit `refused: true`.

Not calling the model at all is deliberate. A refusal produced by the *system*
is deterministic and free; a refusal produced by *asking the model to refuse
when appropriate* is a request for good judgement from the component whose
known failure mode is exactly that judgement.

A hedged answer ("I don't have specific information, but generally…") was
rejected as the worst available outcome: it is ungrounded content wearing a
disclaimer, and the disclaimer is the first thing a reader skims past.

### 5. A grounded answer is never worse than no answer

Stated as the operating principle behind the previous four: **refuse >
hallucinate**. Given the choice between an unsupported answer and an admission
that the corpus does not cover the question, the system takes the admission
every time. An unsupported answer costs the user the time to discover it was
wrong plus the trust in every answer that came before it; a refusal costs one
query and points at a real gap in the corpus.

This is a product decision as much as a technical one, and it has a visible
cost: a system that refuses is a system that sometimes says no to a question a
more talkative competitor answers. That trade is accepted here explicitly.

## Consequences

**Positive**

- Every sentence is checkable against a named chunk in one hop. That is the
  property that makes an answer usable in a context where being wrong matters.
- The refusal path is a graph edge, not an exception buried in a handler, so it
  is testable in isolation and visible in the design (see
  [ADR 0002](0002-langgraph-query.md)).
- Skipping the LLM on the refusal path makes the cheapest outcome also the
  fastest, which is the correct incentive.
- `citation_precision` and `refusal_accuracy` (M6, [ADR 0003](0003-eval-strategy.md))
  become measurable without any change to the response schema: the citations
  are structured data, not prose to be parsed.
- Dropping unresolvable markers, and counting them, turns a class of silent
  model misbehaviour into a metric.

**Negative**

- **Recall of the answer path is now capped by retrieval.** A question whose
  supporting chunk was never retrieved gets a refusal, and a refusal reads to a
  user as "the system does not know" rather than "the retriever missed". This
  is the correct failure, but it moves pressure onto retrieval quality and makes
  the M6 separation of retrieval from generation failures load-bearing rather
  than nice to have.
- **The evidence bar is a tuning surface with an asymmetric failure mode.** Set
  it too high and the system refuses answerable questions — visible, annoying,
  and safe. Set it too low and it answers on noise — invisible, and the exact
  thing this ADR exists to prevent. The default is deliberately the permissive
  end of safe, and any change to it needs eval evidence.
- **Citation markers cost output tokens and constrain style.** The model is
  writing for verifiability, not for reading pleasure. Answers are drier than an
  ungrounded model's, and that is not an accident to be fixed later.
- **The system prompt is now a contract, not a string.** It lives in a file
  under `configs/prompts/`, and changing it is a change to system behaviour that
  belongs in a diff with an eval run attached — not a quick edit.

**Neutral / follow-ups**

- Faithfulness is not yet *measured*. The mechanisms here make an unfaithful
  answer harder to produce and easy to check by hand; proving a rate needs the
  LLM-judge harness, which is M6. Nothing in this repository quotes a
  faithfulness number, and nothing should until that harness runs.
- The `fake` generation provider (offline, no key) exercises this entire
  contract — marker resolution, unresolvable-marker dropping, the refusal edge —
  deterministically. It exercises the *contract*; it says nothing about how a
  real model behaves under it. Same caveat as the fake embedder and the fake
  reranker, one stage later.
- Streaming (`generation.stream`) does not change any of the above, but it does
  mean markers arrive before the citations that resolve them. The resolved
  `citations[]` array is emitted at the end of the stream, and a client that
  renders markers as links must buffer until it arrives.
