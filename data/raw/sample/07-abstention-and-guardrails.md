---
title: Abstention and guardrails — when the right answer is "I do not know"
tags: [rag, abstention, refusal, guardrails, safety]
---

# Abstention and guardrails — when the right answer is "I do not know"

Every corpus has holes. A system that answers anyway is not more capable than
one that refuses; it is less honest, and the dishonesty is invisible precisely
where it is most expensive.

## Abstention is a feature with a cost

Refusing when evidence is absent is the single highest-leverage setting for
perceived quality, and it is also the easiest one to over-tune. Two errors trade
against each other:

- **False answer** — the system answers without support. Costs trust, and costs
  it silently, because the user has no way to detect it.
- **False refusal** — the system refuses although the answer was in the context.
  Costs usefulness, and costs it loudly, because the user sees it immediately.

The asymmetry in *visibility* is why systems drift toward answering: false
refusals generate complaints and false answers do not. Only the unanswerable
slice of an eval set makes the other error visible.

## Three places abstention can be enforced

**At retrieval.** If no chunk clears the score threshold, refuse before spending
a generation call. Cheap and deterministic, but a fused RRF score is a rank
artefact rather than a calibrated confidence, so the threshold is corpus-specific
and easy to set too high. A threshold that is too high converts recall misses
into silent empty results, which reads to the user as "the document is not
there" when it is.

**At generation.** Instruct the model to answer only from the provided context
and to emit a specific refusal string otherwise. Catches the case where chunks
were retrieved but none actually addresses the question. Depends on model
compliance, so it is a strong default and never a guarantee.

**After generation.** Check that every claim carries a citation resolving to a
supplied chunk. An answer whose load-bearing sentence cites nothing is a refusal
in disguise, and treating it as one converts a silent failure into a visible
behaviour.

Defence in depth here is cheap: the three checks fail on different inputs.

## Write one refusal message and reuse it

A fixed refusal string — "I could not find support for that in the indexed
documents" — is worth more than a fluent apology. It is greppable in logs,
countable as a metric, and unambiguous to downstream code that needs to branch
on it. A model free-styling its refusals produces a hundred phrasings and no
measurable refusal rate.

## Refusals are a signal to route, not an endpoint

A spike in refusal rate is diagnostic and should page someone. The three common
causes are distinguishable: ingest silently failed and the collection is stale;
`score_threshold` was raised past what the corpus supports; or users started
asking about a topic the corpus genuinely does not cover.

The third is not a defect. It is a content gap, and the refusal log is the
cheapest corpus roadmap available.

## Guardrails that are not abstention

Abstention handles "the corpus does not support this". Separate concerns handle
separate problems, and folding them together produces a system nobody can tune:

- **Input validation** — length caps, encoding checks, and rejection of prompts
  attempting to override system instructions.
- **Payload filtering as authorisation** — if a user may see only some
  documents, that must be a filter applied to the retrieval query, never an
  instruction in the prompt. Retrieved-then-hidden is not access control.
- **Output checks** — scanning generated text for content the corpus should
  never emit, such as credentials pasted into a source document.

## Never fabricate a citation to satisfy a format

The worst version of this failure is a system required to emit a citation that
emits one anyway when it has nothing to cite. Requiring citations and permitting
refusal must be configured together; requiring the first without the second
produces exactly the fluent, sourced, unsupported answer that survives review.
