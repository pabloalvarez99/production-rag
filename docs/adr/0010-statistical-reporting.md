# ADR 0010 — Paired statistics and a mechanical reporting boundary

- **Status:** Accepted
- **Date:** 2026-08-11
- **Deciders:** production-rag maintainers
- **Relates to:** [ADR 0003](0003-eval-strategy.md) (the two evaluation tiers),
  [ADR 0008](0008-evaluation-corpus.md) (the corpus and 60-item golden set)

## Context

The evaluation set has 60 items in six deliberately different slices, ten items
per slice. That is enough to expose failure modes and far too little to turn a
slice delta into a finding. At `n=10`, the approximate 95% interval on a
difference between two proportions is about **±30 percentage points**. Printing
six slice deltas without that limitation would give unstable counts the visual
authority of measurements.

The aggregate has more information, but only if the comparison preserves its
design. Every configuration answers the same questions. Treating their scores as
independent proportions discards which individual questions changed outcome.
A paired analysis retains that evidence: a 12–3 split among 15 discordant items
is evidence of a systematic change even though the other 45 items agree.

## Decision

All configuration comparisons are paired by construction. The matrix runner
keeps the golden-set order fixed and stores one boolean hit outcome per item per
configuration. Statistics consume those aligned vectors, never aggregate scores.

Confidence intervals use a paired percentile bootstrap. A resample draws item
indices once and applies the same draw to both configurations. It uses 10,000
resamples, a fixed seed, and records both values with the run so the same data
cannot acquire a different interval on repetition.

Significance uses exact, two-sided McNemar. Only discordant pairs contribute.
The p-value is the exact binomial tail rather than the chi-square approximation,
because this golden set will commonly produce fewer than 25 discordant pairs —
the range where the approximation is least defensible.

Every slice also reports an 80%-power minimum detectable effect from a seeded
simulation. It converts “we need more labels” into a quantitative label-planning
decision. It does not rescue an underpowered result after seeing its direction.

The publication rule is fixed in code:

```text
reportable = (n >= 30) and (ci95 excludes zero)
```

Zero on either confidence-interval boundary counts as including zero. Everything
else is labelled **directional** with a machine-written reason. In particular,
every `n=10` slice is `powered: false` and cannot be reportable regardless of
the observed delta or p-value.

## Consequences

The project will publish an aggregate comparison only when its paired interval
clears the rule. It will not publish slice-level winners from ten labels, rank
configurations by unqualified point estimates, substitute an independent-
proportions test for paired evidence, or present fake-provider runs as quality
measurements. Fake runs validate the machinery and carry `billed: false`; their
scores have no evidentiary meaning.

This rule is intentionally stricter than “the p-value is below 0.05.” McNemar
remains in the contract because discordance is useful evidence, but reportability
also requires a minimum sample and a confidence interval that states effect
uncertainty directly.

The cost is visible restraint. Many runs will produce no publishable comparison,
and every current slice is directional. That is preferable to a confident claim
whose uncertainty was hidden by the scorecard.

## Alternatives considered

**Independent proportion intervals and tests.** Rejected because configurations
are evaluated on the same items; treating them as independent throws away the
pairing and power.

**Chi-square McNemar.** Rejected because its large-sample approximation is being
asked to operate precisely where discordant counts are small.

**Human judgement about which deltas “look convincing.”** Rejected because it
makes the reporting threshold change after the result is known. The code applies
one boundary to every configuration and every scope.

**A larger scientific dependency for two short statistics.** Rejected. The exact
binomial sum and paired bootstrap are small pure functions; adding SciPy would
make the default install materially heavier without changing the decision.
