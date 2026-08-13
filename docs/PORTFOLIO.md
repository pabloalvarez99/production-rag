# AI engineering portfolio — a five-project series

Five projects, one thesis: a retrieval system is only as good as the evidence it can show
for its own behaviour. Each project ships a runnable service, an evaluation harness that can
contradict its author, and decision records for the trade-offs that were not obvious.

The series is deliberately sequential. Every project reuses the previous project's core
instead of restarting from a template, and no project starts before its entry condition is
true. That is why exactly one of them is complete.

## The series

| # | Project | What it adds | State |
| --- | --- | --- | --- |
| **P1** | **[production-rag](../README.md)** — hybrid RAG service | Hybrid dense + sparse retrieval with RRF, cross-encoder rerank, grounded answers with resolvable citations, refusal as a first-class outcome, two-tier offline evaluation with paired statistics, server-rendered query UI | **portfolio-complete on the free path**; one hosted baseline run outstanding |
| P2 | Agentic RAG research agent | Query planning, multi-step retrieval, tool use, and a stopping rule, over P1's retrieval core rather than a new one | planned — six entry criteria below, starting with P1's hosted baseline so agent behaviour is measured against a known retrieval floor |
| P3 | Multi-agent system | An orchestrator plus specialists, explicit handoff contracts, shared state, and failure containment between agents | planned — needs P2's single-agent trace and cost profile as its comparison baseline |
| P4 | RepoMind code intelligence | Change-impact answers over a repository: which artefacts a change touches, with evidence, on a corpus and golden set of its own | scoped — three preconditions before any code, chief among them a corpus where BM25 and direct neighbours provably miss the answer |
| P5 | Full production AI platform | The hardening deliberately excluded from P1: auth, rate limits, a metrics endpoint, payload filters, retry and timeout policy, load and concurrency work | planned — the crown project; it wraps P1 through P4 rather than replacing them |

**P2 through P5 are plans, not work in progress.** No code exists for any of them, no
repository is linked, and nothing on this page should be read as a partially built system.
Each one is stated with the entry condition that has to be true before it starts, so the
sequence can be checked rather than trusted. P1 is the only project with something to run.

## P1 — production-rag

Repository: [`production-rag`](https://github.com/pabloalvarez99/production-rag) · full
detail in the [README](../README.md).

### Status

**Portfolio-complete on the free path.** Every milestone from scaffold through two-tier
evaluation is landed on `main`, plus the adversarial corpus, the paired statistics, the query
UI, and the publication path that renders the measurement artefact into the README and fails
CI when the two disagree. Nothing is queued for it, and the credential-free path is
public-ready: a reviewer clones it and reaches a cited answer without an account or a key.
Short version with the CI evidence: [SHIP.md](SHIP.md).

One thing is deliberately open: no hosted-provider baseline has been run, so the published
scorecard is an explicitly labelled local-provider plumbing fixture rather than a quality
result. The repository says so wherever a number appears, which is the honest state — not a
gap hidden behind a screenshot.

### Free demo

No credential, no billed call, no signup:

```powershell
git clone https://github.com/pabloalvarez99/production-rag
cd production-rag
.\scripts\demo_setup.ps1          # macOS or Linux: ./scripts/demo_setup.sh
```

Then open <http://localhost:8000/> and ask *why does hybrid search use reciprocal rank
fusion?* for a cited answer, and *who won the Antarctic underwater chess championship?* for
an explicit refusal. The [README](../README.md#try-it-free-0-no-api-key) covers the rest of
the free path: ingest, the query API, both evaluation tiers, and regenerating the scorecard.

### What it is evidence of

- **Retrieval that survives a real corpus.** Lexical and semantic matching stay separate
  until reciprocal rank fusion combines their ranks, then a cross-encoder reorders the
  survivors, and the reranker degrades the ordering rather than the availability when it
  fails.
- **Answers a reviewer can audit.** Citation markers resolve to structured source records,
  and missing evidence produces a refusal instead of invented support.
- **Measurement that can embarrass its author.** Comparisons are paired by construction,
  intervals come from a seeded bootstrap, and a delta is published as a claim only when
  sample size and interval permit it. The current aggregate does not qualify, so it is
  published as directional.
- **Restraint recorded as decisions.** Every ADR carries the alternatives that were rejected
  and why — including the reporting boundary that keeps a favourable-looking result from
  being announced.

## P2 — Agentic RAG Research Agent: entry criteria

P2 adds a research loop on top of P1: plan a query, retrieve more than once, use a tool, decide
when to stop, and answer with citations that survive the extra hops. It does not start because
the idea is ready. It starts when the conditions below are true, and each one is checkable by
someone who did not write it.

1. **P1's hosted baseline exists.** One billed run with named providers, published through the
   same scorecard contract. Without it there is no retrieval floor, so any agent result is a
   number with nothing to compare against.
2. **P1 is consumed, not copied.** P2 depends on `production_rag` as an installed library — the
   `run_query` entry point and the retrieval modules — or calls `POST /v1/query` over HTTP. A
   forked retrieval stack is a rewrite pretending to be a series, and it would make the
   comparison in gate 5 meaningless.
3. **The free path runs first.** The whole loop — planning, several retrieval passes, tool
   calls, stopping — runs end to end on the deterministic providers, with CI green while the
   provider keys are empty strings, before a hosted call is wired in. Same rule as P1: a
   project nobody can run without paying is not a portfolio project.
4. **A golden set the single pass provably cannot answer.** Questions where P1's one-shot
   retrieval demonstrably fails, established by a mechanical predicate rather than intuition:
   the answer needs evidence from more than one document, or a fact that surfaces only after a
   first retrieval narrows the question. A set the baseline already solves makes the agent look
   useful by construction. This is P1's `deep_rank` lesson, a slice that passed schema
   integrity while measuring nothing.
5. **A paired comparison against that baseline.** Same items, same order, agent against single
   pass, under the reporting boundary in
   [ADR-0010](adr/0010-statistical-reporting.md): a delta becomes a claim only when its sample
   size and interval allow it. An agent that spends more steps and wins nothing has to be
   publishable as exactly that.
6. **A budget and a stopping rule, both recorded.** A step ceiling and a spend ceiling per
   question, enforced in code, with the real step count and cost in every report. An unbounded
   agent loop is an outage with a research paper attached.

Out of scope for P2, stated now: multi-agent orchestration belongs to P3, and auth, rate limits
and load work belong to P5. P2 is one agent, one loop, measured against the floor P1 set.

## Standards applied to every project in the series

1. **Runnable for free.** A reviewer reaches a real answer with no credential, or the project
   is not finished.
2. **Evaluation before features.** A golden set and a harness exist before the next
   capability is added, and the harness is allowed to return an unflattering result.
3. **Provenance travels with numbers.** Any published metric carries its embedder, model,
   judge, sample size, date and commit, enforced mechanically rather than by discipline.
4. **Decisions are recorded, not implied.** ADRs hold the trade-offs; commit history holds
   the sequence; the README describes only what exists at the integrated tip.
5. **Scope is cut in public.** Work moved out of a project is named and assigned to a later
   one, so an absent feature is a decision with an owner rather than an oversight.
