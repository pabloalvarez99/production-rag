# AI engineering portfolio — a five-project series

Five projects, one thesis: a retrieval system is only as good as the evidence it can show
for its own behaviour. Each project ships a runnable service, an evaluation harness that can
contradict its author, and decision records for the trade-offs that were not obvious.

The series is deliberately sequential. Each project consumes the previous project's public
boundary instead of restarting from a template, while keeping a credential-free standalone
path. P1 is released, P2 has reached its evaluation milestone, P3 has reached its traced
API/evaluation milestone, and RepoMind answers fixture-backed code questions; P5 remains a plan.

## The series

| # | Project | What it adds | State |
| --- | --- | --- | --- |
| **P1** | **[production-rag](../README.md)** — hybrid RAG service | Hybrid dense + sparse retrieval with RRF, cross-encoder rerank, grounded answers with resolvable citations, refusal as a first-class outcome, two-tier offline evaluation with paired statistics, server-rendered query UI | **portfolio-complete on the free path**; one hosted baseline run outstanding |
| P2 | [agentic-rag-research](https://github.com/pabloalvarez99/agentic-rag-research) | Bounded plan/retrieve/critique loop, optional P1 HTTP retrieval, explicit stop reasons, trace, UI, local notes tool, and offline goldens | **v0.1.0 / M6 LIVE** |
| P3 | [multi-agent-orchestration](https://github.com/pabloalvarez99/multi-agent-orchestration) | Orchestrator plus Research/Critic/Writer roles, handoff budgets, isolation, degradation, timeline, API/CLI, and offline goldens | **M4 LIVE**; P2 integration/release planned |
| P4 | [RepoMind](https://github.com/pabloalvarez99/repomind) | Safe repository walk, Python AST chunks, deterministic symbol/token retrieval, and grounded `path:line` answers or refusal | **M3 LIVE**; CLI and fixture evals planned |
| P5 | Full production AI platform | The hardening deliberately excluded from P1: auth, rate limits, a metrics endpoint, payload filters, retry and timeout policy, load and concurrency work | planned — the crown project; it wraps P1 through P4 rather than replacing them |

P2 and P3 are runnable and evaluated on deterministic free paths. P3 now exposes library,
`POST /v1/tasks`, and CLI surfaces with a timeline and 12-task routing scorecard. Its optional
P3's optional P2 integration and release remain planned. RepoMind M3 is runnable against its committed
fixture; its CLI and evaluation harness remain planned. P5 remains design intent, not a
partial implementation.

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

## P1 → P2 → P3 boundary

P2 calls P1's versioned `POST /v1/query` contract through an explicit optional HTTP adapter.
It consumes citation passages and ignores P1's generated answer, so P2 owns agent policy while
P1 remains the retrieval substrate. The default P2 path still uses a local fixture; no running
P1 service is required for its tests or demo.

P3 starts from P2's explicit budget and trace lessons, then moves policy into an orchestrator
that coordinates deterministic Research, Critic, and Writer specialists. The M4 path keeps
Writer as the sole final speaker, caps handoffs/research retries, and makes specialist failure
degraded and visible, with a JSON-safe timeline and offline routing goldens.

Two honesty boundaries remain:

- P2's 17-case fake scorecard measures deterministic contract conformance, not retrieval or
  answer quality and not uplift over a one-pass answer baseline.
- P3's fake specialists and goldens prove routing, ownership, budgets, and trace contracts,
  not answer quality or multi-model uplift.

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
