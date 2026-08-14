# AI engineering portfolio — a five-project series

Five projects, one thesis: a retrieval system is only as good as the evidence it can show
for its own behaviour. Each project ships a runnable service, an evaluation harness that can
contradict its author, and decision records for the trade-offs that were not obvious.

The series is deliberately sequential. Each project consumes the previous project's public
boundary instead of restarting from a template, while keeping a credential-free standalone
path. **All five systems are public repositories with a `v0.1.0` tag.** What differs between
them is not whether they shipped, but how much of each system a reviewer can exercise for
free — which each row below states rather than implies.

## The series

| # | Project | What it adds | State |
| --- | --- | --- | --- |
| **P1** | **[production-rag](https://github.com/pabloalvarez99/production-rag)** — hybrid RAG service | Hybrid dense + sparse retrieval with RRF, cross-encoder rerank, grounded answers with resolvable citations, refusal as a first-class outcome, two-tier offline evaluation with paired statistics, server-rendered query UI | **v0.1.0 LIVE** on the free path; one hosted baseline run outstanding |
| P2 | [agentic-rag-research](https://github.com/pabloalvarez99/agentic-rag-research) | Bounded plan/retrieve/critique loop, optional P1 HTTP retrieval, explicit stop reasons, trace, UI, local notes tool, and offline goldens | **v0.1.0 LIVE** |
| P3 | [multi-agent-orchestration](https://github.com/pabloalvarez99/multi-agent-orchestration) | Orchestrator plus Research/Critic/Writer roles, handoff budgets, isolation, degradation, timeline, API/CLI, and offline goldens | **v0.1.0 LIVE**; optional P2 integration behind `AGENTIC_RAG_URL` |
| P4 | [RepoMind](https://github.com/pabloalvarez99/repomind) | Safe repository walk, Python AST chunks, deterministic symbol/token retrieval, and grounded `path:line` answers or refusal | **v0.1.0 LIVE**; JSON CLI and fixture eval |
| P5 | [ai-platform](https://github.com/pabloalvarez99/ai-platform) | The operational edge deliberately excluded from P1: API-key authentication, per-key rate limits, request IDs, bounded prefix proxying, guardrails, status console | **v0.1.0 LIVE as a gateway**; upstreams unconfigured on the free path |

Read the P5 row precisely. What is live is the **gateway process**: a reviewer can start it
alone, be rejected without a key, be throttled with one, and read a status console that
reports its configured upstreams. It does **not** host or bundle P1 through P4 — no
deployment exists where a request to P5 reaches a running P1. Proxying is implemented and
tested against the prefix contract; pointing it at real upstreams is configuration a
reviewer supplies, not a service this portfolio operates.

P2 and P3 are runnable and evaluated on deterministic free paths. P3 exposes library,
`POST /v1/tasks`, and CLI surfaces with a timeline and a routing scorecard. RepoMind ships
the JSON CLI and its fixture eval. A tag means the surface is real and CI-verified; it does
not mean the numbers behind it are hosted-quality results, and every repository says which
of the two it is publishing.

Walk the whole series in front of a reviewer with [DEMO-DAY.md](DEMO-DAY.md): a
minute-boxed script that runs all five systems on the free path in forty-five minutes.

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

P4 leaves the prose corpus entirely: it walks a repository, chunks Python by AST, and answers
with `path:line` citations or refuses. P5 sits in front of the series rather than inside it —
an authenticating, rate-limiting, request-id-stamping edge that proxies by prefix. It is the
hardening P1 deliberately excluded, kept in its own repository so that P1's demo never needs a
key to run.

Four honesty boundaries remain:

- P2's fake scorecard measures deterministic contract conformance, not retrieval or
  answer quality and not uplift over a one-pass answer baseline.
- P3's fake specialists and goldens prove routing, ownership, budgets, and trace contracts,
  not answer quality or multi-model uplift.
- P4's fixture eval proves citation resolution and refusal on a fixed repository snapshot,
  not general code comprehension.
- P5's tag proves the edge behaves — rejects, throttles, stamps, and reports — not that
  anything is deployed behind it.

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
