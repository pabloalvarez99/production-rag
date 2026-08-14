# Pablo Alvarez — AI Engineering

I build production-shaped AI systems across retrieval, agents, evaluation, and the
operational boundaries around them. Every live project has a deterministic free path:
no API key, no billed call, and no signup.

> Production-shaped AI systems: free-path demos, real architecture, measurable behavior,
> honest scope.

**Portfolio:** [paxdev.vercel.app](https://paxdev.vercel.app)

## The five-system series

| # | System | Engineering focus | Status |
| --- | --- | --- | --- |
| 1 | [production-rag](https://github.com/pabloalvarez99/production-rag) | Hybrid dense + sparse retrieval, RRF, optional rerank, grounded citations or refusal, offline evals, UI | **LIVE — v0.1.0** |
| 2 | [agentic-rag-research](https://github.com/pabloalvarez99/agentic-rag-research) | Bounded plan → retrieve → critique loop, tools, explicit stop reasons, deterministic traces | **LIVE — v0.1.0** |
| 3 | [multi-agent-orchestration](https://github.com/pabloalvarez99/multi-agent-orchestration) | Orchestrator, specialist handoffs, isolation, budgets, timelines, Writer-only final output | **LIVE — v0.1.0** |
| 4 | [repomind](https://github.com/pabloalvarez99/repomind) | AST-aware code chunks, `path:line` citations, fixture-backed evals | **LIVE — v0.1.0** |
| 5 | [ai-platform](https://github.com/pabloalvarez99/ai-platform) | Gateway API-key auth, per-key rate limits, request IDs, prefix proxying, status console | **LIVE — v0.1.0, gateway only** |

All five are public repositories carrying a `v0.1.0` tag and a green CI badge. Row 5 is the
one that needs its qualifier read: what runs for free is the **gateway process by itself**.
It authenticates, throttles, stamps request IDs, and reports status with **no** upstream
configured — it does not host systems 1–4, and no deployment exists where a request through
it reaches a running one. Proxying is implemented against a tested prefix contract; supplying
real upstreams is the reviewer's configuration, not a service I operate.

## Start here — free, local, reproducible

- **[production-rag](https://github.com/pabloalvarez99/production-rag#try-it-free-0-no-api-key):**
  one script starts Qdrant, ingests the sample corpus, and opens a UI where a supported
  question returns validated citations and an unsupported one explicitly refuses.
- **[agentic-rag-research](https://github.com/pabloalvarez99/agentic-rag-research#readme):**
  run the deterministic research loop and inspect its evidence, step budget, stop reason,
  and complete trace without a hosted model.
- **[multi-agent-orchestration](https://github.com/pabloalvarez99/multi-agent-orchestration#readme):**
  send one task to an orchestrator, watch the handoff timeline, and see a specialist failure
  come back as a typed degraded outcome instead of a crash or a silent guess.
- **[repomind](https://github.com/pabloalvarez99/repomind#readme):** ask a question about a
  repository and get an answer whose citations are `path:line` pairs you can open — or a
  refusal when the walk found nothing that supports one.
- **[ai-platform](https://github.com/pabloalvarez99/ai-platform#readme):** start the gateway
  alone and get rejected without a key, throttled with one, and a status console that reports
  its upstreams as unconfigured.

The free-provider metrics prove data flow, contracts, and reproducibility. They are not
presented as hosted retrieval or answer-quality results.

Reviewing all five in one sitting:
[DEMO-DAY.md](https://github.com/pabloalvarez99/production-rag/blob/main/docs/DEMO-DAY.md)
is a minute-boxed forty-five-minute script covering every system on the free path.

## Flagship: production-rag

[![production-rag CI](https://github.com/pabloalvarez99/production-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/pabloalvarez99/production-rag/actions/workflows/ci.yml)
[![release](https://img.shields.io/github/v/release/pabloalvarez99/production-rag)](https://github.com/pabloalvarez99/production-rag/releases/tag/v0.1.0)

- Dense and BM25 sparse branches remain separate until reciprocal rank fusion combines
  their rankings; a cross-encoder can then rerank the shortlist.
- The query path validates citation markers against the exact prompt context and refuses
  when evidence is absent.
- Two offline evaluation tiers separate retrieval behaviour from answer/citation
  behaviour, with provenance and paired statistical reporting.
- CI deliberately empties provider keys and runs the free path end to end.
- The [case study](https://github.com/pabloalvarez99/production-rag/blob/main/docs/CASESTUDY.md)
  explains the RRF, fail-open, grounding, and evaluation trade-offs.

## Agent layer: agentic-rag-research

[![agentic-rag-research CI](https://github.com/pabloalvarez99/agentic-rag-research/actions/workflows/ci.yml/badge.svg)](https://github.com/pabloalvarez99/agentic-rag-research/actions/workflows/ci.yml)

This second system adds a bounded research loop over retrieval: typed tool contracts,
hard step budgets, no duplicate sub-question retrieval, explicit terminal states, and a
trace that remains available whether the run completes, exhausts its budget, or refuses.

## Suggested GitHub pins

All six repositories are public; pin them in this order:

1. [production-rag](https://github.com/pabloalvarez99/production-rag)
2. [agentic-rag-research](https://github.com/pabloalvarez99/agentic-rag-research)
3. [multi-agent-orchestration](https://github.com/pabloalvarez99/multi-agent-orchestration)
4. [repomind](https://github.com/pabloalvarez99/repomind)
5. [ai-platform](https://github.com/pabloalvarez99/ai-platform)
6. [paxdev](https://github.com/pabloalvarez99/paxdev)

GitHub profile pins are configured in the web UI; copying this README does not pin the
repositories automatically.
