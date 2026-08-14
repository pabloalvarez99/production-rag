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
| 2 | [agentic-rag-research](https://github.com/pabloalvarez99/agentic-rag-research) | Bounded plan → retrieve → critique loop, tools, explicit stop reasons, deterministic traces | **LIVE THROUGH M2** — API/CLI next |
| 3 | `multi-agent-orchestration` | Orchestrator, specialist handoffs, isolation, budgets, timelines | **PLANNED** |
| 4 | `repomind` | AST-aware code chunks, `path:line` citations, fixture-backed evals | **PLANNED** |
| 5 | `ai-platform` | Gateway auth, rate limits, multi-service compose, aggregate health | **PLANNED** |

Rows 3–5 describe the roadmap, not shipped repositories.

## Start here — free, local, reproducible

- **[production-rag](https://github.com/pabloalvarez99/production-rag#try-it-free-0-no-api-key):**
  one script starts Qdrant, ingests the sample corpus, and opens a UI where a supported
  question returns validated citations and an unsupported one explicitly refuses.
- **[agentic-rag-research](https://github.com/pabloalvarez99/agentic-rag-research#readme):**
  run the deterministic research loop and inspect its evidence, step budget, stop reason,
  and complete trace without a hosted model.

The free-provider metrics prove data flow, contracts, and reproducibility. They are not
presented as hosted retrieval or answer-quality results.

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

Pin these in order as the series ships:

1. [production-rag](https://github.com/pabloalvarez99/production-rag)
2. [agentic-rag-research](https://github.com/pabloalvarez99/agentic-rag-research)
3. [paxdev](https://github.com/pabloalvarez99/paxdev)
4. `multi-agent-orchestration` when public
5. `repomind` when public
6. `ai-platform` when public

GitHub profile pins are configured in the web UI; copying this README does not pin the
repositories automatically.
