# Quality bar (canonical)

This repository is part of an **advanced AI Engineering portfolio**.

The bar is **production AI systems** — the level expected after selective,
practitioner-led programs (e.g. AI Makerspace-style build→ship→share, serious
MLE/AI engineering intensives), **not** “GenAI for beginners” tutorials.

## Non-negotiables

1. **Production shape** — real package layout, FastAPI (or equivalent), config via env, Docker Compose, health/ready, no secrets in git.
2. **Architecture docs** — `docs/architecture.md` plus ADRs for non-obvious trade-offs.
3. **Explicit contracts** — chunk/payload schemas, stable IDs, citable provenance on the query path once generation lands.
4. **Evals** — golden sets and measurable gates (Ragas and/or custom). “Looks good” is not a metric.
5. **Observability** — request IDs, structured logs, seams for traces (OpenTelemetry / Langfuse).
6. **Tests** — offline unit tests always; integration tests optional and marked.
7. **Runbook** — a stranger can bring the system up and exercise the current milestone in under 30 minutes.
8. **Modular code** — ingest / retrieval / generation / evals as separate packages; no god-module.
9. **Honest scope** — README states what is live vs planned.
10. **Hiring-manager README** — problem, architecture, how to run, metrics, roadmap.

## Reject

- Notebook-only demos
- Hardcoded API keys
- Single-file SDK wrappers sold as “production RAG”
- Grounded answers without citations (once generation exists)
- Untested “portfolio” code
- Fake multi-agent graphs with no tools, state, or evaluation

## For agents (Claude Code, etc.)

Read this file and `AGENTS.md` before implementing a milestone.
Prefer boring, testable production code over clever one-liners.
If a change would fail this bar, stop and report — do not ship tutorial-grade work.
