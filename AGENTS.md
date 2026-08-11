# Agent rules — production-rag

## Quality bar (principal rule)

**This project targets elite AI Engineering portfolio quality** — production LLM
systems, not beginner GenAI demos.

Full bar: [`docs/quality-bar.md`](docs/quality-bar.md)

Before any milestone work, assume the standard of selective production-focused
programs (AI Makerspace-style, advanced MLE/AI engineering intensives): modular
services, hybrid retrieval where claimed, evals, observability, Docker, ADRs,
and an honest README.

### Mandatory for every change

- No secrets in the repo (values). Use `.env.example` pointers only.
- Do not claim features that are not implemented.
- Prefer unit tests without network; mark integration tests.
- Respect dual-agent ownership when a prompt defines it.
- Do not push to remote unless the human explicitly asks.
- Kimi (or low-trust models): only on isolated branches when the orchestrator says so; not the default worker.

### Stack direction (locked at portfolio level)

- Qdrant, FastAPI, dense→hybrid→rerank→generate+cite, Ragas/golden evals
- LangGraph on the query path when generation lands
- Fake embedder path for offline demos; OpenAI (or equivalent) optional

### Second brain (human operator)

Session digests and decisions live in the operator vault when working from
firstmate-home; do not write credential **values** into notes or this repo.
