# production-rag

A production-grade Retrieval-Augmented Generation service, built milestone by
milestone with tests and operational endpoints from day one.

## What

A FastAPI service that will ingest documents, index them in Qdrant and answer
questions with retrieved context. Each stage (ingest, retrieval, reranking,
generation, evaluation) is its own subpackage so every piece stays testable
in isolation.

## Why

Most RAG demos stop at a notebook. This project is the path from demo to
something you can actually run: configuration as code, health/readiness
probes, structured logging, and a milestone plan that adds one concern at a
time.

## Status

**M0** — scaffold only, on branch `feat/kimi-m0`:

- Settings via `pydantic-settings` (`.env` supported, secrets never logged).
- Liveness: `GET /health` and `GET /v1/health`.
- Readiness: `GET /v1/ready` (configuration check, no network).
- `X-Request-ID` middleware with structured access logs.
- Offline test suite (`pytest`).

## Quickstart (local, no Docker)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python -m pytest -q
uvicorn production_rag.main:app --reload
```

Then open <http://localhost:8000/docs>.

## Roadmap

M0–M8 are planned: M0 scaffold → ingest/chunking → Qdrant indexing →
retrieval → reranking → generation → evaluation (ragas) → observability →
hardening. Each milestone lands behind its own branch and tests.

---

## Docker quickstart

Prerequisites: Docker Engine 24+ with the Compose v2 plugin, and ports 8000,
6333, 6334 free. Copy `.env.example` to `.env` and fill in `OPENAI_API_KEY`
before exercising the query path — the API boots and serves `/health` without
it.

```bash
cp .env.example .env
docker compose up -d --build   # or: make up  /  .\scripts\up.ps1
docker compose ps              # both services should read (healthy)
make health                    # probe API + Qdrant
make logs                      # tail logs
make down                      # stop, keep the vector index
```

| Surface | URL |
|---|---|
| API docs | <http://localhost:8000/docs> |
| Liveness | <http://localhost:8000/health> |
| Readiness | <http://localhost:8000/v1/ready> |
| Qdrant dashboard | <http://localhost:6333/dashboard> |

The stack is two containers: `api` (built from `Dockerfile`, non-root,
uvicorn on 8000) and `qdrant` (pinned to `qdrant/qdrant:v1.13.2`, storage on
the named volume `production-rag-qdrant-storage`). Only `docker compose down -v`
destroys the vector index; plain `down` keeps it.

## Make targets and PowerShell scripts

`make` is optional — the PowerShell scripts are behaviourally equivalent for
Windows machines without it.

| Entry point | Purpose |
|---|---|
| `make up` / `scripts/up.ps1` | Build and start the stack, wait for `/health` |
| `make down` / `scripts/down.ps1` | Stop the stack (`-Purge` also drops the vector volume, with a prompt) |
| `make restart` | Rebuild and recreate the API container only |
| `make health` / `scripts/health.ps1` | Probe every health surface, non-zero exit on failure |
| `scripts/smoke_health.py` | Stdlib-only smoke test; `--json` for CI, `--retries` for cold boot |
| `make logs` / `make ps` | Tail logs / show status and health |
| `make test` | Run the test suite inside the API container (`tests/` is mounted, not baked in) |
| `make shell-api` | Interactive shell in the running API container |
| `make clean` | `down -v` — destructive, drops the vector index |

## Configuration

| Layer | Where | Precedence |
|---|---|---|
| Declarative runtime config | `configs/default.yaml` | lowest |
| Profile override | file at `CONFIG_PATH` | middle |
| Environment | `.env` / compose `environment:` | highest |

`configs/default.yaml` is deliberately broader than what M0 consumes — the
`rerank`, `evals`, and `observability` blocks fix the shape now so later
milestones change values rather than structure.

**Secrets never live in config files or in the image.** Config files name the
environment variable holding a credential (`api_key_env: OPENAI_API_KEY`), never
its value. `.env` is gitignored and dockerignored; `.env.example` documents the
full set with empty values.

## Documentation

- [Architecture](docs/architecture.md) — components, query path, ingest path
- [Data model](docs/data-model.md) — collection schema, payload fields, chunk ids
- [Runbook](docs/runbook.md) — start, verify, debug, recover, upgrade
- [Evaluation](docs/evaluation.md) — golden dataset, metrics, thresholds
- ADRs: [0001 hybrid retrieval on Qdrant](docs/adr/0001-hybrid-qdrant.md) ·
  [0002 LangGraph query orchestration](docs/adr/0002-langgraph-query.md) ·
  [0003 evaluation strategy](docs/adr/0003-eval-strategy.md)
- Data layout: [`data/raw/`](data/raw/README.md) ·
  [`data/eval/`](data/eval/README.md) · [`data/processed/`](data/processed/README.md)
