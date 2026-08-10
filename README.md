# production-rag

Production-grade Retrieval-Augmented Generation service: **hybrid retrieval over Qdrant, cross-encoder reranking, answers that carry citations, and an evaluation gate that decides whether a change ships.**

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Vector store](https://img.shields.io/badge/vectors-Qdrant-DC244C.svg)](https://qdrant.tech/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## What this is

Most RAG demos are one embedding call, one cosine search and a prompt. They fall over on the first real corpus: acronyms and part numbers that dense embeddings cannot see, top-k results that are similar but not relevant, answers no one can trace back to a source, and no way to tell whether last week's prompt change made things better or worse.

This service is the other thing — the shape a RAG system takes when it has to be operated:

| Concern | Position taken here |
| --- | --- |
| Retrieval | **Hybrid**: dense vectors *and* sparse/BM25 in one Qdrant query, fused with reciprocal rank fusion. Exact identifiers stop being invisible. |
| Precision | A **cross-encoder reranker** (`bge-reranker-base`) reorders the fused candidates before anything reaches the LLM. Retrieval recall and answer precision are separate problems. |
| Trust | Every answer returns **citations** to the chunks that produced it. An answer without a source is not an answer. |
| Change safety | **Ragas** plus a golden set, run as a regression gate. "It feels better" is not a result. |
| Operability | Config from the environment, structured logs with a correlation id per request, liveness and readiness probes, containerised from day one. |

The stack is locked so each milestone is an implementation task rather than an architecture debate: **LlamaIndex** (ingest and node parsing), **LangGraph** (the query graph, from M4), **Qdrant** (dense + sparse vectors, payload filters), **bge-reranker-base** (local reranking, with Cohere as an optional swap), **FastAPI** + **Pydantic v2** (HTTP surface), **Ragas** (evaluation), **structlog** + OpenTelemetry (observability).

## Current status — M0 (scaffold)

M0 is the walking skeleton, and nothing more. What exists today:

- An installable, typed Python package (`src/` layout, `production_rag`).
- Environment-driven configuration with validation and secret masking.
- `GET /health`, `GET /v1/health`, `GET /v1/ready`, plus OpenAPI at `/docs`.
- A correlation id (`X-Request-ID`) bound to every request and every log line.
- A test suite that passes **with no network and no Qdrant running**.

What deliberately does not exist yet: ingestion, embeddings, retrieval, reranking, generation, evaluation. Readiness reports whether a vector store is *configured*; it opens no sockets. See the [roadmap](#roadmap).

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness, unversioned — for container and orchestrator probes that should never chase an API version. |
| `GET` | `/v1/health` | The same payload inside the versioned namespace, for API clients. |
| `GET` | `/v1/ready` | Readiness: configuration parsed, and whether a Qdrant endpoint is set. |
| `GET` | `/docs`, `/openapi.json` | Interactive documentation and the machine-readable schema. |

```console
$ curl -s localhost:8000/health
{"status":"ok","service":"production-rag","version":"0.1.0","environment":"local"}

$ curl -s localhost:8000/v1/ready
{"status":"ready","qdrant_configured":true,"checks":{"settings":"ok"}}
```

Both probes are separate operations in the schema (`health`, `health_unversioned`) so generated clients stay valid.

## Local quickstart (no Docker)

Requires Python 3.12+. Nothing below contacts the network at runtime, and no credentials are needed to boot or to run the tests.

```bash
# 1. Virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1

# 2. Install the package plus the dev toolchain, editable
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

# 3. Run the API
python -m uvicorn production_rag.main:app --reload --port 8000

# 4. Probe it (in another shell)
curl -s localhost:8000/health
curl -s localhost:8000/v1/ready
open http://localhost:8000/docs
```

### Tests, linting, types

```bash
python -m pytest -q            # unit suite: offline, no Qdrant, no API key
python -m ruff check .         # lint
python -m ruff format --check .
python -m mypy                 # strict, config in pyproject.toml
```

### Configuration

Every setting is an environment variable, optionally seeded by a `.env` file in the project root. Unknown keys are ignored, so the same `.env` can be shared with Docker Compose.

| Variable | Default | Meaning |
| --- | --- | --- |
| `APP_NAME` | `production-rag` | Service name reported by the probes. |
| `APP_VERSION` | `0.1.0` | Version reported by the probes. |
| `ENVIRONMENT` | `local` | Deployment environment. Anything other than `local` switches logs to JSON. |
| `LOG_LEVEL` | `INFO` | Standard level name, any case. |
| `API_PREFIX` | `/v1` | Versioned route prefix. Normalised; `/` is rejected. |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Bind address for uvicorn. |
| `QDRANT_URL` | `http://localhost:6333` | Vector store endpoint. Reported by `/v1/ready`; dialled from M1. |
| `QDRANT_COLLECTION` | `documents` | Collection holding chunks and their vectors. |
| `CONFIG_PATH` | *unset* | Optional YAML file with retrieval/rerank knobs; consumed from M1. |
| `OPENAI_API_KEY` | *unset* | Only needed once generation lands (M4). |

`OPENAI_API_KEY` is never logged and never serialised into a response. `Settings.safe_dump()` is the only sanctioned way to render settings, and a test asserts the masking.

> **Never commit a real `.env`.** It is gitignored; the tracked template is `.env.example`.

The quickstart above is the fastest edit loop and needs neither Docker nor a running Qdrant. For the container path, see the next section.

## Docker & Ops

Two containers: `api` (built from `Dockerfile` — `python:3.12-slim`, non-root, uvicorn on 8000) and `qdrant` (pinned to `qdrant/qdrant:v1.13.2`). Both carry healthchecks; the API waits for Qdrant to report healthy before it starts. Prerequisites are Docker Engine 24+ with the Compose v2 plugin and ports 8000, 6333 and 6334 free.

```bash
cp .env.example .env             # PowerShell: Copy-Item .env.example .env
docker compose up -d --build     # or: make up  /  .\scripts\up.ps1
docker compose ps                # both services should read (healthy)
make health                      # probe the API and Qdrant, non-zero exit on failure
make logs                        # tail both services
make down                        # stop, keep the vector index
```

| Surface | URL |
| --- | --- |
| API docs | <http://localhost:8000/docs> |
| Liveness | <http://localhost:8000/health> |
| Readiness | <http://localhost:8000/v1/ready> |
| Qdrant dashboard | <http://localhost:6333/dashboard> |

Vectors live on the named volume `production-rag-qdrant-storage`, which survives `docker compose down`. Only `down -v` (`make clean`, or `scripts/down.ps1 -Purge`, which prompts first) destroys it, and that costs a full re-ingest. The Qdrant tag is pinned because its storage format is version-sensitive — the [runbook](docs/runbook.md) has the backup-then-upgrade procedure.

`configs/` and `data/` are bind-mounted read-only, so a config edit or a new corpus document needs a restart, not a rebuild.

### Make targets and PowerShell scripts

`make` is optional. The PowerShell scripts are behaviourally equivalent for Windows machines without it.

| Entry point | Purpose |
| --- | --- |
| `make up` / `scripts/up.ps1` | Build and start the stack, then wait for `/health`. On timeout, dumps the last 50 API log lines and exits non-zero. |
| `make down` / `scripts/down.ps1` | Stop the stack. `-Purge` also drops the vector volume, after a confirmation prompt. |
| `make restart` | Rebuild and recreate the API container only — the fast loop for ops changes. |
| `make health` / `scripts/health.ps1` | Probe every health surface plus Qdrant readiness and collection presence. |
| `scripts/smoke_health.py` | Stdlib-only smoke test, so it runs from any interpreter even when the install is what you suspect. `--json` for CI, `--retries` for the cold-boot window. |
| `make logs` / `make ps` | Tail logs / show status and health. |
| `make test` | Run the suite inside the API container (`tests/` is mounted, not baked in — the image ships runtime deps only, so `make test-host` is the fallback). |
| `make shell-api` | Interactive shell in the running API container. |
| `make clean` | `down -v`. Destructive: drops the vector index. |

### Layered configuration

Environment variables (documented in the table above) are the highest-precedence layer and the only place a credential appears. Below them sits `configs/default.yaml`: the declarative knobs for ingest, retrieval, rerank, generation, Qdrant topology, evals and observability. It is deliberately broader than what M0 consumes, so later milestones change values rather than structure — `CONFIG_PATH` selects a profile that overrides it.

Config files name the environment variable holding a secret (`api_key_env: OPENAI_API_KEY`), never its value. `.env` is both gitignored and dockerignored; `.env.example` is the tracked template and carries empty values only.

### Documentation

- [Architecture](docs/architecture.md) — components, query and ingest paths, failure behaviour
- [Data model](docs/data-model.md) — collection schema, payload fields, chunk identity
- [Runbook](docs/runbook.md) — start, verify, debug, recover, upgrade Qdrant
- [Evaluation](docs/evaluation.md) — golden dataset, metrics, thresholds
- ADRs — [0001 hybrid retrieval on Qdrant](docs/adr/0001-hybrid-qdrant.md) · [0002 LangGraph query orchestration](docs/adr/0002-langgraph-query.md) · [0003 evaluation strategy](docs/adr/0003-eval-strategy.md)
- Data layout — [`data/raw/`](data/raw/README.md) · [`data/eval/`](data/eval/README.md) · [`data/processed/`](data/processed/README.md)

## Project structure

```text
production-rag/
├── src/production_rag/
│   ├── __init__.py            # package version and public surface
│   ├── config.py              # Settings (pydantic-settings) + cached get_settings()
│   ├── main.py                # create_app() factory, logging setup, ASGI `app`
│   └── api/
│       ├── deps.py            # SettingsDep — the injection seam the tests use
│       ├── middleware.py      # correlation id, timing, one structured access log
│       ├── schemas.py         # response models (HealthResponse, ReadyResponse)
│       └── routes/
│           ├── health.py      # GET /health and GET /v1/health
│           └── ready.py       # GET /v1/ready
├── tests/
│   ├── conftest.py            # fixtures: isolated settings, app + client factories
│   └── unit/                  # health, ready, config, middleware, version
├── configs/                   # YAML profiles (retrieval and rerank knobs, M1+)
├── data/                      # raw corpora, processed chunks, eval sets
├── docs/                      # architecture notes and ADRs
├── scripts/                   # operator helpers
├── docker-compose.yml         # API + Qdrant
├── Dockerfile
├── Makefile
└── pyproject.toml             # deps, ruff, mypy, pytest — one config file
```

Milestones add sibling packages under `src/production_rag/` (`ingest/`, `retrieval/`, `rerank/`, `generation/`, `evaluation/`) rather than growing the existing modules. Each stage is exercisable on its own — that is what makes an evaluation gate possible.

### Design notes

A few decisions worth stating, because they are the ones that get undone by accident:

- **Liveness carries no dependency state.** A liveness probe that fails when Qdrant is down makes the orchestrator restart a healthy process, which does not fix Qdrant and drops in-flight requests. Dependency state belongs to readiness.
- **`create_app()` is a factory, not module-level assembly.** Tests build an app around explicit settings via `app.dependency_overrides`, so no state leaks between cases and the suite never mutates the environment.
- **The correlation id is bound once, into structlog contextvars.** From M4 a single question fans out into embedding, retrieval, rerank and generation calls; without one id on every line, debugging an incident is guesswork. An inbound id that is absent or malformed is replaced with a fresh UUID4 — it reaches both a response header and the logs, so it is validated, not trusted.
- **Offline tests are a feature, not a limitation.** Readiness is a configuration check by design, which is what lets `pytest` be green with the wifi off.

## Roadmap

| Milestone | Scope | Status |
| --- | --- | --- |
| **M0** | Scaffold: package, config, health/readiness, tests, container stack | ✅ done |
| **M1** | Ingest with LlamaIndex; chunking; dense embeddings; Qdrant collection and upsert | ⏳ next |
| **M2** | Hybrid retrieval: sparse/BM25 vectors alongside dense, fused with RRF | 📋 planned |
| **M3** | Cross-encoder reranking (`bge-reranker-base`), Cohere as an optional swap | 📋 planned |
| **M4** | Generation with citations; `POST /v1/query` orchestrated as a LangGraph graph | 📋 planned |
| **M5** | Observability: OpenTelemetry traces, structured logs, token and latency metrics, optional Langfuse | 📋 planned |
| **M6** | Evaluation: Ragas metrics, a golden set, and a regression gate in CI | 📋 planned |
| **M7** | Hardening: rate limits, timeouts, retries and backoff, input validation, graceful degradation | 📋 planned |
| **M8** | Portfolio polish: architecture write-up, benchmark numbers, ADRs, demo corpus | 📋 planned |

Retrieval quality claims arrive with M2–M3 and will be reported as measured numbers on a stated corpus, never as adjectives.

## License

[MIT](LICENSE).
