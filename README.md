# production-rag

> K1 owns app narrative. This file is the ops-facing entry point for now.

## Docker quickstart

Prerequisites: Docker with the Compose plugin. Copy `.env.example` to `.env`
and fill in `OPENAI_API_KEY` if you will exercise the query path.

```bash
docker compose up -d --build   # or: make up  /  .\scripts\up.ps1
make health                    # probe API + Qdrant
make logs                      # tail logs
make down                      # stop, keep the vector index
```

- API: http://localhost:8000/docs
- Qdrant dashboard: http://localhost:6333/dashboard

## Scripts

| Entry point | Purpose |
|-------------|---------|
| `make up` / `scripts/up.ps1` | Build and start the stack, wait for health |
| `make down` / `scripts/down.ps1` | Stop the stack (`-Purge` also drops the vector volume) |
| `make health` / `scripts/health.ps1` | Probe every health surface |
| `scripts/smoke_health.py` | Dependency-free smoke test for a running stack |
| `make test` | Run the test suite inside the API container |

## Documentation

- [Architecture](docs/architecture.md)
- [Data model](docs/data-model.md)
- [Runbook](docs/runbook.md)
- [Evaluation](docs/evaluation.md)
- ADRs: [0001 hybrid Qdrant](docs/adr/0001-hybrid-qdrant.md) ·
  [0002 LangGraph query pipeline](docs/adr/0002-langgraph-query.md) ·
  [0003 eval strategy](docs/adr/0003-eval-strategy.md)
