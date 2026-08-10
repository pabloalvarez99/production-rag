# Runbook

Operational procedures for the local stack. Everything goes through
`docker compose` (or the `Makefile` / `scripts/*.ps1` wrappers, which are
thin shims over the same commands).

## Start the stack

```bash
make up                 # or: docker compose up -d --build
# Windows without make:
.\scripts\up.ps1
```

Wait for health, then verify:

```bash
make health             # probes API /health, /v1/health and Qdrant /readyz
python scripts/smoke_health.py
.\scripts\health.ps1    # PowerShell equivalent
```

Expected: all three probes return 2xx. The Qdrant dashboard is at
http://localhost:6333/dashboard, API docs at http://localhost:8000/docs.

## Day-to-day commands

| Task | Command |
|------|---------|
| Status + health | `make ps` |
| Tail logs | `make logs` |
| Rebuild after dependency change | `make build` |
| Recreate API only | `make restart` |
| Stop, keep index | `make down` |
| Run tests in container | `make test` |

## Common failures

**`api` unhealthy, Qdrant healthy.**
Check `make logs` for the API container. Most common cause in M0: missing
`OPENAI_API_KEY` when a query path that needs a provider is exercised.
Copy `.env.example` to `.env` and fill it in, then `make restart`.

**Qdrant container restarting in a loop.**
Almost always a corrupted storage volume or a port conflict on 6333/6334.
Check `docker compose logs qdrant`. If the volume is corrupt and the corpus
is re-ingestible, rebuild it: `make clean && make up`, then re-run ingest.

**Port already in use (`8000`, `6333`, `6334`).**
Another compose project or a local dev server is holding the port.
`docker ps` to find it; stop the offender or remap ports in
`docker-compose.yml`.

**Collection missing after `up`.**
The volume survived but ingest has not run (or ran against a different
`QDRANT_COLLECTION`). `scripts/health.ps1` reports this as a warning, not a
failure — an empty stack before ingest is a valid state.

## Destructive operations

```bash
make clean              # docker compose down -v --remove-orphans
.\scripts\down.ps1 -Purge
```

Both delete the `qdrant_storage` volume. The vector index is gone and must be
rebuilt by re-running ingest. This is safe only because `data/raw/` is the
source of truth.

## Upgrading Qdrant

The image tag is pinned (`qdrant/qdrant:v1.13.2`) on purpose. To upgrade:
bump the tag, `make clean` (storage format migrations are one-way), `make
up`, re-ingest, and re-run the eval suite before trusting the new version
(see [evaluation.md](evaluation.md)).
