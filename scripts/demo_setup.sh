#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

test -d data/corpus || { echo "data/corpus is missing; integrate the wave 8 corpus first." >&2; exit 1; }
command -v docker >/dev/null || { echo "docker is required." >&2; exit 1; }

export QDRANT_COLLECTION=prag_demo
docker compose up -d qdrant

for _ in $(seq 1 60); do
  if curl --fail --silent http://localhost:6333/readyz >/dev/null; then
    break
  fi
  sleep 2
done
curl --fail --silent http://localhost:6333/readyz >/dev/null || {
  echo "Qdrant did not become ready within 120 seconds." >&2
  exit 1
}

docker compose build api
docker compose run --rm api python -c "from importlib.metadata import version; import httpx; client=version('qdrant-client'); server=httpx.get('http://qdrant:6333/').json()['version']; assert client == server, f'Qdrant version mismatch: client={client} server={server}'"

docker compose run --rm api python -m production_rag.ingest \
  --config configs/default.yaml \
  --source data/corpus \
  --embedder fake \
  --collection prag_demo \
  --recreate-collection
docker compose up -d api

printf '\nDemo ready: http://localhost:8000/\n'
printf '1. Why does hybrid search use reciprocal rank fusion?\n'
printf '2. Who won the Antarctic underwater chess championship?\n'
