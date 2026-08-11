# production-rag — operator entrypoints.
#
# Every target is a thin wrapper over docker compose so there is exactly one
# way to run the stack. Windows users without make have PowerShell equivalents
# in scripts/ (up.ps1, down.ps1, health.ps1) — keep the two in sync.

COMPOSE ?= docker compose
API     ?= api
BASE_URL ?= http://localhost:8000

# Ingest knobs. SOURCE is the corpus *root*: payload `source_path` values are
# relative to it, and its first path segment becomes the filterable `source`
# field. Keeping it at data/raw is what makes a chunk of the sample corpus read
# as `sample/08-bm25-vs-dense.md` — which is exactly what data/eval/golden.jsonl
# labels. Point it deeper and those labels stop matching.
#   make ingest-fake SOURCE=data/raw/my-corpus
SOURCE      ?= data/raw
CONFIG_FILE ?= configs/default.yaml
INGEST      := python -m production_rag.ingest --config $(CONFIG_FILE)
RETRIEVE    := python -m production_rag.retrieval --config $(CONFIG_FILE)

# Retrieval knobs (M2). QUERY has no sensible default: retrieve-fake without one
# would silently score a question nobody asked.
QUERY ?=
MODE  ?= hybrid
TOPK  ?=

.DEFAULT_GOAL := help
.PHONY: help build up down restart logs ps health health-ready ingest-fake ingest-sample \
        ingest-dry reingest-fake retrieve-fake retrieve-sample eval-hit-fake eval-hit-sample \
        test test-host shell-api shell-qdrant clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

build: ## Build the API image (no cache reuse for the source layer)
	$(COMPOSE) build

up: ## Build if needed and start the stack detached
	$(COMPOSE) up -d --build
	@echo "API      -> $(BASE_URL)/docs"
	@echo "Qdrant   -> http://localhost:6333/dashboard"

down: ## Stop the stack, keep the vector volume
	$(COMPOSE) down

restart: ## Recreate the API container only (fastest edit loop for ops changes)
	$(COMPOSE) up -d --build --force-recreate --no-deps $(API)

logs: ## Tail logs from all services
	$(COMPOSE) logs -f --tail=100

ps: ## Show service status and health
	$(COMPOSE) ps

health: ## Probe API liveness and Qdrant readiness
	@curl -fsS $(BASE_URL)/health && echo ""
	@curl -fsS http://localhost:6333/readyz && echo ""

health-ready: health ## Also probe versioned readiness (needs Qdrant reachable)
	@curl -fsS $(BASE_URL)/v1/ready && echo ""

# ---------------------------------------------------------------------------
# Ingest (M1). Runs inside the api container so QDRANT_URL resolves to the
# compose hostname and no host-side install is needed. Windows without make:
# scripts/ingest.ps1 takes the same options.
# ---------------------------------------------------------------------------

# The fake embedder is a deterministic hash embedder: no API key, no network,
# no spend, and identical text always yields an identical vector. That makes the
# full walk -> chunk -> embed -> upsert path runnable in CI. The vectors say
# nothing about relevance, so never read retrieval numbers off this target.
ingest-fake: ## Ingest the corpus at $(SOURCE) with the deterministic fake embedder (no API key)
	$(COMPOSE) run --rm $(API) $(INGEST) --source $(SOURCE) --embedder fake

# Real embeddings. OPENAI_API_KEY comes from the environment or the gitignored
# .env that Compose loads — never from the command line, where it would land in
# shell history and in `docker inspect`.
ingest-sample: ## Ingest $(SOURCE) with the real provider embedder (needs OPENAI_API_KEY)
	$(COMPOSE) run --rm $(API) $(INGEST) --source $(SOURCE) --embedder openai

ingest-dry: ## Walk and chunk $(SOURCE), report counts, write nothing
	$(COMPOSE) run --rm $(API) $(INGEST) --source $(SOURCE) --embedder fake --dry-run

# M2 migration. A collection created by M1 carries only the `dense` named
# vector -- M1 never declared `sparse`, despite what earlier docs claimed -- so
# hybrid retrieval cannot run against it and it cannot be upgraded in place.
# This drops it and re-ingests. Free on the fake embedder; the openai equivalent
# is a full billed re-embed of every chunk, which is why there is no
# reingest-sample target: run `ingest-sample` with --recreate deliberately.
reingest-fake: ## DESTRUCTIVE. Drop the collection and re-ingest with sparse vectors (M2)
	$(COMPOSE) run --rm $(API) $(INGEST) --source $(SOURCE) --embedder fake --recreate-collection

# ---------------------------------------------------------------------------
# Retrieve (M2). Dense + sparse branches, fused with RRF, printed as ranked
# hits. No rerank (M3), no answer, no citations (M4) -- this returns passages.
# The embedder must match the one that built the collection: nothing detects a
# mismatch, both produce 1536 dimensions.
#   make retrieve-fake QUERY="how does reciprocal rank fusion work"
#   make retrieve-fake QUERY="QDRANT__SERVICE__GRPC_PORT" MODE=sparse
# ---------------------------------------------------------------------------

retrieve-fake: ## Query the collection with the fake embedder. QUERY="..." [MODE=] [TOPK=]
	@test -n '$(QUERY)' || { echo 'usage: make retrieve-fake QUERY="your question"'; exit 2; }
	$(COMPOSE) run --rm $(API) $(RETRIEVE) --query '$(QUERY)' --mode $(MODE) --embedder fake \
		$(if $(TOPK),--top-k $(TOPK),)

retrieve-sample: ## Same, with real embeddings (needs OPENAI_API_KEY). QUERY="..."
	@test -n '$(QUERY)' || { echo 'usage: make retrieve-sample QUERY="your question"'; exit 2; }
	$(COMPOSE) run --rm $(API) $(RETRIEVE) --query '$(QUERY)' --mode $(MODE) --embedder openai \
		$(if $(TOPK),--top-k $(TOPK),)

# ---------------------------------------------------------------------------
# Evaluate (M2). Source-level hit@k over data/eval/golden.jsonl. Reports only:
# no thresholds, no gate, no non-zero exit on a low score -- gating a 14-item
# set would be theatre, and the harness is M6. scripts/ is excluded from the
# image (.dockerignore), hence the mount.
# ---------------------------------------------------------------------------

eval-hit-fake: ## Score hit@k on a fake-embedded collection (plumbing, not quality)
	$(COMPOSE) run --rm -v "$(CURDIR)/scripts:/app/scripts:ro" \
		$(API) python scripts/eval_hit.py --config $(CONFIG_FILE) --embedder fake

eval-hit-sample: ## Score hit@k with real embeddings (needs OPENAI_API_KEY; costs money)
	$(COMPOSE) run --rm -v "$(CURDIR)/scripts:/app/scripts:ro" \
		$(API) python scripts/eval_hit.py --config $(CONFIG_FILE) --embedder openai

# tests/ is excluded from the image (see .dockerignore) so it is mounted here.
# Requires the dev extra to be part of the installed dependency set; if pytest
# is dev-only and not installed in the image, use `make test-host` instead.
test: ## Run the test suite inside the API container
	$(COMPOSE) run --rm --no-deps \
		-v "$(CURDIR)/tests:/app/tests:ro" \
		$(API) python -m pytest -q /app/tests

test-host: ## Run the test suite on the host (needs a local venv with dev extras)
	python -m pytest -q

shell-api: ## Interactive shell in the running API container
	$(COMPOSE) exec $(API) /bin/bash

shell-qdrant: ## Interactive shell in the running Qdrant container
	$(COMPOSE) exec qdrant /bin/bash

clean: ## Stop the stack AND delete the vector volume (destructive)
	$(COMPOSE) down -v --remove-orphans
