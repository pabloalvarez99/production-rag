# production-rag — operator entrypoints.
#
# Every target is a thin wrapper over docker compose so there is exactly one
# way to run the stack. Windows users without make have PowerShell equivalents
# in scripts/ (up.ps1, down.ps1, health.ps1) — keep the two in sync.

COMPOSE ?= docker compose
API     ?= api
BASE_URL ?= http://localhost:8000

.DEFAULT_GOAL := help
.PHONY: help build up down restart logs ps health test test-host shell-api shell-qdrant clean

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

health: ## Probe both health endpoints and Qdrant readiness
	@curl -fsS $(BASE_URL)/health && echo ""
	@curl -fsS $(BASE_URL)/v1/health && echo ""
	@curl -fsS http://localhost:6333/readyz && echo ""

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
