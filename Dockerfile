# syntax=docker/dockerfile:1
#
# Runtime image for the production-rag API.
#
# Single stage on purpose: the dependency set is pure-Python wheels, so a
# builder stage would save little and cost a layer of complexity. Revisit if a
# native extension (e.g. a local reranker with torch) ever enters the runtime
# path -- at that point split into builder + runtime and copy only the venv.

FROM python:3.12-slim

# - PYTHONDONTWRITEBYTECODE: no .pyc in a read-only-ish container layer
# - PYTHONUNBUFFERED: logs reach `docker logs` immediately, not at flush time
# - PIP_DISABLE_PIP_VERSION_CHECK: one less network call per build
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# curl is required by the compose healthcheck; ca-certificates by any outbound
# HTTPS call (embeddings, LLM). Both are small and stay in the final image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Dependency metadata first so the pip layer is cached across source edits.
# README.md is copied because most PEP 621 backends resolve `readme = ...`
# at build time and hard-fail when the file is missing.
COPY pyproject.toml README.md ./

# The package lives under src/ (src layout). Copied before install because an
# editable install still needs the tree to exist to write the path hook.
COPY src/ ./src/

RUN pip install --no-cache-dir -e .

# Config and data are also bind-mounted read-only by compose; baking them in
# keeps `docker run` (without compose) working standalone.
COPY configs/ ./configs/

# Non-root runtime. Owning /app lets an editable install rewrite egg-info if a
# later `pip install -e .` runs inside the container during debugging.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Image-level healthcheck mirrors the compose one so plain `docker run` and
# orchestrators that ignore compose still get liveness signal.
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

# No --reload: this is the production-shaped entrypoint. For hot reload use
# `make dev` / uvicorn on the host against a compose-run Qdrant.
CMD ["uvicorn", "production_rag.main:app", "--host", "0.0.0.0", "--port", "8000"]
