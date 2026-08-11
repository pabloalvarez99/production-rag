"""Application factory and logging setup.

``create_app`` is a factory, not a module-level assembly, for one reason: tests
build an app around explicit settings instead of mutating the environment, so
no state leaks between test cases. Uvicorn and the container consume the
module-level ``app`` (``uvicorn production_rag.main:app``).
"""

from __future__ import annotations

import logging

import structlog
from fastapi import FastAPI
from structlog.typing import Processor

from production_rag.api.middleware import RequestContextMiddleware
from production_rag.api.routes import health, query, ready
from production_rag.config import Settings, get_settings

API_DESCRIPTION = """
Production-grade Retrieval-Augmented Generation service.

**Milestone M0 — scaffold.** Only the operational surface exists: liveness,
readiness and the OpenAPI document. Ingestion, hybrid retrieval, reranking,
grounded answers with citations and the evaluation gate arrive in M1-M6; see
the roadmap in the README.
""".strip()


def configure_logging(level: str = "INFO", *, json_logs: bool = False) -> None:
    """Configure structlog (and the stdlib root logger) for the process.

    Renderer choice is deliberate: JSON when running anywhere but a developer's
    machine, because a log line that a human enjoys reading is a log line no
    aggregator can query. ``merge_contextvars`` is what makes the per-request
    correlation id show up on every event without being passed around.
    """
    numeric_level = logging.getLevelNamesMapping().get(level.upper(), logging.INFO)

    # Uvicorn and any library using the stdlib logger share the same level and
    # a bare format, so structlog's rendering is the only formatting in play.
    logging.basicConfig(format="%(message)s", level=numeric_level, force=True)

    shared: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
    ]
    renderer: Processor = (
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer(colors=False)
    )
    structlog.configure(
        processors=[*shared, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.WriteLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application.

    Args:
        settings: Explicit configuration. When omitted the cached process
            settings are used, which is what the container and uvicorn get.
            Passing an instance also registers a dependency override so the
            routes see exactly these settings — that is the seam the tests use.

    Returns:
        A fully wired application: request-context middleware, the unversioned
        infrastructure probe and the versioned operational routes.
    """
    resolved = settings if settings is not None else get_settings()
    configure_logging(resolved.log_level, json_logs=resolved.environment != "local")

    app = FastAPI(
        title=resolved.app_name,
        version=resolved.app_version,
        description=API_DESCRIPTION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        contact={"name": "production-rag"},
        license_info={"name": "MIT", "identifier": "MIT"},
    )
    app.state.settings = resolved
    if settings is not None:
        app.dependency_overrides[get_settings] = lambda: resolved

    app.add_middleware(RequestContextMiddleware)

    # Unversioned first: infrastructure probes hit /health.
    app.include_router(health.unversioned_router)
    app.include_router(health.router, prefix=resolved.api_prefix)
    app.include_router(ready.router, prefix=resolved.api_prefix)
    app.include_router(query.router, prefix=resolved.api_prefix)

    structlog.get_logger(__name__).info(
        "app_created",
        service=resolved.app_name,
        version=resolved.app_version,
        environment=resolved.environment,
        api_prefix=resolved.api_prefix,
        qdrant_configured=resolved.qdrant_configured,
    )
    return app


app = create_app()
"""ASGI entrypoint consumed by uvicorn and the container image."""
