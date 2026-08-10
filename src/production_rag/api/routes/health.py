"""Liveness endpoints.

The same payload is served twice, on purpose:

* ``GET /health`` — unversioned. Container and orchestrator probes are wired
  once, often by someone who is not the API's author (see the Compose and
  Dockerfile healthchecks). They must not break when the API version bumps.
* ``GET /v1/health`` — versioned. API clients stay inside a single namespace
  and never have to special-case one path.

Two routers rather than the same router mounted under two prefixes: mounting
one router twice produces duplicate ``operationId`` values, which makes the
OpenAPI document invalid for client generators.
"""

from __future__ import annotations

from fastapi import APIRouter, status

from production_rag.api.deps import SettingsDep
from production_rag.api.schemas import HealthResponse
from production_rag.config import Settings

router = APIRouter(tags=["ops"])
"""Versioned router; mounted under ``Settings.api_prefix``."""

unversioned_router = APIRouter(tags=["ops"])
"""Unversioned router; mounted at the root for infrastructure probes."""

_RESPONSES: dict[int | str, dict[str, str]] = {
    status.HTTP_200_OK: {"description": "Process is alive and serving."},
}


def _health_payload(settings: Settings) -> HealthResponse:
    """Build the liveness payload from configuration alone (no IO)."""
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    responses=_RESPONSES,
    summary="Liveness probe",
    operation_id="health",
)
async def health(settings: SettingsDep) -> HealthResponse:
    """Report that the process is up. Never touches a dependency."""
    return _health_payload(settings)


@unversioned_router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    responses=_RESPONSES,
    summary="Liveness probe (unversioned, for infrastructure)",
    operation_id="health_unversioned",
)
async def health_unversioned(settings: SettingsDep) -> HealthResponse:
    """Same payload as :func:`health`, on a path that never changes."""
    return _health_payload(settings)
