"""Readiness endpoint.

Readiness answers a different question from liveness: *should traffic be sent
here?* At M0 the honest answer is "yes, if configuration parsed", because there
is nothing downstream to be unready for yet. The response therefore carries the
per-subsystem verdicts rather than folding them into the status code, so the
shape stays stable when M1 adds a real Qdrant round-trip and this route starts
returning 503.
"""

from __future__ import annotations

from fastapi import APIRouter, status

from production_rag.api.deps import SettingsDep
from production_rag.api.schemas import ReadyResponse

router = APIRouter(tags=["health"])
"""Versioned router; mounted under ``Settings.api_prefix``."""


@router.get(
    "/ready",
    response_model=ReadyResponse,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {"description": "Configuration is valid; the service accepts traffic."},
    },
    summary="Readiness probe",
    operation_id="ready",
)
async def ready(settings: SettingsDep) -> ReadyResponse:
    """Report configuration readiness without performing any network IO.

    ``qdrant_configured`` reflects whether ``QDRANT_URL`` names an http(s)
    endpoint. It deliberately does not open a socket: a probe that dials a
    downstream on every call turns a slow dependency into a restart loop, and
    it would make the M0 test suite depend on a running Qdrant.
    """
    return ReadyResponse(
        status="ready",
        qdrant_configured=settings.qdrant_configured,
        checks={"settings": "ok"},
    )
