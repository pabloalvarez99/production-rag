"""Response models for the operational endpoints.

Every route declares an explicit response model. Two reasons, both of which
matter more as the API grows: the OpenAPI document stays accurate for free, and
a field can never leak into a response by accident because the model — not the
handler's return value — decides what is serialised.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Liveness payload: the process is up and can serve requests.

    Intentionally free of dependency state. A liveness probe that fails when a
    downstream is unavailable causes the orchestrator to restart a perfectly
    healthy process, which does not fix the downstream and drops in-flight
    requests. Dependency state belongs in :class:`ReadyResponse`.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "status": "ok",
                    "service": "production-rag",
                    "version": "0.1.0",
                    "environment": "local",
                }
            ]
        }
    )

    status: Literal["ok"] = Field(
        default="ok",
        description="Constant marker; the HTTP status code carries the real signal.",
    )
    service: str = Field(description="Logical service name, from APP_NAME.")
    version: str = Field(description="Deployed package version.")
    environment: str | None = Field(
        default=None,
        description="Deployment environment, e.g. local / staging / production.",
    )


class ReadyResponse(BaseModel):
    """Readiness payload: the process is up *and* configured to do useful work.

    At M0 the only check is that configuration parsed and that a vector-store
    endpoint is set. From M1 ``checks`` grows one entry per dependency that is
    actually contacted, and the route starts answering 503 when a required one
    is down — that is the signal a load balancer should act on.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "status": "ready",
                    "qdrant_configured": True,
                    "checks": {"settings": "ok"},
                }
            ]
        }
    )

    status: Literal["ready"] = Field(default="ready", description="Readiness marker.")
    qdrant_configured: bool = Field(
        description=(
            "Whether QDRANT_URL names an http(s) endpoint. No connection is attempted at M0."
        )
    )
    checks: dict[str, str] = Field(
        default_factory=dict,
        description="Per-subsystem verdicts. One entry per dependency as milestones land.",
    )
