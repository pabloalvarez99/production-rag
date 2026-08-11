"""Public request and response models for the HTTP API.

Every route declares an explicit response model. Two reasons, both of which
matter more as the API grows: the OpenAPI document stays accurate for free, and
a field can never leak into a response by accident because the model — not the
handler's return value — decides what is serialised.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class QueryRequest(BaseModel):
    """One grounded-answer request.

    Unknown fields are rejected so a misspelled control never silently falls
    back to a default. Whitespace-only questions are rejected by Pydantic after
    global string stripping, before the query pipeline or a provider is called.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question: str = Field(
        min_length=1,
        max_length=8_000,
        description="Natural-language question to answer from indexed evidence.",
    )
    mode: Literal["dense", "sparse", "hybrid"] | None = Field(
        default=None,
        description="Retrieval mode override; omitted uses the configured default.",
    )
    rerank: Literal["off", "auto", "fake", "local", "cohere"] | None = Field(
        default=None,
        description="Reranker override; omitted uses the query pipeline default.",
    )
    llm: Literal["fake", "openai"] = Field(
        default="fake",
        description="Generation provider. 'fake' is deterministic offline plumbing only.",
    )
    debug: bool = Field(
        default=False,
        description="Ask the pipeline to collect diagnostics; never exposes credentials.",
    )


class CitationOut(BaseModel):
    """A resolved answer marker pointing to one retrieved chunk."""

    model_config = ConfigDict(from_attributes=True)

    marker: int = Field(ge=1, description="One-based marker ordinal used in the answer.")
    chunk_id: str = Field(description="Stable identifier of the supporting chunk.")
    source_path: str = Field(description="Corpus-relative source path.")
    text: str = Field(description="Supporting passage exactly as it was shown to the model.")
    rank: int = Field(ge=1, description="Position of the passage in the retrieved ranking.")
    title: str | None = Field(default=None, description="Source document title, when known.")
    heading_path: str | None = Field(
        default=None,
        description="Heading ancestry within the source document, when known.",
    )


class QueryResponse(BaseModel):
    """Grounded answer or an explicit refusal."""

    model_config = ConfigDict(from_attributes=True)

    answer: str = Field(description="Answer text with inline citation markers, or refusal text.")
    citations: list[CitationOut] = Field(
        default_factory=list,
        description="Resolved evidence referenced by the answer.",
    )
    refused: bool = Field(description="Whether the evidence guardrail declined to answer.")
    refusal_reason: str | None = Field(
        default=None,
        description="Machine-readable refusal explanation; null for grounded answers.",
    )


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
