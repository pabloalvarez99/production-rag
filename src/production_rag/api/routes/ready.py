"""Readiness endpoint.

Readiness answers a different question from liveness: *should traffic be sent
here?* Configuration still parses offline; collection identity is added so a
reviewer can see *which* index the process believes it owns without dialing
Qdrant on every probe.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, status

from production_rag.api.deps import SettingsDep
from production_rag.api.schemas import ReadyResponse
from production_rag.corpus_identity import (
    build_corpus_identity,
    default_identity_path,
    load_identity_sidecar,
)

router = APIRouter(tags=["health"])
"""Versioned router; mounted under ``Settings.api_prefix``."""


def resolve_collection_identity(settings: SettingsDep) -> dict[str, object]:
    """Load sidecar identity or compute from the configured corpus root.

    Offline and free-path safe: hashing local files is allowed; opening a
    Qdrant socket is not.
    """
    collection = settings.qdrant_collection
    sidecar = load_identity_sidecar(default_identity_path(collection))
    if sidecar is not None:
        return {
            "embedder_id": sidecar.get("embedder_id"),
            "chunker_version": sidecar.get("chunker_version"),
            "doc_count": sidecar.get("doc_count"),
            "corpus_hash": sidecar.get("corpus_hash"),
            "collection": sidecar.get("collection") or collection,
            "source": "sidecar",
        }
    root = Path(settings.corpus_root)
    if not root.exists():
        return {
            "embedder_id": settings.ready_embedder_id,
            "chunker_version": None,
            "doc_count": None,
            "corpus_hash": None,
            "collection": collection,
            "source": "missing_corpus",
        }
    identity = build_corpus_identity(
        corpus_root=root,
        embedder_id=settings.ready_embedder_id,
        collection=collection,
    )
    payload = identity.as_public_dict()
    payload["source"] = "computed"
    return payload


@router.get(
    "/ready",
    response_model=ReadyResponse,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "description": "Configuration is valid; collection identity is advertised when known."
        },
    },
    summary="Readiness probe",
    operation_id="ready",
)
async def ready(settings: SettingsDep) -> ReadyResponse:
    """Report configuration readiness without dialing Qdrant.

    ``qdrant_configured`` reflects whether ``QDRANT_URL`` names an http(s)
    endpoint. Collection identity fields come from a sidecar written at ingest
    or from a free-path hash of ``CORPUS_ROOT`` / ``corpus_root``.
    """
    identity = resolve_collection_identity(settings)
    checks = {"settings": "ok", "identity": str(identity.get("source", "unknown"))}
    raw_docs = identity.get("doc_count")
    doc_count = raw_docs if isinstance(raw_docs, int) else None
    raw_embedder = identity.get("embedder_id")
    raw_chunker = identity.get("chunker_version")
    raw_hash = identity.get("corpus_hash")
    return ReadyResponse(
        status="ready",
        qdrant_configured=settings.qdrant_configured,
        checks=checks,
        embedder_id=str(raw_embedder) if raw_embedder is not None else None,
        chunker_version=str(raw_chunker) if raw_chunker is not None else None,
        doc_count=doc_count,
        corpus_hash=str(raw_hash) if raw_hash is not None else None,
        collection=str(identity.get("collection") or settings.qdrant_collection),
    )
