"""Grounded query HTTP adapter.

The route owns transport concerns only: request validation, correlation-id
forwarding and response projection. Retrieval and generation remain behind
``run_query``. During a split milestone checkout where A1's pipeline has not
landed yet, the same endpoint fails honestly with 503 instead of growing a
second implementation here.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from importlib import import_module
from typing import Annotated, Any, Protocol, cast

from fastapi import APIRouter, Depends, HTTPException, Request, status

from production_rag.api.deps import SettingsDep
from production_rag.api.middleware import get_request_id
from production_rag.api.schemas import QueryRequest, QueryResponse
from production_rag.config import Settings
from production_rag.config_loader import load_yaml_config
from production_rag.ingest.cli import resolve_embedder
from production_rag.retrieval.cli import resolve_searchable_store
from production_rag.retrieval.hybrid import Retriever
from production_rag.retrieval.rerank import RERANK_AUTO, build_reranker


def _load_query_symbols() -> tuple[Callable[..., Any] | None, Callable[..., Any] | None]:
    """Load A1's entrypoints without making split-branch imports fatal."""
    try:
        pipeline_module = import_module("production_rag.query_pipeline")
        llm_module = import_module("production_rag.generation.llm")
    except ImportError:  # pragma: no cover - covered through the public 503 behaviour
        return None, None
    return (
        cast(Callable[..., Any], pipeline_module.run_query),
        cast(Callable[..., Any], llm_module.build_llm),
    )


_run_query, _build_llm = _load_query_symbols()


class QueryExecutor(Protocol):
    """Transport-facing seam around A1's library entrypoint."""

    def __call__(
        self,
        payload: QueryRequest,
        *,
        settings: Settings,
        request_id: str,
    ) -> QueryResponse:
        """Execute one validated query request."""
        ...


class QueryPipelineUnavailableError(RuntimeError):
    """The checkout does not contain the A1 query pipeline yet."""


router = APIRouter(tags=["query"])
"""Versioned router; mounted under ``Settings.api_prefix``."""


def execute_query(
    payload: QueryRequest,
    *,
    settings: Settings,
    request_id: str,
    embedder_kind: str | None = None,
) -> QueryResponse:
    """Call A1's public pipeline and project its result onto the API schema."""
    if _run_query is None or _build_llm is None:
        raise QueryPipelineUnavailableError("query pipeline not installed")

    config = load_yaml_config(settings.config_path)
    collection = (
        settings.qdrant_collection
        if "qdrant_collection" in settings.model_fields_set
        else config.qdrant.collection
    )
    embedder = resolve_embedder(
        embedder_kind or config.ingest.embedding.provider,
        config=config,
        settings=settings,
    )
    store = resolve_searchable_store(
        config=config,
        settings=settings,
        collection=collection,
        url=settings.qdrant_url,
    )
    reranker = build_reranker(
        payload.rerank or RERANK_AUTO,
        config=config.rerank,
        api_key=os.environ.get(config.rerank.api_key_env),
    )
    llm = _build_llm(
        payload.llm,
        config=config.generation,
        api_key=settings.openai_api_key or os.environ.get(config.generation.api_key_env),
    )
    retriever = Retriever.from_config(store=store, embedder=embedder, config=config)
    result = _run_query(
        payload.question,
        retriever=retriever,
        llm=llm,
        config=config,
        mode=payload.mode,
        reranker=reranker,
    )
    return QueryResponse.model_validate(result, from_attributes=True)


def get_query_executor() -> QueryExecutor:
    """Return the query adapter; tests override this dependency with a fake."""
    return execute_query


QueryExecutorDep = Annotated[QueryExecutor, Depends(get_query_executor)]
"""Injected query executor; replaced by an offline fake in API unit tests."""


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {"description": "Grounded answer or explicit refusal."},
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "Query pipeline is absent from this split milestone checkout."
        },
    },
    summary="Answer a question from indexed evidence",
    operation_id="query",
)
def query(
    payload: QueryRequest,
    request: Request,
    settings: SettingsDep,
    executor: QueryExecutorDep,
) -> QueryResponse:
    """Return a grounded answer without exposing internal pipeline metadata."""
    try:
        return executor(
            payload,
            settings=settings,
            request_id=get_request_id(request),
        )
    except QueryPipelineUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


__all__ = [
    "QueryExecutor",
    "QueryPipelineUnavailableError",
    "execute_query",
    "get_query_executor",
    "router",
]
