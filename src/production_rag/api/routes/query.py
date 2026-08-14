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
from inspect import signature
from typing import Annotated, Any, Protocol, cast

from fastapi import APIRouter, Depends, HTTPException, Request, status

from production_rag.api.deps import SettingsDep
from production_rag.api.middleware import get_request_id
from production_rag.api.schemas import QueryDebug, QueryRequest, QueryResponse
from production_rag.config import Settings
from production_rag.config_loader import YamlConfig, load_yaml_config
from production_rag.ingest.cli import resolve_embedder
from production_rag.retrieval.cli import resolve_searchable_store
from production_rag.retrieval.filters import FilterError, FilterPolicy
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

HTTP_422_UNPROCESSABLE = 422
"""The code an unanswerable-but-well-formed body gets.

Written as a number, like ``ui.py`` already does, because Starlette renamed the
constant (``..._ENTITY`` to ``..._CONTENT``) and importing either name pins this
module to a version range for no benefit. The number is the contract."""


def _accepts_request_id(query_callable: Callable[..., Any]) -> bool:
    """Return whether an A1 entrypoint accepts ``request_id`` as a keyword."""
    try:
        signature(query_callable).bind_partial("question", request_id="request-id")
    except (TypeError, ValueError):
        return False
    return True


def _project_debug(result: Any) -> QueryDebug:
    """Project internal results onto the deliberately small public allowlist."""
    return QueryDebug(
        timings_ms=dict(getattr(result, "timings_ms", None) or {}),
        invalid_markers=list(getattr(result, "invalid_markers", ()) or ()),
    )


def filter_policy(config: YamlConfig) -> FilterPolicy:
    """The filter allowlist and index map this profile describes.

    Built from the file in one place so the route, the CLI and the retriever all
    answer "is this field allowed?" from the same two config keys.
    """
    return FilterPolicy.from_fields(
        config.retrieval.filters.allowed_fields,
        [index.field for index in config.qdrant.payload_indexes],
    )


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
    # Validated at the edge, before an embedder, a store or a model is
    # constructed: a filter this deployment does not allow is a client mistake,
    # and finding out should cost nothing. The retriever validates again with the
    # same policy object — one implementation, so the two cannot disagree.
    if payload.filters:
        filter_policy(config).build(payload.filters)
    collection = (
        settings.qdrant_collection
        if "qdrant_collection" in settings.model_fields_set
        else config.qdrant.collection
    )
    embedder = resolve_embedder(
        embedder_kind or payload.embedder,
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
    query_kwargs: dict[str, Any] = {
        "retriever": retriever,
        "llm": llm,
        "config": config,
        "mode": payload.mode,
        "reranker": reranker,
    }
    if payload.filters:
        # Only when asked for, so an unfiltered request calls the pipeline with
        # exactly the arguments it took before this field existed.
        query_kwargs["filters"] = payload.filters
    if _accepts_request_id(_run_query):
        query_kwargs["request_id"] = request_id

    result = _run_query(payload.question, **query_kwargs)
    response = QueryResponse.model_validate(result, from_attributes=True)
    if payload.debug:
        return response.model_copy(update={"debug": _project_debug(result)})
    return response


def get_query_executor() -> QueryExecutor:
    """Return the query adapter; tests override this dependency with a fake."""
    return execute_query


QueryExecutorDep = Annotated[QueryExecutor, Depends(get_query_executor)]
"""Injected query executor; replaced by an offline fake in API unit tests."""


@router.post(
    "/query",
    response_model=QueryResponse,
    response_model_exclude_unset=True,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {"description": "Grounded answer or explicit refusal."},
        HTTP_422_UNPROCESSABLE: {
            "description": (
                "The body failed validation, or `filters` names a field outside "
                "`retrieval.filters.allowed_fields`. A filter rejection carries "
                "`error_type` in the detail object."
            )
        },
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
    except FilterError as exc:
        # 422, the same code an unknown body field already gets: the request is
        # well-formed and unanswerable as written. The detail is a typed object
        # rather than a sentence, so a client branches on `error_type` and never
        # on wording that may change.
        raise HTTPException(
            status_code=HTTP_422_UNPROCESSABLE,
            detail={
                "error_type": exc.error_type,
                "field": exc.field,
                "message": str(exc),
            },
        ) from exc
    except QueryPipelineUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


__all__ = [
    "QueryExecutor",
    "QueryPipelineUnavailableError",
    "execute_query",
    "filter_policy",
    "get_query_executor",
    "router",
]
