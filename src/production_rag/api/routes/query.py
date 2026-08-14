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
from production_rag.config_loader import ConfigFileError, YamlConfig, load_yaml_config
from production_rag.corpus_identity import (
    WrongCollectionError,
    default_identity_path,
    load_identity_sidecar,
)
from production_rag.generation.llm import LLMError
from production_rag.generation.streaming import DeltaSink, StreamingTee
from production_rag.ingest.cli import resolve_embedder
from production_rag.query_cache import (
    CacheKey,
    CacheStatus,
    canonical_filters,
    get_query_cache,
    retrieval_fingerprint,
)
from production_rag.retrieval.cli import resolve_searchable_store
from production_rag.retrieval.embeddings import EmbeddingError
from production_rag.retrieval.filters import FilterError, FilterPolicy
from production_rag.retrieval.hybrid import Retriever
from production_rag.retrieval.rerank import RERANK_AUTO, build_reranker
from production_rag.retrieval.store import CollectionMismatchError, VectorStoreError


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


class StreamingQueryExecutor(Protocol):
    """The same seam, plus somewhere to publish provisional model output.

    Separate from :class:`QueryExecutor` rather than an optional argument on it,
    because a fake that satisfies the plain protocol must keep satisfying it: a
    test written before streaming existed should not have to learn about a sink
    it never uses. :func:`execute_query` satisfies both.
    """

    def __call__(
        self,
        payload: QueryRequest,
        *,
        settings: Settings,
        request_id: str,
        on_delta: DeltaSink | None = None,
    ) -> QueryResponse:
        """Execute one validated query request, publishing chunks to *on_delta*."""
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


def _project_debug(
    result: Any,
    *,
    cache_status: CacheStatus | None = None,
) -> QueryDebug:
    """Project internal results onto the deliberately small public allowlist."""
    # cache is omitted unless the cache ran: a null field would still widen the
    # response shape for every debug caller and break clients that expect the
    # previous allowlist exactly.
    fields: dict[str, Any] = {
        "timings_ms": dict(getattr(result, "timings_ms", None) or {}),
        "invalid_markers": list(getattr(result, "invalid_markers", ()) or ()),
    }
    if cache_status is not None:
        fields["cache"] = cache_status
    return QueryDebug(**fields)


def _cache_enabled(settings: Settings, config: YamlConfig) -> bool:
    """Whether this process may reuse a prior answer.

    Either the YAML profile or the ``CACHE_ENABLED`` env flag is enough. The
    env flag is what the local compose demo sets; production-shaped profiles
    leave both false.
    """
    yaml_enabled = bool(getattr(getattr(config, "cache", None), "enabled", False))
    return bool(getattr(settings, "cache_enabled", False) or yaml_enabled)


def filter_policy(config: YamlConfig) -> FilterPolicy:
    """The filter allowlist and index map this profile describes.

    Built from the file in one place so the route, the CLI and the retriever all
    answer "is this field allowed?" from the same two config keys.
    """
    return FilterPolicy.from_fields(
        config.retrieval.filters.allowed_fields,
        [index.field for index in config.qdrant.payload_indexes],
    )


def deployment_filter_policy(settings: Settings) -> FilterPolicy:
    """The allowlist this deployment enforces, read from its profile.

    Read from configuration rather than hard-coded at each call site, so the
    field the UI offers, the field the JSON route accepts and the field the
    streamed route accepts cannot drift apart. A profile that cannot be parsed
    yields an **empty** policy: every filter is then rejected and any control
    built from it disappears, which is the fail-closed direction. Falling back
    to "allow everything" on an unreadable config would turn a deployment
    mistake into a data-exposure one.
    """
    try:
        config = load_yaml_config(settings.config_path)
    except ConfigFileError:
        return FilterPolicy()
    return filter_policy(config)


def execute_query(
    payload: QueryRequest,
    *,
    settings: Settings,
    request_id: str,
    embedder_kind: str | None = None,
    on_delta: DeltaSink | None = None,
) -> QueryResponse:
    """Call A1's public pipeline and project its result onto the API schema.

    ``on_delta`` is the streaming seam and changes nothing else: the model is
    wrapped in a :class:`~production_rag.generation.streaming.StreamingTee`, the
    pipeline is called with exactly the arguments it takes without it, and the
    returned response is the same object either way. Streaming is therefore
    additive by construction — there is no branch here where a streamed request
    could be answered by different code than a plain one.
    """
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
    if payload.collection is not None and payload.collection != collection:
        raise WrongCollectionError(
            f"collection {payload.collection!r} does not match process collection "
            f"{collection!r}",
            collection=payload.collection,
        )
    embedder_name = embedder_kind or payload.embedder
    identity_sidecar = load_identity_sidecar(default_identity_path(collection)) or {}
    corpus_identity_material = ""
    if identity_sidecar.get("corpus_hash"):
        corpus_identity_material = (
            f"{identity_sidecar.get('corpus_hash', '')}|"
            f"{identity_sidecar.get('chunker_version', '')}|"
            f"{identity_sidecar.get('doc_count', '')}|"
            f"{identity_sidecar.get('embedder_id', embedder_name)}"
        )
    cache_status: CacheStatus | None = None
    cache_key: CacheKey | None = None
    if _cache_enabled(settings, config):
        cache = get_query_cache(max_entries=config.cache.max_entries)
        cache_key = CacheKey(
            collection=collection,
            query=payload.question,
            filters=canonical_filters(payload.filters),
            embedder_id=embedder_name,
            llm_id=payload.llm,
            retrieval=retrieval_fingerprint(
                mode=payload.mode or config.retrieval.mode,
                top_k=config.retrieval.top_k,
                dense_top_k=config.retrieval.dense_top_k,
                sparse_top_k=config.retrieval.sparse_top_k,
                rrf_k=config.retrieval.fusion.k,
                rerank=payload.rerank,
            ),
            corpus_identity=corpus_identity_material,
        )
        cached, cache_status = cache.get(cache_key)
        if cached is not None:
            if payload.debug:
                # A hit skips the graph, so there are no fresh node timings. Report
                # the allowlisted cache status; empty timings are honest, not a lie
                # about work that did not run.
                return cached.model_copy(
                    update={
                        "debug": QueryDebug(
                            timings_ms={},
                            invalid_markers=[],
                            cache=cache_status,
                        )
                    }
                )
            return cached.model_copy(update={"debug": None})

    embedder = resolve_embedder(
        embedder_name,
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
    if on_delta is not None:
        llm = StreamingTee(llm, on_delta)
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
    if cache_key is not None:
        # Store without debug: debug is a per-request widening, not part of the
        # answer identity. A later non-debug hit must not inherit timings from a
        # previous debug caller.
        get_query_cache(max_entries=config.cache.max_entries).put(
            cache_key,
            response.model_copy(update={"debug": None}),
        )
        cache_status = "miss"
    if payload.debug:
        return response.model_copy(
            update={"debug": _project_debug(result, cache_status=cache_status)}
        )
    return response


def get_query_executor() -> QueryExecutor:
    """Return the query adapter; tests override this dependency with a fake."""
    return execute_query


QueryExecutorDep = Annotated[QueryExecutor, Depends(get_query_executor)]
"""Injected query executor; replaced by an offline fake in API unit tests."""


def get_streaming_query_executor() -> StreamingQueryExecutor:
    """Return the streaming-capable adapter; tests override it with a fake.

    A second dependency over the same function, so overriding one in a test does
    not silently change the other — the streamed and unstreamed routes are meant
    to be exercised against different doubles in the same suite.
    """
    return execute_query


StreamingQueryExecutorDep = Annotated[
    StreamingQueryExecutor, Depends(get_streaming_query_executor)
]
"""Injected streaming executor; replaced by an offline fake in API unit tests."""


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
        status.HTTP_404_NOT_FOUND: {
            "description": "Wrong collection name relative to this process identity."
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": (
                "Query pipeline absent, vector store unreachable, or provider failure. "
                "Never a refusal."
            )
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
    except WrongCollectionError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error_type": exc.error_type,
                "collection": exc.collection,
                "message": str(exc),
                "refused": False,
            },
        ) from exc
    except CollectionMismatchError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error_type": "collection_mismatch",
                "message": str(exc),
                "refused": False,
            },
        ) from exc
    except (VectorStoreError, EmbeddingError, LLMError) as exc:
        # Dependency / provider failure is never a refusal. Clients branch on
        # error_type; refused stays false so a soft-fail cannot be mistaken for
        # "corpus has no answer".
        error_type = (
            "store_unavailable"
            if isinstance(exc, VectorStoreError)
            else "provider_error"
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error_type": error_type,
                "message": (
                    "The query could not be completed because a service dependency failed."
                ),
                "refused": False,
            },
        ) from exc
    except QueryPipelineUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error_type": "pipeline_unavailable",
                "message": str(exc),
                "refused": False,
            },
        ) from exc


__all__ = [
    "QueryExecutor",
    "QueryPipelineUnavailableError",
    "StreamingQueryExecutor",
    "deployment_filter_policy",
    "execute_query",
    "filter_policy",
    "get_query_executor",
    "get_streaming_query_executor",
    "router",
]
