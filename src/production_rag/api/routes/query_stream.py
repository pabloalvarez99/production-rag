"""Streamed grounded query: the same answer, delivered while it is being made.

``POST /v1/query/stream`` is **additive**. ``POST /v1/query`` is the contract and
does not change; this route exists because a user staring at a blank panel for
several seconds cannot tell a working system from a hung one, and that is a
product problem rather than a retrieval one.

What is streamed is deliberately not "the answer":

    meta    request id, before any work
    delta   provisional model output, zero or more, never authoritative
    result  the body POST /v1/query would have returned — terminal
    error   the run failed — terminal, and never a refusal

The pipeline decides whether to serve an answer *after* generation, in the
guardrails, against citations that are resolved from the full text. So a delta
is text the system has not yet agreed to serve. A refusal that arrives after
forty deltas is the system working: it drafted, it could not ground what it
drafted, and it said so. A client renders deltas as a draft and replaces them
when ``result`` arrives; the UI in this repository does exactly that, and the
event names are chosen so a client that renders ``delta`` as an answer is
visibly reading the wrong event rather than subtly reading the right one wrong.

Failures that can be known **before** any byte is written keep their status
code: a filter outside the allowlist is the same typed 422 the JSON route
returns, because nothing has been streamed and there is still a status line to
spend. Once the stream is open the status is 200 and cannot be taken back, so
everything later becomes an ``error`` event carrying an ``error_type``. That
asymmetry is the honest one — a 200 that later says "error" is worse than a 422,
and pretending a mid-stream provider outage could have been a 503 is worse than
both.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from production_rag.api.deps import SettingsDep
from production_rag.api.middleware import get_request_id
from production_rag.api.routes.query import (
    HTTP_422_UNPROCESSABLE,
    QueryPipelineUnavailableError,
    StreamingQueryExecutor,
    StreamingQueryExecutorDep,
    deployment_filter_policy,
)
from production_rag.api.schemas import QueryRequest, QueryResponse
from production_rag.api.sse import (
    EVENT_DELTA,
    EVENT_ERROR,
    EVENT_META,
    EVENT_RESULT,
    SSE_HEADERS,
    SSE_MEDIA_TYPE,
    SSEEvent,
)
from production_rag.config import Settings
from production_rag.generation.llm import LLMError
from production_rag.retrieval.filters import FilterError

_log = structlog.get_logger(__name__)

router = APIRouter(tags=["query"])
"""Versioned router; mounted under ``Settings.api_prefix`` beside ``query``."""

ERROR_PROVIDER = "provider_error"
"""A model or embedding provider failed.

Its own type, and emphatically not a refusal. "The documents do not say" and
"the provider is down" are different facts about the world; a client that
cannot tell them apart retries the one that will never succeed and gives up on
the one that would have.
"""

ERROR_PIPELINE_UNAVAILABLE = "pipeline_unavailable"
"""The query pipeline is absent from this checkout — the 503 of the JSON route,
demoted to an event because the stream was already open."""

ERROR_INTERNAL = "internal_error"
"""Anything unclassified. The message is generic; the request id is the handle."""


def _error_type(exc: BaseException) -> str:
    """Classify a failure into the closed set a client may branch on."""
    if isinstance(exc, FilterError):
        return exc.error_type
    if isinstance(exc, LLMError):
        return ERROR_PROVIDER
    if isinstance(exc, QueryPipelineUnavailableError):
        return ERROR_PIPELINE_UNAVAILABLE
    return ERROR_INTERNAL


def _error_message(exc: BaseException) -> str:
    """A safe message for a client.

    Filter rejections quote themselves: they describe the request, and the
    caller has to know which field was refused to fix it. Everything else gets
    one sentence, because provider and store exceptions may quote a URL, a
    payload or a header, and an error body is the least controlled place in the
    system to discover that.
    """
    if isinstance(exc, FilterError | QueryPipelineUnavailableError):
        return str(exc)
    return "The query could not be completed because a service dependency failed."


def iter_query_stream(
    executor: StreamingQueryExecutor,
    payload: QueryRequest,
    *,
    settings: Settings,
    request_id: str,
) -> Iterator[bytes]:
    """Yield the SSE byte stream for one query.

    The pipeline is synchronous and publishes chunks through a callback, so the
    two ends are joined by a thread and a queue: the worker runs the ordinary
    query path and drops each chunk into the queue, this generator drains the
    queue and frames what it finds. Starlette drives a sync generator from its
    threadpool, so the blocking ``get`` here parks a pool thread rather than the
    event loop.

    A thread is the cost of *not* forking the query path. The alternative — an
    async re-implementation that can yield from inside generation — means two
    pipelines with two sets of guardrail behaviour, and the second one is the
    one nobody re-reads when a rule changes.
    """
    chunks: queue.SimpleQueue[str | None] = queue.SimpleQueue()
    outcome: list[tuple[str, Any]] = []

    def run() -> None:
        try:
            response = executor(
                payload,
                settings=settings,
                request_id=request_id,
                on_delta=chunks.put,
            )
        except Exception as exc:  # noqa: BLE001 - reported to the client as an event
            outcome.append(("error", exc))
        else:
            outcome.append(("result", response))
        finally:
            # Always, including after an exception: the sentinel is what ends
            # the drain loop, and a stream that hangs open on failure is worse
            # than the failure.
            chunks.put(None)

    worker = threading.Thread(target=run, name=f"query-stream-{request_id}", daemon=True)
    # Before the work starts, so a client that dies mid-request still holds the
    # id that identifies this run in the server's log.
    yield SSEEvent(EVENT_META, {"request_id": request_id}).encode()
    worker.start()

    deltas = 0
    while True:
        chunk = chunks.get()
        if chunk is None:
            break
        deltas += 1
        yield SSEEvent(EVENT_DELTA, {"text": chunk}).encode()
    worker.join()

    kind, value = outcome[0]
    if kind == "error":
        _log.warning(
            "query_stream_failed",
            request_id=request_id,
            error_type=_error_type(value),
            error=type(value).__name__,
        )
        yield SSEEvent(
            EVENT_ERROR,
            {
                "error_type": _error_type(value),
                "message": _error_message(value),
                "request_id": request_id,
                # Explicit rather than absent. A client holding provisional text
                # has to be told that this is not a refusal it can display; the
                # field says so in the payload instead of in prose somewhere.
                "refused": False,
            },
        ).encode()
        return

    response: QueryResponse = value
    yield SSEEvent(
        EVENT_RESULT,
        {
            **response.model_dump(mode="json"),
            "request_id": request_id,
            # How many pieces the answer actually arrived in. One means the
            # provider does not stream and the whole thing landed at the end —
            # which is a fine outcome and a bad thing to hide.
            "deltas": deltas,
        },
    ).encode()


@router.post(
    "/query/stream",
    status_code=status.HTTP_200_OK,
    response_class=StreamingResponse,
    responses={
        status.HTTP_200_OK: {
            "content": {SSE_MEDIA_TYPE: {}},
            "description": (
                "An SSE stream: one `meta` event, zero or more provisional `delta` "
                "events, then exactly one terminal `result` or `error` event. A "
                "refusal is a `result`, never an `error`."
            ),
        },
        HTTP_422_UNPROCESSABLE: {
            "description": (
                "The body failed validation, or `filters` names a field outside "
                "`retrieval.filters.allowed_fields`. Checked before the stream "
                "opens, so it is still a status code rather than an event."
            )
        },
    },
    summary="Answer a question from indexed evidence, streamed",
    operation_id="query_stream",
)
def query_stream(
    payload: QueryRequest,
    request: Request,
    settings: SettingsDep,
    executor: StreamingQueryExecutorDep,
) -> StreamingResponse:
    """Stream one grounded answer, or the refusal that replaces it."""
    request_id = get_request_id(request)
    if payload.filters:
        # The same policy object the JSON route and the retriever use, applied
        # here for one reason the others do not have: after the first byte there
        # is no status code left, and "filters were rejected" deserves one.
        try:
            deployment_filter_policy(settings).build(payload.filters)
        except FilterError as exc:
            raise HTTPException(
                status_code=HTTP_422_UNPROCESSABLE,
                detail={
                    "error_type": exc.error_type,
                    "field": exc.field,
                    "message": str(exc),
                },
            ) from exc

    return StreamingResponse(
        iter_query_stream(
            executor,
            payload,
            settings=settings,
            request_id=request_id,
        ),
        media_type=SSE_MEDIA_TYPE,
        headers=SSE_HEADERS,
    )


__all__ = [
    "ERROR_INTERNAL",
    "ERROR_PIPELINE_UNAVAILABLE",
    "ERROR_PROVIDER",
    "iter_query_stream",
    "router",
]
