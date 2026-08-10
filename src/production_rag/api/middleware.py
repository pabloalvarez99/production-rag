"""Request-scoped context: correlation id, structured access log, timing.

A RAG request fans out into embedding, retrieval, rerank and generation calls,
each of which can be slow or fail on its own. Without a correlation id threaded
through every log line, a production incident becomes an exercise in guessing
which log belongs to which question. Binding it once here — into structlog's
contextvars — means later milestones get it for free: any ``log.info(...)``
anywhere inside the request already carries ``request_id``.

Bodies, headers and secrets are never logged.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from time import perf_counter
from uuid import uuid4

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from structlog.contextvars import bind_contextvars, clear_contextvars

REQUEST_ID_HEADER = "X-Request-ID"
"""Inbound and outbound header carrying the correlation id."""

RESPONSE_TIME_HEADER = "X-Response-Time-ms"
"""Server-side handling time, useful when the client sees latency the server does not."""

_SAFE_REQUEST_ID = re.compile(r"\A[A-Za-z0-9._:-]{1,128}\Z")
"""Shape an inbound correlation id must have to be trusted.

The value is client-supplied and gets written back into a response header and
into structured logs. Constraining it to a conservative character set keeps
control characters out of both sinks, and the length bound stops a caller from
parking kilobytes in every log line of a request.
"""

_log = structlog.get_logger("production_rag.access")

CallNext = Callable[[Request], Awaitable[Response]]


def resolve_request_id(raw: str | None) -> str:
    """Return a trustworthy correlation id, generating one when needed.

    An absent *or* malformed inbound id yields a fresh UUID4 rather than an
    error: a bad correlation header is not worth failing a request over, and a
    silently missing id is worse than a substituted one.
    """
    if raw is not None and _SAFE_REQUEST_ID.match(raw):
        return raw
    return str(uuid4())


def get_request_id(request: Request) -> str:
    """Read the correlation id bound to *request* by the middleware."""
    request_id: str = getattr(request.state, "request_id", "")
    return request_id


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach a correlation id to the request, the response and the logs."""

    async def dispatch(self, request: Request, call_next: CallNext) -> Response:
        """Bind request context, time the handler, emit one access log line."""
        request_id = resolve_request_id(request.headers.get(REQUEST_ID_HEADER))
        request.state.request_id = request_id

        clear_contextvars()
        bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )
        started = perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # Log and re-raise: swallowing here would rob the exception
            # handlers (and the test client) of the real traceback.
            _log.exception(
                "request_failed",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                duration_ms=_elapsed_ms(started),
            )
            raise
        finally:
            clear_contextvars()

        duration_ms = _elapsed_ms(started)
        response.headers[REQUEST_ID_HEADER] = request_id
        response.headers[RESPONSE_TIME_HEADER] = f"{duration_ms:.2f}"
        _log.info(
            "request_completed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
        )
        return response


def _elapsed_ms(started: float) -> float:
    """Milliseconds since *started*, rounded to microsecond resolution."""
    return round((perf_counter() - started) * 1000, 3)
