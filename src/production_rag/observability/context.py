"""The request id, and the contextvar binding that spreads it across a request.

One identifier per query, bound once at the pipeline boundary and picked up by
every log line that follows — ``configure_logging`` installs
``structlog.contextvars.merge_contextvars`` as its first processor, so a module
five frames down logs the id without knowing it exists. That is the whole point:
correlation must not be a parameter threaded through every signature, or it
stops being threaded the moment someone adds a call site in a hurry.

The API middleware already does exactly this for HTTP requests. This module is
the same seam for callers that are not HTTP — the CLI, the eval script, a
notebook — and it accepts an id from outside so the two never disagree.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from uuid import uuid4

import structlog

REQUEST_ID_KEY = "request_id"
"""Context key. Matches the field the API middleware binds, deliberately."""


def new_request_id() -> str:
    """Return a fresh request id.

    Returns:
        A UUID4 string. Random rather than sequential because ids from separate
        processes share one log stream and must not collide.
    """
    return str(uuid4())


def resolve_request_id(raw: str | None) -> str:
    """Return the caller's id, or mint one.

    Args:
        raw: An id supplied from outside — an HTTP header the middleware already
            validated, a job id, a CLI invocation id. Blank counts as absent.

    Returns:
        ``raw`` stripped when it holds anything, otherwise a new UUID4.

    Note:
        Unlike :func:`production_rag.api.middleware.resolve_request_id`, this
        does not re-validate the shape. Values reaching the library have already
        been through that check at the untrusted edge, and a library that
        silently discards the id its caller is logging under would break the
        correlation it exists to provide.
    """
    if raw is not None and raw.strip():
        return raw.strip()
    return new_request_id()


def current_request_id() -> str | None:
    """Return the request id bound in this context, if any.

    Returns:
        The bound id, or ``None`` outside any bound scope. Useful for attaching
        the id to something that is not a log line — a trace, an error payload.
    """
    value = structlog.contextvars.get_contextvars().get(REQUEST_ID_KEY)
    return value if isinstance(value, str) else None


@contextmanager
def request_context(request_id: str | None = None, /, **fields: Any) -> Iterator[str]:
    """Bind a request id (and any extra fields) for the duration of a block.

    Args:
        request_id: An id from the caller, or ``None`` to mint one.
        **fields: Additional context to bind alongside — anything cheap and
            non-sensitive that helps correlate. Never prompts or passages.

    Yields:
        The id that ended up bound, so the caller can return it to a client.

    Note:
        Bindings are reset on exit, including when the body raises: a failed
        request that leaks its id into the context poisons every later log line
        on that thread or task, and the resulting reports are wrong in the
        specific way that is hardest to notice — they look correlated.

        Only the keys this call bound are reset, so nesting inside the API
        middleware's own binding leaves the outer context intact.
    """
    resolved = resolve_request_id(request_id)
    bound = {REQUEST_ID_KEY: resolved, **fields}
    tokens = structlog.contextvars.bind_contextvars(**bound)
    try:
        yield resolved
    finally:
        structlog.contextvars.reset_contextvars(**tokens)
