"""Langfuse adapter. Imported lazily, required by nothing.

Three gates, all of which must pass before a single span leaves the process:

1. ``observability.tracing.enabled`` is true;
2. the ``langfuse`` package is importable — it lives in the ``obs`` extra, so a
   default install does not have it;
3. all three named environment variables resolve to a value.

Any gate failing yields a :class:`~production_rag.observability.tracer.NullTracer`
and a single warning. A partial configuration does not half-enable tracing: two
of three credentials is a typo, and the useful behaviour is to say so and carry
on serving, not to fail at startup or to retry a broken client per request.

The SDK's surface has changed across major versions, so the calls here are
probed rather than assumed, and every one of them is wrapped. This module is
allowed to be defensive in a way the rest of the codebase is not: it talks to a
third-party client whose failure must cost a diagnostic and nothing else.

Nothing here sends prompt or passage text. Span attributes are ids, counts,
model names and durations — see ADR-0006.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import structlog

from production_rag.observability.tracer import NULL_SPAN, NullTracer, Span

if TYPE_CHECKING:
    from production_rag.config_loader import TracingConfig
    from production_rag.observability.tracer import Tracer

_log = structlog.get_logger(__name__)

_START_METHODS = ("start_span", "span")
"""Client methods that open a span, newest SDK spelling first."""

_END_METHODS = ("end", "close")
"""Span methods that close one."""


class LangfuseSpan:
    """Wraps one SDK span object behind the :class:`Span` Protocol."""

    __slots__ = ("_span",)

    def __init__(self, span: Any) -> None:
        """Store the SDK span."""
        self._span = span

    def set_attribute(self, key: str, value: object) -> None:
        """Attach one attribute, ignoring an SDK that will not take it."""
        update = getattr(self._span, "update", None)
        if update is None:
            return
        try:
            update(metadata={key: value})
        except Exception as exc:  # noqa: BLE001 - telemetry never fails a request
            _log.warning("tracing_attribute_failed", key=key, error=str(exc))

    def record_error(self, exc: BaseException) -> None:
        """Mark the span failed with the exception type and message.

        The message only. A provider error body can contain the request that
        produced it, which is the prompt, which is corpus text.
        """
        update = getattr(self._span, "update", None)
        if update is None:
            return
        try:
            update(level="ERROR", status_message=f"{type(exc).__name__}: {exc}")
        except Exception as inner:  # noqa: BLE001 - telemetry never fails a request
            _log.warning("tracing_error_failed", error=str(inner))

    def end(self) -> None:
        """Close the SDK span, whichever spelling it uses."""
        for name in _END_METHODS:
            method = getattr(self._span, name, None)
            if method is None:
                continue
            try:
                method()
            except Exception as exc:  # noqa: BLE001 - telemetry never fails a request
                _log.warning("tracing_span_end_failed", error=str(exc))
            return


class LangfuseTracer:
    """A tracer backed by a Langfuse client.

    Note:
        Parent/child nesting is left to the SDK's own context propagation. This
        adapter deliberately does not model a trace tree: the pipeline opens
        spans in a strict sequence per request, and reimplementing parenting
        here would be a second, worse copy of something the SDK already does.
    """

    __slots__ = ("_client",)

    def __init__(self, client: Any) -> None:
        """Store an already-constructed SDK client."""
        self._client = client

    @property
    def enabled(self) -> bool:
        """Always ``True``; an unusable client never becomes a tracer."""
        return True

    @contextmanager
    def span(self, name: str, **attributes: object) -> Iterator[Span]:
        """Open a Langfuse span, degrading to the null span if the SDK refuses."""
        started = self._start(name, attributes)
        if started is None:
            yield NULL_SPAN
            return

        try:
            yield started
        except BaseException as exc:
            started.record_error(exc)
            raise
        finally:
            started.end()

    def _start(self, name: str, attributes: dict[str, object]) -> LangfuseSpan | None:
        """Open a span through whichever SDK method exists, or return ``None``."""
        for method_name in _START_METHODS:
            method = getattr(self._client, method_name, None)
            if method is None:
                continue
            try:
                return LangfuseSpan(method(name=name, metadata=dict(attributes)))
            except Exception as exc:  # noqa: BLE001 - telemetry never fails a request
                _log.warning("tracing_span_failed", span=name, error=str(exc))
                return None
        _log.warning("tracing_span_unsupported", span=name)
        return None

    def flush(self) -> None:
        """Push buffered spans. Langfuse batches, so a short-lived process needs this."""
        flush = getattr(self._client, "flush", None)
        if flush is None:
            return
        try:
            flush()
        except Exception as exc:  # noqa: BLE001 - telemetry never fails a request
            _log.warning("tracing_flush_failed", error=str(exc))


def resolve_credentials(config: TracingConfig) -> dict[str, str] | None:
    """Read the named environment variables.

    Args:
        config: The tracing block, which stores variable *names*, never values.

    Returns:
        A ``public_key``/``secret_key``/``host`` mapping when all three resolve,
        otherwise ``None``. The names of any missing variables are logged; their
        values are not, and neither are the values of the ones that did resolve.
    """
    wanted = {
        "public_key": config.public_key_env,
        "secret_key": config.secret_key_env,
        "host": config.host_env,
    }
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for field, env_name in wanted.items():
        value = os.environ.get(env_name, "").strip()
        if value:
            resolved[field] = value
        else:
            missing.append(env_name)

    if missing:
        _log.warning("tracing_credentials_incomplete", missing=missing, using="null")
        return None
    return resolved


def build_langfuse_tracer(config: TracingConfig) -> Tracer:
    """Build a Langfuse tracer, or the null tracer if anything is not in place.

    Args:
        config: The ``observability.tracing`` block, already known to be enabled.

    Returns:
        A :class:`LangfuseTracer` when the SDK imports and the credentials
        resolve, otherwise a :class:`~production_rag.observability.tracer.NullTracer`.
    """
    credentials = resolve_credentials(config)
    if credentials is None:
        return NullTracer()

    try:
        from langfuse import Langfuse
    except ImportError:
        _log.warning("tracing_sdk_missing", provider="langfuse", extra="obs", using="null")
        return NullTracer()

    try:
        client = Langfuse(**credentials)
    except Exception as exc:  # noqa: BLE001 - telemetry never fails startup
        _log.warning("tracing_client_failed", provider="langfuse", error=str(exc), using="null")
        return NullTracer()

    _log.info("tracing_enabled", provider="langfuse", host=credentials["host"])
    return LangfuseTracer(client)
