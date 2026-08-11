"""The tracing seam: a Protocol, a default that does nothing, and adapters.

Same shape as the embedder, the reranker and the LLM. The pipeline depends on a
Protocol; the concrete backend is chosen once, at the edge, from config. That
keeps :mod:`production_rag.graph` and :mod:`production_rag.query_pipeline` free
of any vendor import and lets every test run against a double.

Two properties are load-bearing, and both are about a trace backend being a
*diagnostic*, not a dependency:

* **The default is :class:`NullTracer`.** Not "a tracer that is configured off"
  — an object whose spans allocate one small context manager and record nothing.
  A caller that never mentions tracing pays no import and no network.
* **Everything fails open.** A backend that is down, slow, misconfigured or
  simply not installed loses a span. It never turns a served answer into a 500,
  because a request failing due to its own telemetry is the worst possible
  trade: the observability costs you the thing it was meant to observe.

Span attributes must stay non-sensitive — ids, counts, model names, durations.
Prompts and retrieved passages are not attributes; see ADR-0006.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import structlog

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from production_rag.config_loader import TracingConfig

_log = structlog.get_logger(__name__)

TRACER_KINDS = ("null", "langfuse", "otel")
"""Backends ``build_tracer`` can construct by name."""

INSTRUMENTATION_NAME = "production_rag"
"""Instrumentation scope reported to OpenTelemetry."""


@runtime_checkable
class Span(Protocol):
    """One timed unit of work inside a request."""

    def set_attribute(self, key: str, value: object) -> None:
        """Attach one non-sensitive key/value to this span."""
        ...

    def record_error(self, exc: BaseException) -> None:
        """Mark this span as failed, recording the exception *type and message*.

        Implementations must not attach a payload that could carry corpus text
        or credentials; a provider error body is summarised, never echoed.
        """
        ...


@runtime_checkable
class Tracer(Protocol):
    """Creates spans. The only tracing surface the pipeline knows about."""

    @property
    def enabled(self) -> bool:
        """Whether spans reach a backend. ``False`` for the null tracer."""
        ...

    def span(self, name: str, **attributes: object) -> AbstractContextManager[Span]:
        """Open a span, closing it when the context exits.

        Args:
            name: Span name. For graph nodes these are the node names verbatim,
                so a trace and a ``timings_ms`` dict use one vocabulary.
            **attributes: Non-sensitive attributes set at open time.

        Returns:
            A context manager yielding the span. It must not raise for backend
            problems, and it must not suppress exceptions from its body.
        """
        ...

    def flush(self) -> None:
        """Push anything buffered. A no-op when there is no backend."""
        ...


class NullSpan:
    """A span that discards everything. The default, and the fallback."""

    __slots__ = ()

    def set_attribute(self, key: str, value: object) -> None:
        """Discard an attribute."""

    def record_error(self, exc: BaseException) -> None:
        """Discard an error."""


NULL_SPAN = NullSpan()
"""Shared instance. Stateless, so one is enough for the whole process."""


class NullTracer:
    """The default tracer: correct, free, and offline.

    Chosen deliberately over ``tracer: Tracer | None = None`` plus null checks.
    An optional dependency that every call site has to remember to guard is one
    that eventually is not guarded, and the failure surfaces in production where
    tracing is configured rather than in the tests where it is not.
    """

    __slots__ = ()

    @property
    def enabled(self) -> bool:
        """Always ``False``."""
        return False

    @contextmanager
    def span(self, name: str, **attributes: object) -> Iterator[Span]:
        """Yield the shared null span."""
        yield NULL_SPAN

    def flush(self) -> None:
        """Nothing is buffered."""


class RecordedSpan:
    """An in-memory span. Test double, and the record a fake tracer keeps."""

    __slots__ = ("attributes", "error", "name")

    def __init__(self, name: str, attributes: Mapping[str, object]) -> None:
        """Store the span name and its opening attributes."""
        self.name = name
        self.attributes: dict[str, object] = dict(attributes)
        self.error: str | None = None

    def set_attribute(self, key: str, value: object) -> None:
        """Record an attribute."""
        self.attributes[key] = value

    def record_error(self, exc: BaseException) -> None:
        """Record the exception type and message, matching the real adapters."""
        self.error = f"{type(exc).__name__}: {exc}"

    def __repr__(self) -> str:
        """Show the name, so a failed assertion names the span."""
        return f"RecordedSpan(name={self.name!r})"


class RecordingTracer:
    """Collects spans in a list instead of shipping them.

    Lives beside the production tracers rather than in ``tests/`` because it is
    also the honest way to demonstrate what gets traced without configuring a
    vendor — and because a double kept next to the Protocol it implements is a
    double that stays in step with it.
    """

    __slots__ = ("spans",)

    def __init__(self) -> None:
        """Start with no recorded spans."""
        self.spans: list[RecordedSpan] = []

    @property
    def enabled(self) -> bool:
        """Always ``True``: spans do reach a sink, just an in-memory one."""
        return True

    @contextmanager
    def span(self, name: str, **attributes: object) -> Iterator[Span]:
        """Record a span, marking it failed if the body raises."""
        recorded = RecordedSpan(name, attributes)
        self.spans.append(recorded)
        try:
            yield recorded
        except BaseException as exc:
            recorded.record_error(exc)
            raise

    def flush(self) -> None:
        """Nothing to flush."""

    def names(self) -> list[str]:
        """Return recorded span names in order."""
        return [span.name for span in self.spans]


@contextmanager
def guarded_span(tracer: Tracer, name: str, **attributes: object) -> Iterator[Span]:
    """Open a span that cannot break the caller.

    Args:
        tracer: Any tracer.
        name: Span name.
        **attributes: Non-sensitive opening attributes.

    Yields:
        The backend's span, or :data:`NULL_SPAN` when the backend raised while
        opening one.

    Note:
        This is where fail-open is actually implemented, in one place, so no
        node has to remember it. The context manager is driven by hand rather
        than with a ``with`` block for one reason: a span wraps the work, so it
        sees both its own failures and the body's, and only the former may be
        swallowed. Entering and exiting explicitly keeps those two cases apart
        instead of guessing from a traceback. Tracing failures are logged once
        at warning level, so a gap in the traces is still visible somewhere.
    """
    try:
        manager = tracer.span(name, **attributes)
        span = manager.__enter__()
    except Exception as exc:  # noqa: BLE001 - telemetry must never fail a request
        _log.warning("tracing_span_failed", span=name, phase="open", error=str(exc))
        yield NULL_SPAN
        return

    try:
        yield span
    except BaseException as body_error:
        try:
            suppressed = manager.__exit__(type(body_error), body_error, body_error.__traceback__)
        except Exception as exit_error:  # noqa: BLE001 - see above
            if exit_error is body_error:
                raise
            _log.warning("tracing_span_failed", span=name, phase="close", error=str(exit_error))
            raise body_error from None
        if not suppressed:
            raise
    else:
        try:
            manager.__exit__(None, None, None)
        except Exception as exc:  # noqa: BLE001 - see above
            _log.warning("tracing_span_failed", span=name, phase="close", error=str(exc))


class OTelSpan:
    """Wraps an OpenTelemetry span behind the :class:`Span` Protocol."""

    __slots__ = ("_span",)

    def __init__(self, span: Any) -> None:
        """Store the SDK span."""
        self._span = span

    def set_attribute(self, key: str, value: object) -> None:
        """Set one attribute, coercing anything OTel will not accept to a string."""
        if not isinstance(value, str | bool | int | float):
            value = str(value)
        self._span.set_attribute(key, value)

    def record_error(self, exc: BaseException) -> None:
        """Record the exception on the span, without a stack payload."""
        self._span.record_exception(exc)


class OTelTracer:
    """A tracer backed by whatever OpenTelemetry provider the host configured.

    Deliberately a thin stub. This project does not configure exporters, a
    resource or a sampler — a host that already runs OTel has all of that, and a
    library that installs its own provider fights the application it lives in.
    With no provider configured, ``get_tracer`` hands back a no-op and this
    costs one wrapper object per span.
    """

    __slots__ = ("_tracer",)

    def __init__(self, tracer: Any) -> None:
        """Store an OpenTelemetry tracer."""
        self._tracer = tracer

    @property
    def enabled(self) -> bool:
        """Always ``True``; whether spans are exported is the host's decision."""
        return True

    @contextmanager
    def span(self, name: str, **attributes: object) -> Iterator[Span]:
        """Open an OTel span as the current one, recording a failing body."""
        with self._tracer.start_as_current_span(name) as raw:
            wrapped = OTelSpan(raw)
            for key, value in attributes.items():
                wrapped.set_attribute(key, value)
            try:
                yield wrapped
            except BaseException as exc:
                wrapped.record_error(exc)
                raise

    def flush(self) -> None:
        """No-op: flushing a provider the host owns is the host's call."""


def build_otel_tracer() -> Tracer:
    """Return an OpenTelemetry tracer, or the null tracer if OTel is absent.

    Returns:
        An :class:`OTelTracer` when ``opentelemetry-api`` is importable — it is
        in the ``obs`` extra — otherwise :class:`NullTracer`.
    """
    try:
        from opentelemetry import trace
    except ImportError:
        _log.warning("tracing_sdk_missing", provider="otel", extra="obs", using="null")
        return NullTracer()

    return OTelTracer(trace.get_tracer(INSTRUMENTATION_NAME))


def build_tracer(config: TracingConfig | None = None) -> Tracer:
    """Build the tracer a config asks for, degrading to :class:`NullTracer`.

    Args:
        config: The ``observability.tracing`` block, or ``None`` for the default.

    Returns:
        A configured tracer when tracing is enabled, the provider is known, its
        SDK is importable and its credentials resolve. :class:`NullTracer` in
        every other case, including every failure — an unusable trace backend is
        a missing diagnostic, not a startup error.
    """
    if config is None or not config.enabled:
        return NullTracer()

    if config.provider == "langfuse":
        from production_rag.observability.langfuse_client import build_langfuse_tracer

        return build_langfuse_tracer(config)

    if config.provider in {"otel", "opentelemetry"}:
        return build_otel_tracer()

    _log.warning("tracing_provider_unknown", provider=config.provider, using="null")
    return NullTracer()


def span_attributes(**values: Any) -> dict[str, object]:
    """Drop ``None`` values from span attributes.

    Args:
        **values: Candidate attributes.

    Returns:
        Only the entries that carry a value. Backends differ on what they do
        with a null attribute and none of them do anything useful.
    """
    return {key: value for key, value in values.items() if value is not None}
