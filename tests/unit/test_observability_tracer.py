"""Unit tests for the tracing seam. Offline: no SDK is installed or needed."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

from production_rag.config_loader import ObservabilityConfig, TracingConfig, YamlConfig
from production_rag.observability import langfuse_client
from production_rag.observability.tracer import (
    NULL_SPAN,
    NullTracer,
    OTelTracer,
    RecordingTracer,
    Span,
    Tracer,
    build_tracer,
    guarded_span,
    span_attributes,
)


class BrokenTracer:
    """Fails on every operation. A backend having a bad day."""

    @property
    def enabled(self) -> bool:
        return True

    def span(self, name: str, **attributes: object) -> Any:
        raise RuntimeError("trace backend unreachable")

    def flush(self) -> None:
        raise RuntimeError("trace backend unreachable")


class BrokenOnCloseTracer:
    """Opens a span fine, then fails while closing it."""

    @property
    def enabled(self) -> bool:
        return True

    @contextmanager
    def span(self, name: str, **attributes: object) -> Iterator[Span]:
        yield NULL_SPAN
        raise RuntimeError("flush failed on close")

    def flush(self) -> None:
        return None


class TestNullTracer:
    def test_it_is_not_enabled(self) -> None:
        assert NullTracer().enabled is False

    def test_a_span_yields_something_usable(self) -> None:
        with NullTracer().span("retrieve", hits=3) as span:
            span.set_attribute("k", "v")
            span.record_error(RuntimeError("boom"))

    def test_it_satisfies_the_protocol(self) -> None:
        assert isinstance(NullTracer(), Tracer)

    def test_a_failing_body_still_propagates(self) -> None:
        with pytest.raises(ValueError, match="boom"), NullTracer().span("retrieve"):
            raise ValueError("boom")


class TestRecordingTracer:
    def test_spans_are_recorded_in_order(self) -> None:
        tracer = RecordingTracer()
        for name in ("retrieve", "generate"):
            with tracer.span(name):
                pass
        assert tracer.names() == ["retrieve", "generate"]

    def test_opening_attributes_are_kept(self) -> None:
        tracer = RecordingTracer()
        with tracer.span("retrieve", request_id="req-1"):
            pass
        assert tracer.spans[0].attributes == {"request_id": "req-1"}

    def test_attributes_can_be_added_inside_the_span(self) -> None:
        tracer = RecordingTracer()
        with tracer.span("retrieve") as span:
            span.set_attribute("hits", 3)
        assert tracer.spans[0].attributes["hits"] == 3

    def test_a_failing_body_is_recorded_and_re_raised(self) -> None:
        tracer = RecordingTracer()
        with pytest.raises(ValueError, match="provider down"), tracer.span("generate"):
            raise ValueError("provider down")
        assert tracer.spans[0].error == "ValueError: provider down"


class TestGuardedSpanFailsOpen:
    def test_a_backend_that_cannot_open_a_span_does_not_fail_the_caller(self) -> None:
        with guarded_span(BrokenTracer(), "retrieve") as span:
            span.set_attribute("hits", 3)

    def test_a_backend_that_fails_on_close_does_not_fail_the_caller(self) -> None:
        with guarded_span(BrokenOnCloseTracer(), "retrieve"):
            pass

    def test_the_body_still_runs_when_the_backend_is_broken(self) -> None:
        ran = False
        with guarded_span(BrokenTracer(), "retrieve"):
            ran = True
        assert ran is True

    def test_a_failing_body_is_never_swallowed(self) -> None:
        # The opposite mistake to swallowing telemetry errors, and the worse one:
        # telemetry that hides a real failure is worse than no telemetry.
        with pytest.raises(ValueError, match="boom"), guarded_span(RecordingTracer(), "generate"):
            raise ValueError("boom")

    def test_a_failing_body_propagates_through_a_broken_backend_too(self) -> None:
        with pytest.raises(ValueError, match="boom"), guarded_span(BrokenOnCloseTracer(), "gen"):
            raise ValueError("boom")

    def test_the_span_reaches_the_backend(self) -> None:
        tracer = RecordingTracer()
        with guarded_span(tracer, "retrieve", request_id="req-1"):
            pass
        assert tracer.spans[0].attributes == {"request_id": "req-1"}


class TestSpanAttributes:
    def test_none_values_are_dropped(self) -> None:
        assert span_attributes(request_id=None, hits=0) == {"hits": 0}


class TestBuildTracer:
    def test_no_config_is_the_null_tracer(self) -> None:
        assert isinstance(build_tracer(None), NullTracer)

    def test_disabled_is_the_null_tracer(self) -> None:
        assert isinstance(build_tracer(TracingConfig()), NullTracer)

    def test_tracing_is_off_in_the_default_profile(self) -> None:
        assert YamlConfig().observability.tracing.enabled is False

    def test_an_unknown_provider_degrades_instead_of_raising(self) -> None:
        config = TracingConfig(enabled=True, provider="jaeger-by-carrier-pigeon")
        assert isinstance(build_tracer(config), NullTracer)

    def test_langfuse_without_credentials_degrades(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"):
            monkeypatch.delenv(name, raising=False)
        assert isinstance(build_tracer(TracingConfig(enabled=True)), NullTracer)

    def test_langfuse_without_the_sdk_degrades(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The default install has no langfuse. Enabling tracing on it must cost a
        # warning, not a startup crash.
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public-value")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "secret-value")
        monkeypatch.setenv("LANGFUSE_HOST", "https://cloud.example.test")
        monkeypatch.setitem(sys.modules, "langfuse", None)
        assert isinstance(build_tracer(TracingConfig(enabled=True)), NullTracer)

    def test_otel_without_the_sdk_degrades(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "opentelemetry", None)
        config = TracingConfig(enabled=True, provider="otel")
        assert isinstance(build_tracer(config), NullTracer)


class TestLangfuseCredentials:
    def test_all_three_variables_must_resolve(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public-value")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "secret-value")
        monkeypatch.delenv("LANGFUSE_HOST", raising=False)
        assert langfuse_client.resolve_credentials(TracingConfig(enabled=True)) is None

    def test_a_blank_value_counts_as_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "   ")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "secret-value")
        monkeypatch.setenv("LANGFUSE_HOST", "https://cloud.example.test")
        assert langfuse_client.resolve_credentials(TracingConfig(enabled=True)) is None

    def test_the_names_are_configurable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TRACE_PUBLIC", "public-value")
        monkeypatch.setenv("TRACE_SECRET", "secret-value")
        monkeypatch.setenv("TRACE_HOST", "https://cloud.example.test")
        config = TracingConfig(
            enabled=True,
            public_key_env="TRACE_PUBLIC",
            secret_key_env="TRACE_SECRET",  # noqa: S106 - a variable name, not a secret
            host_env="TRACE_HOST",
        )
        assert langfuse_client.resolve_credentials(config) == {
            "public_key": "public-value",
            "secret_key": "secret-value",
            "host": "https://cloud.example.test",
        }

    def test_the_config_stores_names_not_values(self) -> None:
        # Regla 3, as a test: the YAML names an environment variable, and the
        # credential itself never enters a tracked file.
        config = TracingConfig()
        assert config.public_key_env == "LANGFUSE_PUBLIC_KEY"
        assert config.secret_key_env == "LANGFUSE_SECRET_KEY"  # noqa: S105 - a name


class RecordingSDKSpan:
    """Stands in for a Langfuse span object."""

    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []
        self.ended = False

    def update(self, **fields: Any) -> None:
        self.updates.append(fields)

    def end(self) -> None:
        self.ended = True


class RecordingSDKClient:
    """Stands in for a Langfuse client."""

    def __init__(self) -> None:
        self.spans: list[RecordingSDKSpan] = []
        self.flushed = 0

    def start_span(self, name: str, metadata: dict[str, Any]) -> RecordingSDKSpan:
        span = RecordingSDKSpan()
        self.spans.append(span)
        return span

    def flush(self) -> None:
        self.flushed += 1


class ExplodingSDKClient:
    """A client whose every call fails."""

    def start_span(self, name: str, metadata: dict[str, Any]) -> Any:
        raise RuntimeError("unauthorised")

    def flush(self) -> None:
        raise RuntimeError("unauthorised")


class TestLangfuseTracer:
    def test_a_span_is_opened_and_closed(self) -> None:
        client = RecordingSDKClient()
        with langfuse_client.LangfuseTracer(client).span("retrieve", request_id="req-1"):
            pass
        assert client.spans[0].ended is True

    def test_attributes_are_sent_as_metadata(self) -> None:
        client = RecordingSDKClient()
        with langfuse_client.LangfuseTracer(client).span("retrieve") as span:
            span.set_attribute("hits", 3)
        assert client.spans[0].updates == [{"metadata": {"hits": 3}}]

    def test_a_failing_body_is_marked_and_re_raised(self) -> None:
        client = RecordingSDKClient()
        tracer = langfuse_client.LangfuseTracer(client)
        with pytest.raises(ValueError, match="provider down"), tracer.span("generate"):
            raise ValueError("provider down")
        assert client.spans[0].updates[0]["level"] == "ERROR"
        assert client.spans[0].updates[0]["status_message"] == "ValueError: provider down"
        assert client.spans[0].ended is True

    def test_flush_reaches_the_client(self) -> None:
        client = RecordingSDKClient()
        langfuse_client.LangfuseTracer(client).flush()
        assert client.flushed == 1

    def test_an_sdk_that_raises_does_not_fail_the_caller(self) -> None:
        tracer = langfuse_client.LangfuseTracer(ExplodingSDKClient())
        with tracer.span("retrieve") as span:
            span.set_attribute("hits", 3)
        tracer.flush()

    def test_an_sdk_without_a_span_method_does_not_fail_the_caller(self) -> None:
        with langfuse_client.LangfuseTracer(object()).span("retrieve") as span:
            span.set_attribute("hits", 3)


class RecordingOTelSpan:
    def __init__(self) -> None:
        self.attributes: dict[str, Any] = {}
        self.errors: list[BaseException] = []

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def record_exception(self, exc: BaseException) -> None:
        self.errors.append(exc)


class RecordingOTelTracer:
    def __init__(self) -> None:
        self.spans: list[RecordingOTelSpan] = []

    @contextmanager
    def start_as_current_span(self, name: str) -> Iterator[RecordingOTelSpan]:
        span = RecordingOTelSpan()
        self.spans.append(span)
        yield span


class TestOTelTracer:
    def test_attributes_reach_the_span(self) -> None:
        sdk = RecordingOTelTracer()
        with OTelTracer(sdk).span("retrieve", request_id="req-1") as span:
            span.set_attribute("hits", 3)
        assert sdk.spans[0].attributes == {"request_id": "req-1", "hits": 3}

    def test_an_unsupported_attribute_type_is_stringified(self) -> None:
        sdk = RecordingOTelTracer()
        with OTelTracer(sdk).span("retrieve") as span:
            span.set_attribute("markers", (1, 2))
        assert sdk.spans[0].attributes["markers"] == "(1, 2)"

    def test_a_failing_body_is_recorded_and_re_raised(self) -> None:
        sdk = RecordingOTelTracer()
        with pytest.raises(ValueError, match="boom"), OTelTracer(sdk).span("generate"):
            raise ValueError("boom")
        assert isinstance(sdk.spans[0].errors[0], ValueError)


class TestObservabilityConfig:
    def test_prompt_and_passage_logging_are_off_by_default(self) -> None:
        # The single most expensive default in this file: a prompt carries
        # retrieved corpus text verbatim.
        logging = ObservabilityConfig().logging
        assert logging.log_prompts is False
        assert logging.log_retrieved_text is False

    def test_the_default_profile_parses_into_the_typed_block(self) -> None:
        config = YamlConfig()
        assert config.observability.tracing.sample_rate == 1.0
        assert config.observability.metrics.path == "/metrics"
