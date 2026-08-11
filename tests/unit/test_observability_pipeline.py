"""Unit tests for observability on the query path itself.

Offline throughout, like the rest of the query tests: in-memory store, fake
embedder, fake LLM. The point of these tests is that instrumentation is present
and inert — a pipeline nobody configured for tracing behaves exactly as it did
before M5, and one that is configured emits spans named after the graph nodes.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager

import pytest
import structlog

from production_rag.config_loader import RetrievalConfig
from production_rag.generation.llm import FakeLLM, LLMError, LLMResponse
from production_rag.generation.prompts import ChatMessage
from production_rag.graph.nodes import QueryDeps
from production_rag.graph.state import NODE_NAMES
from production_rag.ingest.models import Chunk, Document
from production_rag.observability.context import current_request_id
from production_rag.observability.tracer import NullTracer, RecordingTracer, Span
from production_rag.query_pipeline import QUERY_SPAN, build_query_pipeline, run_query
from production_rag.retrieval.embeddings import FakeEmbeddingProvider
from production_rag.retrieval.hybrid import Retriever
from production_rag.retrieval.sparse import Bm25Encoder
from production_rag.retrieval.store import InMemoryVectorStore

RUN_ID = "run-observability-test"

CORPUS = {
    "sample/01-qdrant.md": "Qdrant stores dense and sparse vectors in one collection.",
    "sample/02-fusion.md": "Reciprocal rank fusion merges two ranked lists by position.",
}


@pytest.fixture(autouse=True)
def _clean_context() -> None:
    """Never let one test's binding be another test's context."""
    structlog.contextvars.clear_contextvars()


def _retriever(*, corpus: dict[str, str] | None = None) -> Retriever:
    documents = CORPUS if corpus is None else corpus
    embedder = FakeEmbeddingProvider()
    store = InMemoryVectorStore()
    store.ensure_collection(vector_size=embedder.dimensions, with_sparse=True)
    chunks = [
        Chunk.build(
            document=Document(source_path=path, text=text, title=path, source="sample"),
            chunk_index=index,
            text=text,
            embed_text=text,
        )
        for index, (path, text) in enumerate(documents.items())
    ]
    texts = [chunk.embed_text for chunk in chunks]
    sparse_vectors = None
    if texts:
        encoder = Bm25Encoder()
        encoder.fit(texts)
        sparse_vectors = encoder.encode_documents(texts)
    store.upsert_chunks(
        chunks,
        embedder.embed_documents(texts),
        ingest_run_id=RUN_ID,
        embedded_model=embedder.model,
        sparse_vectors=sparse_vectors,
    )
    return Retriever(store=store, embedder=embedder, config=RetrievalConfig())


def _empty_retriever() -> Retriever:
    embedder = FakeEmbeddingProvider()
    store = InMemoryVectorStore()
    store.ensure_collection(vector_size=embedder.dimensions, with_sparse=True)
    return Retriever(store=store, embedder=embedder)


class ContextReadingLLM:
    """Records the request id that was bound while the model was called."""

    def __init__(self) -> None:
        self.seen: list[str | None] = []

    @property
    def model(self) -> str:
        return "context-reading"

    def complete(self, messages: Sequence[ChatMessage]) -> LLMResponse:
        self.seen.append(current_request_id())
        return LLMResponse(text="Grounded [1].", model=self.model, finish_reason="stop")


class OutageLLM:
    """A provider that is down."""

    @property
    def model(self) -> str:
        return "outage"

    def complete(self, messages: Sequence[ChatMessage]) -> LLMResponse:
        raise LLMError("generation request failed: TimeoutError: upstream")


class TestDefaults:
    def test_the_default_pipeline_traces_nothing(self) -> None:
        pipeline = build_query_pipeline(retriever=_retriever(), llm=FakeLLM())
        assert isinstance(pipeline.deps.tracer, NullTracer)

    def test_bare_deps_carry_the_null_tracer(self) -> None:
        deps = QueryDeps(
            retriever=_retriever(),
            llm=FakeLLM(),
            generation=build_query_pipeline(retriever=_retriever(), llm=FakeLLM()).deps.generation,
        )
        assert deps.tracer.enabled is False

    def test_an_untraced_query_still_answers(self) -> None:
        result = run_query("what is fusion?", retriever=_retriever(), llm=FakeLLM())
        assert result.refused is False
        assert result.citations


class TestRequestId:
    def test_the_callers_id_is_returned_on_the_result(self) -> None:
        result = run_query(
            "what is fusion?",
            retriever=_retriever(),
            llm=FakeLLM(),
            request_id="req-1",
        )
        assert result.request_id == "req-1"

    def test_an_id_is_minted_when_the_caller_has_none(self) -> None:
        result = run_query("what is fusion?", retriever=_retriever(), llm=FakeLLM())
        assert result.request_id

    def test_two_queries_get_different_ids(self) -> None:
        first = run_query("what is fusion?", retriever=_retriever(), llm=FakeLLM())
        second = run_query("what is fusion?", retriever=_retriever(), llm=FakeLLM())
        assert first.request_id != second.request_id

    def test_the_id_is_bound_while_the_model_runs(self) -> None:
        # The whole point of contextvars over a threaded parameter: a module
        # several frames down logs the id without being handed it.
        llm = ContextReadingLLM()
        run_query("what is fusion?", retriever=_retriever(), llm=llm, request_id="req-1")
        assert llm.seen == ["req-1"]

    def test_the_context_is_clear_after_the_query(self) -> None:
        run_query("what is fusion?", retriever=_retriever(), llm=FakeLLM(), request_id="req-1")
        assert current_request_id() is None

    def test_the_context_is_clear_after_a_failed_query(self) -> None:
        with pytest.raises(LLMError):
            run_query("what is fusion?", retriever=_retriever(), llm=OutageLLM())
        assert current_request_id() is None

    def test_the_id_is_on_the_serialised_result(self) -> None:
        result = run_query(
            "what is fusion?", retriever=_retriever(), llm=FakeLLM(), request_id="req-1"
        )
        assert result.to_dict()["request_id"] == "req-1"


class TestSpans:
    def test_every_node_gets_a_span_named_after_it(self) -> None:
        tracer = RecordingTracer()
        run_query("what is fusion?", retriever=_retriever(), llm=FakeLLM(), tracer=tracer)
        assert tracer.names() == [QUERY_SPAN, *NODE_NAMES]

    def test_span_names_are_exactly_the_timing_keys(self) -> None:
        # One vocabulary for both signals: a span found in a trace can be looked
        # up in ``timings_ms`` without a translation table.
        tracer = RecordingTracer()
        result = run_query("what is fusion?", retriever=_retriever(), llm=FakeLLM(), tracer=tracer)
        traced = set(tracer.names()) - {QUERY_SPAN}
        assert traced == set(result.timings_ms or {})

    def test_spans_carry_the_request_id(self) -> None:
        tracer = RecordingTracer()
        run_query(
            "what is fusion?",
            retriever=_retriever(),
            llm=FakeLLM(),
            tracer=tracer,
            request_id="req-1",
        )
        assert all(span.attributes["request_id"] == "req-1" for span in tracer.spans)

    def test_a_refusal_stops_tracing_at_the_guard(self) -> None:
        tracer = RecordingTracer()
        result = run_query(
            "what is fusion?", retriever=_empty_retriever(), llm=FakeLLM(), tracer=tracer
        )
        assert result.refused is True
        assert tracer.names() == [QUERY_SPAN, "retrieve", "rerank", "guard"]

    def test_a_provider_outage_is_recorded_on_the_span(self) -> None:
        tracer = RecordingTracer()
        with pytest.raises(LLMError):
            run_query("what is fusion?", retriever=_retriever(), llm=OutageLLM(), tracer=tracer)
        failed = {span.name: span.error for span in tracer.spans if span.error is not None}
        # The failing node and the request span both, so a trace shows where it
        # broke and that the request as a whole did.
        assert set(failed) == {"generate", QUERY_SPAN}
        assert failed["generate"].startswith("LLMError:")

    def test_no_span_carries_the_question_or_a_passage(self) -> None:
        # Attributes are ids and counts. Prompt and passage text is not a signal;
        # an aggregator holding it is a copy of the corpus without its controls.
        tracer = RecordingTracer()
        run_query(
            "what is reciprocal rank fusion?",
            retriever=_retriever(),
            llm=FakeLLM(),
            tracer=tracer,
        )
        rendered = " ".join(
            f"{key}={value}" for span in tracer.spans for key, value in span.attributes.items()
        )
        assert "reciprocal" not in rendered.lower()
        assert all(text not in rendered for text in CORPUS.values())


class TestTracingNeverFailsAQuery:
    def test_a_broken_tracer_still_answers(self) -> None:
        class BrokenTracer:
            @property
            def enabled(self) -> bool:
                return True

            def span(
                self, name: str, **attributes: object
            ) -> AbstractContextManager[Span, bool | None]:
                raise RuntimeError("trace backend unreachable")

            def flush(self) -> None:
                raise RuntimeError("trace backend unreachable")

        result = run_query(
            "what is fusion?", retriever=_retriever(), llm=FakeLLM(), tracer=BrokenTracer()
        )
        assert result.refused is False
        assert result.citations
