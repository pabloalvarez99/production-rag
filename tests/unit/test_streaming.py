"""Unit tests for the streaming seam: chunks, the tee, and the SSE wire format.

Offline throughout, like the rest of the query tests. The thing under test here
is not "does text arrive" but the two properties a streamed answer has to have
to be worth trusting:

* concatenating the deltas reproduces the answer the pipeline finalised, byte
  for byte, so a client that renders the stream and a client that renders the
  result never disagree; and
* a provider failure part way through a stream is still a failure, never a
  short answer and never a refusal.
"""

from __future__ import annotations

from collections.abc import Generator, Sequence

import pytest

from production_rag.api.sse import SSEEvent
from production_rag.config_loader import RetrievalConfig
from production_rag.generation.llm import (
    FakeLLM,
    LLMError,
    LLMResponse,
    OpenAILLM,
    fake_chunks,
)
from production_rag.generation.prompts import (
    ABSTAIN_TOKEN,
    CONTEXT_HEADER,
    QUESTION_HEADER,
    ChatMessage,
)
from production_rag.generation.streaming import (
    StreamingLLM,
    StreamingTee,
    supports_streaming,
)
from production_rag.ingest.models import Chunk, Document
from production_rag.query_pipeline import run_query
from production_rag.retrieval.embeddings import FakeEmbeddingProvider
from production_rag.retrieval.hybrid import Retriever
from production_rag.retrieval.sparse import Bm25Encoder
from production_rag.retrieval.store import InMemoryVectorStore

RUN_ID = "run-streaming-test"

CORPUS = {
    "sample/01-qdrant.md": "Qdrant stores dense and sparse vectors in one collection.",
    "sample/02-fusion.md": "Reciprocal rank fusion merges two ranked lists by position.",
}


BLOCK_BODY = "Qdrant stores dense and sparse vectors."


def _messages(question: str, *, body: str = BLOCK_BODY) -> list[ChatMessage]:
    """A user message in the real rendered-prompt shape.

    Hand-built rather than taken from ``build_prompt``, because these tests are
    about the model double and the tee; using the real assembler here would make
    a prompt-format change fail them for a reason that has nothing to do with
    streaming. The format is still the real one — :func:`parse_context_blocks`
    is what reads it back.
    """
    block = f"[1] Hybrid search\n{body}"
    user = f"{CONTEXT_HEADER}\n\n{block}\n\n{QUESTION_HEADER} {question}"
    return [ChatMessage(role="user", content=user)]


def _drain(chunks: Generator[str, None, LLMResponse]) -> tuple[list[str], LLMResponse]:
    """Consume a stream, keeping both the pieces and the response it returned.

    ``list(generator)`` throws the return value away — it is carried on the
    ``StopIteration`` the loop swallows — and that value is the whole point of
    the contract, so the tests drain by hand.
    """
    seen: list[str] = []
    while True:
        try:
            seen.append(next(chunks))
        except StopIteration as stop:
            return seen, stop.value


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


class WholeAnswerLLM:
    """A provider with no ``stream`` method at all."""

    @property
    def model(self) -> str:
        return "whole-answer"

    def complete(self, messages: Sequence[ChatMessage]) -> LLMResponse:
        return LLMResponse(text="Grounded [1].", model=self.model, finish_reason="stop")


class MidStreamOutageLLM:
    """A provider that fails after it has already yielded text."""

    @property
    def model(self) -> str:
        return "mid-stream-outage"

    def complete(self, messages: Sequence[ChatMessage]) -> LLMResponse:
        raise LLMError("generation request failed: APIError: upstream")

    def stream(self, messages: Sequence[ChatMessage]) -> Generator[str, None, LLMResponse]:
        yield "Reciprocal "
        yield "rank "
        raise LLMError("generation request failed: APIError: upstream")


class ForgetfulLLM:
    """A ``stream`` that yields text and forgets to return the response."""

    @property
    def model(self) -> str:
        return "forgetful"

    def complete(self, messages: Sequence[ChatMessage]) -> LLMResponse:
        return LLMResponse(text="Grounded [1].", model=self.model, finish_reason="stop")

    def stream(self, messages: Sequence[ChatMessage]) -> Generator[str, None, None]:
        yield "Grounded [1]."


class TestFakeChunks:
    def test_joining_the_chunks_reproduces_the_text(self) -> None:
        text = "Reciprocal rank fusion merges two ranked lists [1]."
        assert "".join(fake_chunks(text)) == text

    def test_each_chunk_is_one_word_and_keeps_its_space(self) -> None:
        assert fake_chunks("alpha beta gamma") == ("alpha ", "beta ", "gamma")

    def test_a_single_word_is_a_single_chunk_with_no_trailing_space(self) -> None:
        assert fake_chunks(ABSTAIN_TOKEN) == (ABSTAIN_TOKEN,)

    def test_empty_text_produces_no_chunks(self) -> None:
        assert fake_chunks("") == ()


class TestFakeLLMStream:
    def test_the_stream_and_the_completion_agree(self) -> None:
        messages = _messages("Which store keeps sparse vectors?")
        completed = FakeLLM().complete(messages)
        chunks, returned = _drain(FakeLLM().stream(messages))

        assert "".join(chunks) == completed.text
        assert returned == completed

    def test_the_same_prompt_streams_the_same_bytes_every_time(self) -> None:
        messages = _messages("Which store keeps sparse vectors?")
        first, _ = _drain(FakeLLM().stream(messages))
        second, _ = _drain(FakeLLM().stream(messages))

        assert first == second
        assert first == ["Qdrant ", "stores ", "dense ", "and ", "sparse ", "vectors ", "[1]."]

    def test_an_abstention_streams_the_sentinel_rather_than_nothing(self) -> None:
        # The sentinel is provisional text like any other: it is the guardrails,
        # not the model double, that decide what a user is shown.
        chunks, _ = _drain(FakeLLM().stream(_messages("Explain quantum chromodynamics")))
        assert chunks == [ABSTAIN_TOKEN]

    def test_usage_is_recorded_once_per_streamed_completion(self) -> None:
        recorded: list[tuple[int, int]] = []

        def recorder(*, prompt_tokens: int, completion_tokens: int) -> None:
            recorded.append((prompt_tokens, completion_tokens))

        llm = FakeLLM(usage_recorder=recorder)
        list(llm.stream([ChatMessage(role="user", content="[1] Qdrant.\n\nQuestion: qdrant")]))
        assert len(recorded) == 1


class TestSupportsStreaming:
    def test_the_offline_double_streams(self) -> None:
        assert supports_streaming(FakeLLM()) is True

    def test_the_hosted_provider_streams(self) -> None:
        assert supports_streaming(OpenAILLM(api_key="", client=object())) is True

    def test_a_provider_without_the_method_does_not(self) -> None:
        assert supports_streaming(WholeAnswerLLM()) is False


class TestStreamingTee:
    def test_chunks_reach_the_sink_in_order(self) -> None:
        seen: list[str] = []
        messages = _messages("Which store keeps sparse vectors?")

        response = StreamingTee(FakeLLM(), seen.append).complete(messages)

        assert "".join(seen) == response.text
        assert len(seen) > 1

    def test_the_tee_reports_the_wrapped_model_not_itself(self) -> None:
        tee = StreamingTee(FakeLLM(), lambda _: None)
        assert tee.model == FakeLLM().model
        assert isinstance(tee.inner, FakeLLM)

    def test_a_non_streaming_provider_still_publishes_one_chunk(self) -> None:
        seen: list[str] = []
        response = StreamingTee(WholeAnswerLLM(), seen.append).complete([])
        assert seen == [response.text]

    def test_a_mid_stream_failure_raises_and_does_not_return_a_partial_answer(self) -> None:
        seen: list[str] = []
        with pytest.raises(LLMError):
            StreamingTee(MidStreamOutageLLM(), seen.append).complete([])
        # The chunks published before the failure are exactly what a client had
        # already seen. The pipeline gets nothing, which is the point.
        assert seen == ["Reciprocal ", "rank "]

    def test_a_stream_that_returns_nothing_is_a_contract_error(self) -> None:
        with pytest.raises(TypeError, match="must return an LLMResponse"):
            StreamingTee(ForgetfulLLM(), lambda _: None).complete([])

    def test_the_protocol_matches_the_double_structurally(self) -> None:
        assert isinstance(FakeLLM(), StreamingLLM)
        assert not isinstance(WholeAnswerLLM(), StreamingLLM)


class TestStreamedPipeline:
    """The tee inside the real query path, with no transport involved."""

    def test_the_deltas_reconstruct_the_finalised_answer(self) -> None:
        seen: list[str] = []
        result = run_query(
            "Which fusion merges ranked lists?",
            retriever=_retriever(),
            llm=StreamingTee(FakeLLM(), seen.append),
        )

        assert result.refused is False
        assert result.citations
        assert "".join(seen) == result.answer

    def test_the_model_recorded_is_the_wrapped_one(self) -> None:
        result = run_query(
            "Which fusion merges ranked lists?",
            retriever=_retriever(),
            llm=StreamingTee(FakeLLM(), lambda _: None),
        )
        assert result.model == FakeLLM().model

    def test_a_refusal_with_no_evidence_streams_nothing_at_all(self) -> None:
        seen: list[str] = []
        result = run_query(
            "Which fusion merges ranked lists?",
            retriever=_empty_retriever(),
            llm=StreamingTee(FakeLLM(), seen.append),
        )

        # The pre-generation guardrail refuses before the model is called, so
        # there is nothing provisional to retract: no evidence, no tokens spent.
        assert result.refused is True
        assert seen == []

    def test_an_abstention_streams_text_that_the_refusal_then_replaces(self) -> None:
        seen: list[str] = []
        result = run_query(
            "Explain quantum chromodynamics",
            retriever=_retriever(),
            llm=StreamingTee(FakeLLM(), seen.append),
        )

        assert result.refused is True
        assert result.refusal_reason == "model_abstained"
        assert seen == [ABSTAIN_TOKEN]
        # What the user is shown is the configured refusal, never the sentinel.
        assert ABSTAIN_TOKEN not in result.answer

    def test_a_provider_outage_propagates_rather_than_refusing(self) -> None:
        with pytest.raises(LLMError):
            run_query(
                "Which fusion merges ranked lists?",
                retriever=_retriever(),
                llm=StreamingTee(MidStreamOutageLLM(), lambda _: None),
            )


class TestSSEEncoding:
    def test_an_event_is_framed_exactly(self) -> None:
        assert SSEEvent("delta", {"text": "Hybrid "}).encode() == (
            b'event: delta\ndata: {"text":"Hybrid "}\n\n'
        )

    def test_a_newline_in_the_payload_stays_on_one_data_line(self) -> None:
        encoded = SSEEvent("delta", {"text": "one\ntwo"}).encode()
        assert encoded == b'event: delta\ndata: {"text":"one\\ntwo"}\n\n'
        assert encoded.count(b"\n\n") == 1

    def test_non_ascii_text_is_not_escaped_into_ascii(self) -> None:
        encoded = SSEEvent("delta", {"text": "café"}).encode()
        assert "café".encode() in encoded
