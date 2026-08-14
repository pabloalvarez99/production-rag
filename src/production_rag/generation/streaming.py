"""Incremental delivery for the generation stage, without a second pipeline.

    llm.stream(messages) -> yields provisional text, returns the final LLMResponse

The whole point of this module is a distinction the transport layer must not
blur: **a delta is provisional model output, not an answer.** Whether an answer
is served at all is decided after the model has finished, by
:mod:`production_rag.generation.guardrails`, against citations that only exist
once the full text has been mapped onto the prompt blocks. A stream that
presented deltas as the answer would be showing text the pipeline has not yet
agreed to serve — and would occasionally have to take it back, which is the
failure mode this repository refuses everywhere else.

So the seam here is deliberately small:

* :class:`StreamingLLM` — a provider that can hand over its completion in
  pieces. It is a :class:`~typing.Protocol` next to
  :class:`~production_rag.generation.llm.LLM`, not a replacement for it: every
  provider still has to answer :meth:`~production_rag.generation.llm.LLM.complete`,
  because the graph, the CLI and the eval harness call that and must keep
  working unchanged.
* :class:`StreamingTee` — an :class:`~production_rag.generation.llm.LLM` that
  wraps another one and publishes each piece to a sink as it arrives. This is
  what lets the SSE route reuse ``run_query`` verbatim rather than growing a
  parallel copy of the query path that would drift from it.

``stream`` is a generator whose *return* value is the authoritative
:class:`~production_rag.generation.llm.LLMResponse`. The alternative — yielding
text and looking the model name, finish reason and token usage up afterwards —
loses exactly the fields that must travel with the answer (see
:class:`~production_rag.generation.llm.LLMResponse`), and an abstention would
arrive indistinguishable from prose.

A provider that cannot stream is not an error. :class:`StreamingTee` falls back
to one call and one chunk, and the caller can see that it was one chunk. Nothing
pretends otherwise.
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Sequence
from typing import Protocol, runtime_checkable

from production_rag.generation.llm import LLM, LLMResponse
from production_rag.generation.prompts import ChatMessage

DeltaSink = Callable[[str], None]
"""Where a chunk goes the moment it exists.

Deliberately ``str -> None``: the sink cannot answer back, cannot cancel and
cannot fail the generation it is observing. A transport that wants to stop
reading closes its connection, which is the caller's business, not the model's.
"""


@runtime_checkable
class StreamingLLM(Protocol):
    """An :class:`~production_rag.generation.llm.LLM` that can also stream.

    Implementations satisfy :class:`~production_rag.generation.llm.LLM` as well;
    ``stream`` is an addition, never a substitute. ``isinstance`` against this
    Protocol is a structural check for the ``stream`` attribute only — which is
    all the tee needs, and all a runtime-checkable Protocol can honestly offer.
    """

    def stream(
        self, messages: Sequence[ChatMessage]
    ) -> Generator[str, None, LLMResponse]:
        """Yield the completion in pieces, then return the whole response.

        Concatenating every yielded piece must reproduce
        :attr:`~production_rag.generation.llm.LLMResponse.text` exactly. A
        consumer that has to guess whether to re-insert whitespace between
        chunks will guess wrong somewhere, and the answer that reaches citation
        mapping would then differ from the one the user watched appear.

        Raises:
            LLMError: The provider failed, at the start or part way through. A
                stream that has already yielded text and then fails is still a
                failure — not a short answer, and never a refusal.
        """
        ...


def supports_streaming(llm: LLM) -> bool:
    """Whether *llm* can hand its completion over in pieces."""
    return isinstance(llm, StreamingLLM)


class StreamingTee:
    """An LLM that publishes each chunk to *sink* while completing normally.

    Wrapping rather than branching is what keeps one query path. The graph calls
    :meth:`complete` exactly as it always has and gets exactly the response it
    always got; the sink is a side channel the pipeline knows nothing about.

    The tee is single-use per call and holds no state between calls, so the same
    instance can serve one request without a lock. It is not shared across
    requests — the sink belongs to one client's connection.
    """

    def __init__(self, inner: LLM, sink: DeltaSink) -> None:
        """Publish *inner*'s output to *sink* as it is produced.

        Args:
            inner: The real model. Not required to stream; see
                :meth:`complete` for what happens when it cannot.
            sink: Called once per chunk, in order, before :meth:`complete`
                returns.
        """
        self._inner = inner
        self._sink = sink

    @property
    def inner(self) -> LLM:
        """The wrapped model. Useful to a caller that needs its identity."""
        return self._inner

    @property
    def model(self) -> str:
        """The wrapped model's identifier, unchanged.

        The tee is transport, not a model, and must never appear in a result:
        an eval row attributing an answer to ``StreamingTee`` would be wrong
        about the only field that makes the row comparable.
        """
        return self._inner.model

    def complete(self, messages: Sequence[ChatMessage]) -> LLMResponse:
        """Complete *messages*, publishing chunks to the sink on the way.

        Falls back to a single non-streamed call when the wrapped provider does
        not implement :class:`StreamingLLM`. That still publishes one chunk,
        so a client's rendering path is the same in both cases and only the
        number of deltas differs — which the caller can count and report,
        rather than being told a non-streaming provider streamed.

        Raises:
            LLMError: Propagated from the provider, mid-stream or otherwise.
                Chunks already published are not retracted here: the transport
                decides what to tell a client that has seen provisional text,
                and the guardrails never see a partial answer at all because
                this call raised instead of returning one.
        """
        if not isinstance(self._inner, StreamingLLM):
            response = self._inner.complete(messages)
            self._sink(response.text)
            return response
        chunks = self._inner.stream(messages)
        while True:
            try:
                chunk = next(chunks)
            except StopIteration as stop:
                return _returned_response(stop)
            self._sink(chunk)


def _returned_response(stop: StopIteration) -> LLMResponse:
    """Read the response a finished ``stream`` generator returned.

    A generator that falls off its end returns ``None``, which would reach the
    graph as a missing model and a missing finish reason several stages later.
    Failing here names the implementation that broke the contract.
    """
    response = stop.value
    if not isinstance(response, LLMResponse):
        raise TypeError(
            "a StreamingLLM.stream generator must return an LLMResponse when it "
            f"finishes; got {type(response).__name__}"
        )
    return response


__all__ = [
    "DeltaSink",
    "StreamingLLM",
    "StreamingTee",
    "supports_streaming",
]
