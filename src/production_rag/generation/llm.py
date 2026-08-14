"""The LLM seam: one Protocol, one offline double, one real provider.

Same shape as :mod:`production_rag.retrieval.embeddings` and
:mod:`production_rag.retrieval.rerank`, for the same reason — every expensive or
networked dependency in this project sits behind a Protocol with a deterministic
offline implementation, so the unit suite is green with the wifi off and nobody
needs a credential to run the pipeline end to end.

Both also implement
:class:`~production_rag.generation.streaming.StreamingLLM`, which hands the same
completion over in pieces. That is an addition to this seam, never a fork of it:
:meth:`LLM.complete` stays the method the graph, the CLI and the evals call, and
streaming changes what a *transport* can show a user while waiting, not what the
pipeline decides to serve.

* :class:`FakeLLM` — extractive, deterministic, offline. It reads the numbered
  context blocks back out of the prompt it was handed and answers with the
  leading sentence of each, carrying the matching ``[n]`` marker; with no blocks
  it abstains. It also refuses when no substantive question term occurs in any
  block. It is a **plumbing double, not a quality claim**: lexical overlap is
  not understanding. What it does guarantee is
  that citation mapping, the guardrails and the graph are exercised by a real
  prompt and a real, marker-bearing answer.
* :class:`OpenAILLM` — the real one. Model and temperature come from the YAML
  block; the credential comes from the environment.

Neither implementation decides whether an answer is acceptable. Abstention is
signalled with :data:`~production_rag.generation.prompts.ABSTAIN_TOKEN` and
judged in :mod:`production_rag.generation.guardrails`, so "the model refused"
and "the pipeline refused" stay one decision made in one place.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import structlog

from production_rag.config_loader import GenerationConfig
from production_rag.generation.prompts import ABSTAIN_TOKEN, ChatMessage, parse_context_blocks

_WORD = re.compile(r"[a-z0-9]+")
_QUESTION_HEADER = "Question:"
_QUESTION_STOPWORDS = frozenset(
    {
        "about",
        "does",
        "explain",
        "from",
        "have",
        "into",
        "that",
        "the",
        "this",
        "what",
        "when",
        "where",
        "which",
        "with",
        "why",
    }
)


def _substantive_terms(text: str) -> set[str]:
    return {
        token
        for token in _WORD.findall(text.lower())
        if len(token) >= 4 and token not in _QUESTION_STOPWORDS
    }

_log = structlog.get_logger(__name__)

LLM_FAKE = "fake"
LLM_OPENAI = "openai"
LLM_KINDS = (LLM_FAKE, LLM_OPENAI)
"""What ``build_llm`` accepts. ``fake`` is the offline path; ``openai`` needs a key."""

FAKE_MODEL = "fake-extractive-v1"
"""Recorded in the result, so an eval row can never be mistaken for a real run."""

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def fake_chunks(text: str) -> tuple[str, ...]:
    """Split *text* the way :meth:`FakeLLM.stream` does. One word per chunk.

    The rule is stated here, in one function, because it is a **fixture**: the
    streaming tests assert the exact byte sequence a client receives, and a test
    that asserts bytes needs the bytes to be decided somewhere a reader can
    check rather than emerge from a provider's tokenizer.

    Each chunk carries its own trailing space, so ``"".join(fake_chunks(t)) ==
    t`` for every ``t``. A consumer therefore concatenates and is done; it never
    has to decide whether to re-insert whitespace, which is the bug that makes a
    streamed answer differ from the completed one by a space and then differ
    from it by a citation.
    """
    if not text:
        return ()
    words = text.split(" ")
    return (*(f"{word} " for word in words[:-1]), words[-1])


class LLMError(RuntimeError):
    """The model could not be built, or could not produce a completion."""


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """A completion, plus what produced it.

    ``model`` is carried rather than looked up later because the answer and the
    model that wrote it have to travel together: an eval score attributed to the
    wrong model is worse than no score.
    """

    text: str
    model: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@runtime_checkable
class LLM(Protocol):
    """Turns chat messages into a completion."""

    @property
    def model(self) -> str:
        """Identifier recorded in the result, e.g. ``"gpt-4o-mini"``."""

    def complete(self, messages: Sequence[ChatMessage]) -> LLMResponse:
        """Return one completion for *messages*.

        Raises:
            LLMError: The provider failed. Implementations must not return a
                partial or empty string to signal failure — an empty answer is a
                legitimate outcome the guardrails handle, and conflating the two
                turns an outage into a silent refusal.
        """


class FakeLLM:
    """Deterministic extractive model for offline runs and the unit suite.

    **Not a quality claim.** The answer is the opening sentence of each of the
    top blocks with its marker appended — no reasoning, no synthesis, no reading
    of the question. It exists so the whole query path (prompt, citations,
    guardrails, graph edges, HTTP response shape) runs with no key and no
    network, and so a demo of this repository works on a fresh clone.

    It abstains exactly when it should: no context blocks in the prompt means
    :data:`~production_rag.generation.prompts.ABSTAIN_TOKEN`, which the
    guardrails turn into a refusal. That makes the refusal path testable without
    persuading a real model to decline.
    """

    def __init__(
        self,
        *,
        max_sentences: int = 2,
        model: str = FAKE_MODEL,
        usage_recorder: Callable[..., None] | None = None,
    ) -> None:
        """Configure the double.

        Args:
            max_sentences: How many blocks to quote. Two by default: enough that
                multi-citation answers are exercised, few enough that the output
                stays readable in a CLI.
            model: Identifier reported in the result.
            usage_recorder: Optional response-accounting callback used by evals.
        """
        self._max_sentences = max(1, max_sentences)
        self._model = model
        self._usage_recorder = usage_recorder

    @property
    def model(self) -> str:
        """The identifier reported in the result."""
        return self._model

    def complete(self, messages: Sequence[ChatMessage]) -> LLMResponse:
        """Answer from the numbered blocks in the last user message."""
        response = self._answer(messages)
        self._record_usage(response)
        return response

    def stream(self, messages: Sequence[ChatMessage]) -> Generator[str, None, LLMResponse]:
        """Yield the same answer :meth:`complete` gives, one word at a time.

        The answer is computed whole and then split (:func:`fake_chunks`). That
        is the honest description of this double: it is a fixture for the
        *transport*, and it makes no claim that a model composes an answer
        incrementally. What it does guarantee is that a client sees the same
        bytes in the same order on every run, on every machine — which is what
        makes an assertion about a stream worth writing.

        An abstention yields the sentinel like any other text rather than
        yielding nothing. Suppressing it here would put a rendering decision
        inside the model double and hide the general case: every delta is
        provisional until the guardrails have run, and the abstain path is
        simply the clearest example of provisional text that must be replaced.
        """
        response = self._answer(messages)
        yield from fake_chunks(response.text)
        self._record_usage(response)
        return response

    def _answer(self, messages: Sequence[ChatMessage]) -> LLMResponse:
        """The completion, before anything is recorded or chunked."""
        user = next(
            (message.content for message in reversed(list(messages)) if message.role == "user"),
            "",
        )
        blocks = parse_context_blocks(user)
        if not blocks:
            return LLMResponse(text=ABSTAIN_TOKEN, model=self._model, finish_reason="abstain")

        question = user.rsplit(_QUESTION_HEADER, 1)[-1]
        question_terms = _substantive_terms(question)
        context_terms = _substantive_terms(" ".join(body for _, body in blocks))
        if question_terms and question_terms.isdisjoint(context_terms):
            return LLMResponse(text=ABSTAIN_TOKEN, model=self._model, finish_reason="abstain")

        sentences = [
            _cited_sentence(_first_sentence(body), marker)
            for marker, body in blocks[: self._max_sentences]
            if _first_sentence(body)
        ]
        if not sentences:
            # Blocks that are whitespace once the header is stripped carry no
            # quotable evidence. Abstaining beats emitting a bare marker.
            return LLMResponse(text=ABSTAIN_TOKEN, model=self._model, finish_reason="abstain")
        return LLMResponse(text=" ".join(sentences), model=self._model, finish_reason="stop")

    def _record_usage(self, response: LLMResponse) -> None:
        if self._usage_recorder is not None:
            self._usage_recorder(
                prompt_tokens=response.prompt_tokens or 0,
                completion_tokens=response.completion_tokens or 0,
            )


class OpenAILLM:
    """Chat completions from the OpenAI API.

    The client is built lazily, so importing this module never needs the
    ``openai`` package to be importable or a credential to be present — the CLI
    and the tests import it unconditionally.

    Retries and timeouts are the SDK's job: it already knows which status codes
    are retryable and applies jittered backoff. Re-implementing that here would
    mean two retry budgets multiplying into a request that takes minutes to fail.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_output_tokens: int = 800,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
        client: Any | None = None,
        usage_recorder: Callable[..., None] | None = None,
    ) -> None:
        """Configure the provider. No connection is opened until :meth:`complete`.

        Raises:
            LLMError: No API key. Refusing here, before any work, makes it a
                usage error rather than a mid-request failure.
        """
        if not api_key and client is None:
            raise LLMError(
                "the OpenAI generator needs a credential: set OPENAI_API_KEY, "
                "or run with --llm fake"
            )
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._client = client
        self._usage_recorder = usage_recorder

    @property
    def model(self) -> str:
        """The configured chat model."""
        return self._model

    def _get_client(self) -> Any:
        """Build the OpenAI client on first use."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - dependency is declared
                raise LLMError(
                    "the openai package is not installed; run with --llm fake "
                    'or reinstall with pip install -e "."'
                ) from exc
            self._client = OpenAI(
                api_key=self._api_key,
                max_retries=self._max_retries,
                timeout=self._timeout_seconds,
            )
        return self._client

    def complete(self, messages: Sequence[ChatMessage]) -> LLMResponse:
        """Send one chat completion request.

        Raises:
            LLMError: The request failed, or the response carried no content.
        """
        client = self._get_client()
        try:
            response = client.chat.completions.create(
                model=self._model,
                messages=[message.as_dict() for message in messages],
                temperature=self._temperature,
                max_tokens=self._max_output_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - provider errors are opaque by design
            # The message is included; the credential never is, and the SDK does
            # not put it in exception text.
            raise LLMError(f"generation request failed: {type(exc).__name__}: {exc}") from exc

        choices = getattr(response, "choices", None) or []
        if not choices:
            raise LLMError(f"model {self._model!r} returned no choices")
        message = choices[0].message
        content = getattr(message, "content", None)
        if content is None:
            # A refusal from the safety layer, or a tool call this path never
            # asked for. Either way there is no answer, and pretending it was an
            # empty answer would hide the reason.
            raise LLMError(f"model {self._model!r} returned a message with no content")

        usage = getattr(response, "usage", None)
        result = LLMResponse(
            text=str(content).strip(),
            model=str(getattr(response, "model", self._model)),
            finish_reason=getattr(choices[0], "finish_reason", None),
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
        )
        if self._usage_recorder is not None:
            self._usage_recorder(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
            )
        return result

    def stream(self, messages: Sequence[ChatMessage]) -> Generator[str, None, LLMResponse]:
        """Send one chat completion request and yield content deltas.

        Token usage is requested explicitly (``include_usage``) because a
        streamed response does not carry it otherwise, and an eval row with a
        blank cost column is a row that gets filled in later by guessing.

        The finish reason is taken from whichever chunk carries one — the API
        sends it on the penultimate chunk, not the last, and a stream that ends
        without one is reported as ``None`` rather than assumed to be ``stop``:
        a truncated answer and a completed one must not look alike.

        Raises:
            LLMError: The request failed, at the start or part way through. A
                partial answer is never returned as if it were whole.
        """
        client = self._get_client()
        try:
            stream = client.chat.completions.create(
                model=self._model,
                messages=[message.as_dict() for message in messages],
                temperature=self._temperature,
                max_tokens=self._max_output_tokens,
                stream=True,
                stream_options={"include_usage": True},
            )
            pieces: list[str] = []
            state = _StreamState(model=self._model)
            for chunk in stream:
                delta = state.absorb(chunk)
                if delta:
                    pieces.append(delta)
                    yield delta
        except Exception as exc:  # noqa: BLE001 - provider errors are opaque by design
            # Same wording as `complete`, and for the same reason: the SDK's
            # message may name a status code worth reading, and never the key.
            raise LLMError(f"generation request failed: {type(exc).__name__}: {exc}") from exc

        result = LLMResponse(
            text="".join(pieces).strip(),
            model=state.model,
            finish_reason=state.finish_reason,
            prompt_tokens=state.prompt_tokens,
            completion_tokens=state.completion_tokens,
        )
        if self._usage_recorder is not None:
            self._usage_recorder(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
            )
        return result


@dataclass
class _StreamState:
    """What a chat-completion stream reveals one chunk at a time.

    A streamed response is the same object as a completed one, delivered in
    pieces that each carry a different part of it: text in most, the finish
    reason in one, usage in the last. Accumulating that in a small mutable
    object keeps :meth:`OpenAILLM.stream` readable and keeps the "which chunk
    had which field?" knowledge in one place.
    """

    model: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    def absorb(self, chunk: Any) -> str:
        """Record what *chunk* carries and return its text delta, if any."""
        self.model = str(getattr(chunk, "model", None) or self.model)
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            self.prompt_tokens = getattr(usage, "prompt_tokens", None)
            self.completion_tokens = getattr(usage, "completion_tokens", None)
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            # The usage-only chunk `include_usage` appends after the last
            # content chunk. Not an error, and not text.
            return ""
        choice = choices[0]
        self.finish_reason = getattr(choice, "finish_reason", None) or self.finish_reason
        delta = getattr(choice, "delta", None)
        content = getattr(delta, "content", None)
        return str(content) if content else ""


def build_llm(
    kind: str = LLM_FAKE,
    *,
    config: GenerationConfig | None = None,
    api_key: str | None = None,
    usage_recorder: Callable[..., None] | None = None,
) -> LLM:
    """Construct the model named by *kind*.

    Args:
        kind: One of :data:`LLM_KINDS`. Defaults to ``fake`` so a caller that
            forgets to choose gets the offline path rather than a bill.
        config: The ``generation`` block; documented defaults when omitted.
        api_key: Credential for a hosted provider. Read from the environment by
            the caller — never a CLI flag, since a key on a command line ends up
            in shell history.
        usage_recorder: Optional response-accounting callback used by evals.

    Raises:
        LLMError: Unknown kind, or a hosted provider with no key.
    """
    settings = config or GenerationConfig()
    resolved = kind.strip().lower()
    if resolved not in LLM_KINDS:
        raise LLMError(f"unknown llm {kind!r}; expected one of {', '.join(LLM_KINDS)}")
    if resolved == LLM_FAKE:
        return FakeLLM(usage_recorder=usage_recorder)
    _log.info("llm_selected", provider=LLM_OPENAI, model=settings.model)
    return OpenAILLM(
        api_key=api_key or "",
        model=settings.model,
        temperature=settings.temperature,
        max_output_tokens=settings.max_output_tokens,
        timeout_seconds=settings.timeout_seconds,
        max_retries=settings.max_retries,
        usage_recorder=usage_recorder,
    )


def _cited_sentence(sentence: str, marker: int) -> str:
    """Attach ``[n]`` inside the sentence it supports, before the full stop.

    Matches what the system prompt asks a real model for, so the offline path
    and the hosted one produce the same shape and the same citation-coverage
    measurement rather than differing by a punctuation convention.
    """
    if sentence[-1] in ".!?":
        return f"{sentence[:-1].rstrip()} [{marker}]{sentence[-1]}"
    return f"{sentence} [{marker}]."


def _first_sentence(text: str) -> str:
    """First sentence of *text*, collapsed onto one line."""
    flat = " ".join(text.split())
    if not flat:
        return ""
    return _SENTENCE_END.split(flat, maxsplit=1)[0].strip()
