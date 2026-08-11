"""The LLM seam: one Protocol, one offline double, one real provider.

Same shape as :mod:`production_rag.retrieval.embeddings` and
:mod:`production_rag.retrieval.rerank`, for the same reason — every expensive or
networked dependency in this project sits behind a Protocol with a deterministic
offline implementation, so the unit suite is green with the wifi off and nobody
needs a credential to run the pipeline end to end.

* :class:`FakeLLM` — extractive, deterministic, offline. It reads the numbered
  context blocks back out of the prompt it was handed and answers with the
  leading sentence of each, carrying the matching ``[n]`` marker; with no blocks
  it abstains. It is a **plumbing double, not a quality claim**: it does not
  understand the question, it quotes the top passages. What it does guarantee is
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
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import structlog

from production_rag.config_loader import GenerationConfig
from production_rag.generation.prompts import ABSTAIN_TOKEN, ChatMessage, parse_context_blocks

_log = structlog.get_logger(__name__)

LLM_FAKE = "fake"
LLM_OPENAI = "openai"
LLM_KINDS = (LLM_FAKE, LLM_OPENAI)
"""What ``build_llm`` accepts. ``fake`` is the offline path; ``openai`` needs a key."""

FAKE_MODEL = "fake-extractive-v1"
"""Recorded in the result, so an eval row can never be mistaken for a real run."""

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


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

    def __init__(self, *, max_sentences: int = 2, model: str = FAKE_MODEL) -> None:
        """Configure the double.

        Args:
            max_sentences: How many blocks to quote. Two by default: enough that
                multi-citation answers are exercised, few enough that the output
                stays readable in a CLI.
            model: Identifier reported in the result.
        """
        self._max_sentences = max(1, max_sentences)
        self._model = model

    @property
    def model(self) -> str:
        """The identifier reported in the result."""
        return self._model

    def complete(self, messages: Sequence[ChatMessage]) -> LLMResponse:
        """Answer from the numbered blocks in the last user message."""
        user = next(
            (message.content for message in reversed(list(messages)) if message.role == "user"),
            "",
        )
        blocks = parse_context_blocks(user)
        if not blocks:
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
        return LLMResponse(
            text=str(content).strip(),
            model=str(getattr(response, "model", self._model)),
            finish_reason=getattr(choices[0], "finish_reason", None),
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
        )


def build_llm(
    kind: str = LLM_FAKE,
    *,
    config: GenerationConfig | None = None,
    api_key: str | None = None,
) -> LLM:
    """Construct the model named by *kind*.

    Args:
        kind: One of :data:`LLM_KINDS`. Defaults to ``fake`` so a caller that
            forgets to choose gets the offline path rather than a bill.
        config: The ``generation`` block; documented defaults when omitted.
        api_key: Credential for a hosted provider. Read from the environment by
            the caller — never a CLI flag, since a key on a command line ends up
            in shell history.

    Raises:
        LLMError: Unknown kind, or a hosted provider with no key.
    """
    settings = config or GenerationConfig()
    resolved = kind.strip().lower()
    if resolved not in LLM_KINDS:
        raise LLMError(f"unknown llm {kind!r}; expected one of {', '.join(LLM_KINDS)}")
    if resolved == LLM_FAKE:
        return FakeLLM()
    _log.info("llm_selected", provider=LLM_OPENAI, model=settings.model)
    return OpenAILLM(
        api_key=api_key or "",
        model=settings.model,
        temperature=settings.temperature,
        max_output_tokens=settings.max_output_tokens,
        timeout_seconds=settings.timeout_seconds,
        max_retries=settings.max_retries,
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
