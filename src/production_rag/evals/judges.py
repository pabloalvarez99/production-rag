"""Answer-side judges: a Protocol, an offline default, and a hosted adapter.

Tier 2 asks two questions no string comparison can answer — is the answer
supported by the passages it was given, and does it address the question — and
both are graded by something fallible. This module makes *which* something an
explicit, swappable choice:

* :class:`FakeJudge` — deterministic lexical overlap. No key, no network, no
  cost, and no claim to measure meaning. It exists so the tier-2 plumbing runs
  in CI on every change, and so a broken pipeline is caught by a number moving
  to zero rather than by nobody running the command.
* :class:`OpenAIJudge` — a model grading a model. Costs money, varies run to
  run, and is gated behind ``RUN_LLM_EVALS=1`` plus a credential.

The honest framing, which the reports repeat: a **FakeJudge score is not a
quality measurement.** Word overlap rewards an answer that copies the passage
and punishes a correct paraphrase, which is close to the opposite of what
faithfulness means. It is a smoke test with a number attached. Every report
carries the judge's name so no reader has to guess which kind of number they
are looking at.

Ragas is deliberately absent. ADR-0003 names it, the ``eval`` extra declares it,
and wiring it here without ever running it would produce a column labelled
"ragas" holding numbers Ragas never computed — the one failure this whole file
is arranged to prevent. See :data:`JUDGE_KINDS`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import structlog

from production_rag.generation.llm import LLM, LLMError, build_llm
from production_rag.generation.prompts import ChatMessage

if TYPE_CHECKING:
    from production_rag.config_loader import GenerationConfig

_log = structlog.get_logger(__name__)

JUDGE_FAKE = "fake"
JUDGE_OPENAI = "openai"
JUDGE_KINDS = (JUDGE_FAKE, JUDGE_OPENAI)
"""Judges that exist. ``ragas`` is not among them, on purpose — see the module
docstring: a name here would let a report claim numbers nothing computed."""

LLM_EVALS_ENV = "RUN_LLM_EVALS"
"""Opt-in for judges that cost money. Checked by the runner, not here."""

FAKE_JUDGE_NAME = "fake-overlap-v1"
"""Versioned, because a scoring change must not look like a quality change."""

_WORD = re.compile(r"[a-z0-9_]+")
_MARKER = re.compile(r"\[\d+\]")
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

_STOPWORD_TEXT = """
a an and are as at be but by for from has have how in is it its of on or that the
their there these this to was were what when where which who why will with does do
"""
_STOPWORDS = frozenset(_STOPWORD_TEXT.split())


class JudgeError(RuntimeError):
    """The judge could not produce a score."""


@dataclass(frozen=True, slots=True)
class Judgement:
    """One judge's verdict on one answer.

    Scores are in ``[0, 1]`` or ``None`` when the judge declines to score — a
    refusal has no claims to be faithful about, and averaging a zero there would
    punish the system for doing the right thing.
    """

    faithfulness: float | None = None
    relevance: float | None = None
    rationale: str = ""
    judge: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form for the per-case report."""
        return {
            "faithfulness": self.faithfulness,
            "relevance": self.relevance,
            "rationale": self.rationale,
            "judge": self.judge,
        }


@runtime_checkable
class AnswerJudge(Protocol):
    """Scores one answer against the passages it was given."""

    @property
    def name(self) -> str:
        """Judge identity, recorded in every report that quotes its numbers."""
        ...

    def score(
        self,
        question: str,
        answer: str,
        contexts: Sequence[str],
        *,
        refused: bool = False,
    ) -> Judgement:
        """Grade one answer.

        Args:
            question: The golden question, verbatim.
            answer: What the pipeline produced.
            contexts: The passage texts that were actually cited or retrieved —
                the evidence the answer was supposed to stay inside.
            refused: Whether the pipeline refused. A refusal is graded as a
                decision, not as prose.

        Returns:
            A :class:`Judgement`.

        Raises:
            JudgeError: The judge failed and no score can be reported. Never a
                zero: a failed judge and a bad answer must not look alike.
        """
        ...


def tokenise(text: str) -> set[str]:
    """Lowercase content words, minus citation markers and stopwords.

    Args:
        text: Any prose.

    Returns:
        A set of tokens. Markers are stripped first, so ``[1]`` never counts as
        evidence of anything, and single characters are kept because the corpus
        genuinely contains ``k1`` and ``b``.
    """
    stripped = _MARKER.sub(" ", text.lower())
    return {word for word in _WORD.findall(stripped) if word not in _STOPWORDS}


def _overlap(subject: set[str], reference: set[str]) -> float | None:
    """Fraction of *subject* present in *reference*, or ``None`` if empty."""
    if not subject:
        return None
    return len(subject & reference) / len(subject)


class FakeJudge:
    """Deterministic lexical overlap. Offline, free, and not a quality metric.

    ``faithfulness`` is the share of the answer's content words that appear in
    the supplied passages, and ``relevance`` is the share of the question's
    content words that appear in the answer. Both are computable with no model,
    which is the entire point: tier 2 has to run on a laptop with no key, or it
    does not run.

    What it actually catches is a pipeline that broke — an answer built from the
    wrong passages, a generator ignoring its context, an empty answer. What it
    cannot catch is a fluent wrong answer that reuses the passage's vocabulary,
    which is the failure mode a judge is wanted for in the first place.
    """

    __slots__ = ()

    @property
    def name(self) -> str:
        """The versioned fake-judge identity."""
        return FAKE_JUDGE_NAME

    def score(
        self,
        question: str,
        answer: str,
        contexts: Sequence[str],
        *,
        refused: bool = False,
    ) -> Judgement:
        """Score by word overlap, or decline to score a refusal."""
        if refused:
            return Judgement(
                faithfulness=None,
                relevance=None,
                rationale="refused; scored by refusal_accuracy instead",
                judge=self.name,
            )

        answer_tokens = tokenise(answer)
        context_tokens: set[str] = set()
        for context in contexts:
            context_tokens |= tokenise(context)

        return Judgement(
            faithfulness=_round(_overlap(answer_tokens, context_tokens)),
            relevance=_round(_overlap(tokenise(question), answer_tokens)),
            rationale="lexical overlap; not a semantic judgement",
            judge=self.name,
        )


def _round(value: float | None) -> float | None:
    """Round a score, preserving ``None``."""
    return None if value is None else round(value, 4)


JUDGE_SYSTEM_PROMPT = """\
You are grading a retrieval-augmented answer. You are given a question, the \
passages the system retrieved, and the answer it produced.

Score two things, each from 0.0 to 1.0:
- faithfulness: how much of the answer is supported by the passages. An answer \
that states anything the passages do not support scores below 1.0, however \
plausible the statement is.
- relevance: how well the answer addresses the question that was asked.

Judge only against the passages. Do not use outside knowledge, and do not \
reward fluency, length or confidence.

Reply with one JSON object and nothing else:
{"faithfulness": <number>, "relevance": <number>, "rationale": "<one sentence>"}\
"""


class OpenAIJudge:
    """A hosted model grading the answer. Costs money and varies between runs.

    Takes an :class:`~production_rag.generation.llm.LLM` rather than building
    one, so the same Protocol that keeps generation testable keeps the judge
    testable: the unit tests drive this class with a scripted double and never
    reach the network.

    Uncalibrated, and ADR-0003 is explicit that an uncalibrated judge produces a
    number rather than a measurement. Nothing here compares its scores against
    hand labels yet, so treat a first run as a baseline to calibrate against and
    not as a grade.
    """

    __slots__ = ("_llm",)

    def __init__(self, llm: LLM) -> None:
        """Wrap a model to be used as a judge."""
        self._llm = llm

    @property
    def name(self) -> str:
        """The judge model, so a report says which model produced its numbers."""
        return f"openai:{self._llm.model}"

    def score(
        self,
        question: str,
        answer: str,
        contexts: Sequence[str],
        *,
        refused: bool = False,
    ) -> Judgement:
        """Ask the model for two scores and a one-line rationale."""
        if refused:
            return Judgement(
                faithfulness=None,
                relevance=None,
                rationale="refused; scored by refusal_accuracy instead",
                judge=self.name,
            )

        messages = (
            ChatMessage(role="system", content=JUDGE_SYSTEM_PROMPT),
            ChatMessage(role="user", content=_render_judge_request(question, answer, contexts)),
        )
        try:
            response = self._llm.complete(messages)
        except LLMError as exc:
            raise JudgeError(f"judge model failed: {exc}") from exc

        payload = _parse_scores(response.text)
        return Judgement(
            faithfulness=_clamp(payload.get("faithfulness")),
            relevance=_clamp(payload.get("relevance")),
            rationale=str(payload.get("rationale", ""))[:400],
            judge=self.name,
        )


def _render_judge_request(question: str, answer: str, contexts: Sequence[str]) -> str:
    """Lay out the judge's input in a fixed order, numbered like the prompt."""
    blocks = "\n\n".join(f"[{index}] {text}" for index, text in enumerate(contexts, start=1))
    return f"Question:\n{question}\n\nPassages:\n{blocks or '(none)'}\n\nAnswer:\n{answer}"


def _parse_scores(text: str) -> dict[str, Any]:
    """Pull the JSON object out of a judge reply.

    Raises:
        JudgeError: Nothing JSON-shaped came back. Deliberately an error rather
            than a default score: a judge that silently returns zero on a parse
            failure turns an infrastructure problem into a quality report.
    """
    match = _JSON_OBJECT.search(text)
    if match is None:
        raise JudgeError("judge reply contained no JSON object")
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise JudgeError(f"judge reply is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise JudgeError("judge reply is not a JSON object")
    return payload


def _clamp(value: object) -> float | None:
    """Coerce a judge's score into ``[0, 1]``, or ``None`` when it is not a number.

    A judge that answers ``1.5`` or ``"high"`` is misbehaving; clamping the
    first and dropping the second keeps one bad reply from poisoning an average
    while leaving the case visible as unscored.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return round(min(1.0, max(0.0, float(value))), 4)


def build_judge(
    kind: str = JUDGE_FAKE,
    *,
    config: GenerationConfig | None = None,
    api_key: str | None = None,
    llm: LLM | None = None,
) -> AnswerJudge:
    """Construct the judge named by *kind*.

    Args:
        kind: One of :data:`JUDGE_KINDS`. Defaults to ``fake``, so a caller who
            forgets to choose gets the offline judge rather than a bill.
        config: The ``generation`` block, used for the judge model's settings.
        api_key: Credential for a hosted judge, read from the environment by the
            caller — never a flag, since a key on a command line ends up in
            shell history.
        llm: An already-built model, for tests and for reusing one client.

    Returns:
        A judge.

    Raises:
        JudgeError: Unknown kind, or a hosted judge that cannot be built.
    """
    resolved = kind.strip().lower()
    if resolved not in JUDGE_KINDS:
        raise JudgeError(f"unknown judge {kind!r}; expected one of {', '.join(JUDGE_KINDS)}")
    if resolved == JUDGE_FAKE:
        return FakeJudge()

    if llm is None:
        try:
            llm = build_llm("openai", config=config, api_key=api_key)
        except LLMError as exc:
            raise JudgeError(f"judge model unavailable: {exc}") from exc
    _log.info("judge_selected", judge=JUDGE_OPENAI, model=llm.model)
    return OpenAIJudge(llm)
