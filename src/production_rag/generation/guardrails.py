"""Guardrails: the two places this pipeline is allowed to refuse.

    before the call:  no evidence            -> refuse, do not spend a token
    after the call:   nothing citable        -> refuse, do not serve it

Refusal is a **feature of a RAG system, not a failure of one**. The failure mode
that destroys trust is an answer that sounds exactly like a grounded one and is
not, and a user cannot tell those apart by reading. So both checks live here,
both produce a machine-readable reason code, and neither is spread across the
call site.

The pre-check also has a cost argument: with no hits there is nothing to ground
an answer in, so calling the model can only produce a hallucination that has
been paid for.

The post-check is the narrow one on purpose. It refuses when the answer resolves
to **no** citation at all, or when the model emitted
:data:`~production_rag.generation.prompts.ABSTAIN_TOKEN`. It does *not* refuse
because one sentence in an otherwise cited answer lacks a marker — that would
turn "the second sentence is a transition" into a refusal, and a guardrail with
a high false-positive rate gets switched off, which leaves the system with none.
Partially-uncited answers are **reported** instead (:func:`uncited_sentences`),
which is the signal an eval scores and a reviewer reads.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import structlog

from production_rag.config_loader import CitationsConfig
from production_rag.generation.citations import Citation
from production_rag.generation.prompts import ABSTAIN_TOKEN
from production_rag.retrieval.hybrid import RetrievalHit

_log = structlog.get_logger(__name__)

REASON_NO_EVIDENCE = "no_evidence"
"""Retrieval returned nothing; the model was never called."""

REASON_MODEL_ABSTAINED = "model_abstained"
"""The model said the context does not support an answer."""

REASON_NO_CITATIONS = "no_citations"
"""An answer was produced, but nothing in it resolved to a retrieved chunk."""

REASON_EMPTY_ANSWER = "empty_answer"
"""The model returned only whitespace."""

REFUSAL_REASONS = (
    REASON_NO_EVIDENCE,
    REASON_MODEL_ABSTAINED,
    REASON_NO_CITATIONS,
    REASON_EMPTY_ANSWER,
)
"""Every reason this pipeline can refuse for. A closed set, so a caller can
branch on it and an eval can group by it."""

# A marker that trails a full stop ("... by position. [1]") belongs to the
# sentence before it, not to the one after. Splitting there would report a cited
# sentence as uncited on every answer that puts the marker outside the period,
# which is a formatting choice, not a grounding failure.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?!\[\d)")
_MARKER = re.compile(r"\[\d+\]")
_MIN_CLAIM_CHARS = 24
"""Below this, a sentence is a connective ("In short.") rather than a claim, and
flagging it as uncited would bury the real signal in noise."""


@dataclass(frozen=True, slots=True)
class Refusal:
    """A decision not to answer, with a code and the text to show.

    The code is what an eval groups by and what an operator alerts on; the
    message is what a user reads. Keeping both means the user-facing wording can
    change without invalidating a dashboard.
    """

    reason: str
    message: str


def check_evidence(
    hits: Sequence[RetrievalHit],
    *,
    config: CitationsConfig | None = None,
) -> Refusal | None:
    """Decide whether there is anything to ground an answer in.

    Args:
        hits: What retrieval (and any rerank) returned.
        config: The ``generation.citations`` block; defaults when omitted.

    Returns:
        A :class:`Refusal` when the model must not be called, else ``None``.
        With ``refuse_without_evidence`` disabled this always returns ``None`` —
        an escape hatch for a deployment that wants a general-knowledge fallback,
        and one that gives up the central promise of the system, which is why it
        is off by default.
    """
    settings = config or CitationsConfig()
    if hits:
        return None
    if not settings.refuse_without_evidence:
        return None
    _log.info("refused", reason=REASON_NO_EVIDENCE)
    return Refusal(reason=REASON_NO_EVIDENCE, message=settings.refusal_message)


def check_answer(
    answer: str,
    citations: Sequence[Citation],
    *,
    config: CitationsConfig | None = None,
) -> Refusal | None:
    """Decide whether a produced answer may be served.

    Args:
        answer: The model's text, after invalid markers were stripped.
        citations: What resolved against the prompt's blocks.
        config: The ``generation.citations`` block; defaults when omitted.

    Returns:
        A :class:`Refusal`, or ``None`` when the answer stands.

    Note:
        The abstention check runs against the *raw* sentinel appearing anywhere
        in the text, not only as the whole message: models routinely wrap a
        sentinel in a sentence, and treating that as an answer would serve the
        user a refusal dressed as a result.
    """
    settings = config or CitationsConfig()
    stripped = answer.strip()
    if not stripped:
        _log.info("refused", reason=REASON_EMPTY_ANSWER)
        return Refusal(reason=REASON_EMPTY_ANSWER, message=settings.refusal_message)
    if ABSTAIN_TOKEN in stripped:
        _log.info("refused", reason=REASON_MODEL_ABSTAINED)
        return Refusal(reason=REASON_MODEL_ABSTAINED, message=settings.refusal_message)
    if settings.require_citation and not citations:
        _log.info("refused", reason=REASON_NO_CITATIONS, answer_chars=len(stripped))
        return Refusal(reason=REASON_NO_CITATIONS, message=settings.refusal_message)
    return None


def uncited_sentences(answer: str) -> tuple[str, ...]:
    """Sentences in *answer* that carry no citation marker.

    Reported, never fatal — see the module docstring. A non-empty result on a
    served answer is the number to watch: it is citation coverage measured on
    every request, not sampled by an offline judge.

    Very short sentences are ignored, because a connective is not a claim.
    """
    flagged = [
        sentence.strip()
        for sentence in _SENTENCE_SPLIT.split(answer.strip())
        if len(sentence.strip()) >= _MIN_CLAIM_CHARS and not _MARKER.search(sentence)
    ]
    return tuple(flagged)
