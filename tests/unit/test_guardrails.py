"""Unit tests for the guardrails: when this pipeline is allowed to refuse."""

from __future__ import annotations

from production_rag.config_loader import CitationsConfig
from production_rag.generation import guardrails
from production_rag.generation.citations import Citation
from production_rag.generation.prompts import ABSTAIN_TOKEN
from production_rag.retrieval.hybrid import RetrievalHit

HIT = RetrievalHit(
    chunk_id="a",
    source_path="data/raw/a.md",
    text="Qdrant holds both vector kinds.",
    score=0.5,
    rank=1,
)
CITATION = Citation(
    marker=1,
    chunk_id="a",
    source_path="data/raw/a.md",
    text=HIT.text,
    score=0.5,
    rank=1,
)


class TestEvidenceCheck:
    def test_no_hits_refuses_before_the_model_is_called(self) -> None:
        refusal = guardrails.check_evidence([])
        assert refusal is not None
        assert refusal.reason == guardrails.REASON_NO_EVIDENCE
        assert refusal.message == CitationsConfig().refusal_message

    def test_hits_pass(self) -> None:
        assert guardrails.check_evidence([HIT]) is None

    def test_the_refusal_can_be_switched_off(self) -> None:
        config = CitationsConfig(refuse_without_evidence=False)
        assert guardrails.check_evidence([], config=config) is None

    def test_the_message_is_configurable(self) -> None:
        config = CitationsConfig(refusal_message="no idea, sorry")
        refusal = guardrails.check_evidence([], config=config)
        assert refusal is not None
        assert refusal.message == "no idea, sorry"


class TestAnswerCheck:
    def test_a_cited_answer_is_served(self) -> None:
        assert guardrails.check_answer("Grounded [1].", [CITATION]) is None

    def test_an_uncited_answer_is_refused(self) -> None:
        refusal = guardrails.check_answer("A confident, unsourced claim.", [])
        assert refusal is not None
        assert refusal.reason == guardrails.REASON_NO_CITATIONS

    def test_the_citation_requirement_can_be_relaxed(self) -> None:
        config = CitationsConfig(require_citation=False)
        assert guardrails.check_answer("Unsourced.", [], config=config) is None

    def test_whitespace_is_not_an_answer(self) -> None:
        refusal = guardrails.check_answer("   \n ", [CITATION])
        assert refusal is not None
        assert refusal.reason == guardrails.REASON_EMPTY_ANSWER

    def test_the_abstain_sentinel_is_a_refusal(self) -> None:
        refusal = guardrails.check_answer(ABSTAIN_TOKEN, [])
        assert refusal is not None
        assert refusal.reason == guardrails.REASON_MODEL_ABSTAINED

    def test_a_sentinel_wrapped_in_prose_is_still_a_refusal(self) -> None:
        # Models routinely dress a sentinel in a sentence; serving that as an
        # answer would show a user a refusal formatted like a result.
        refusal = guardrails.check_answer(f"I have to say {ABSTAIN_TOKEN} here [1].", [CITATION])
        assert refusal is not None
        assert refusal.reason == guardrails.REASON_MODEL_ABSTAINED

    def test_every_reason_is_in_the_declared_set(self) -> None:
        reasons = {
            guardrails.check_evidence([]),
            guardrails.check_answer("", []),
            guardrails.check_answer(ABSTAIN_TOKEN, []),
            guardrails.check_answer("Unsourced but long enough to be a claim.", []),
        }
        assert all(
            refusal is not None and refusal.reason in guardrails.REFUSAL_REASONS
            for refusal in reasons
        )


class TestUncitedClaims:
    def test_a_fully_cited_answer_flags_nothing(self) -> None:
        answer = "Qdrant holds both vector kinds [1]. RRF merges by position [2]."
        assert guardrails.uncited_sentences(answer) == ()

    def test_a_marker_after_the_full_stop_still_counts_as_cited(self) -> None:
        answer = "Qdrant holds both vector kinds. [1] RRF merges by position. [2]"
        assert guardrails.uncited_sentences(answer) == ()

    def test_an_uncited_sentence_is_reported(self) -> None:
        answer = "Qdrant holds both vector kinds [1]. This part is entirely made up."
        assert guardrails.uncited_sentences(answer) == ("This part is entirely made up.",)

    def test_a_short_connective_is_not_a_claim(self) -> None:
        answer = "Qdrant holds both vector kinds [1]. In short."
        assert guardrails.uncited_sentences(answer) == ()

    def test_reporting_is_not_refusing(self) -> None:
        # A partially uncited answer is a coverage signal, not a rejection: a
        # guardrail with a high false-positive rate is a guardrail that gets
        # switched off.
        answer = "Qdrant holds both vector kinds [1]. This part is entirely made up."
        assert guardrails.check_answer(answer, [CITATION]) is None
        assert guardrails.uncited_sentences(answer)
