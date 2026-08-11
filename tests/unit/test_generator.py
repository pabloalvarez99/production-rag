"""Unit tests for grounded generation. Offline: FakeLLM and hand-written doubles."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from production_rag.config_loader import CitationsConfig, GenerationConfig, PromptConfig
from production_rag.generation import guardrails
from production_rag.generation.generator import generate_answer
from production_rag.generation.llm import FakeLLM, LLMError, LLMResponse
from production_rag.generation.prompts import ABSTAIN_TOKEN, ChatMessage
from production_rag.retrieval.hybrid import RetrievalHit


def make_hit(chunk_id: str, text: str, rank: int) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        source_path=f"data/raw/{chunk_id}.md",
        text=text,
        score=1.0 / rank,
        rank=rank,
        title=f"Doc {chunk_id}",
        point_id=f"point-{chunk_id}",
    )


HITS = [
    make_hit("a", "Qdrant stores dense and sparse vectors in one collection.", 1),
    make_hit("b", "Reciprocal rank fusion merges ranked lists by position.", 2),
    make_hit("c", "A cross-encoder reads the query and the passage together.", 3),
]


class ScriptedLLM:
    """Returns a fixed answer, and records the prompt it was given."""

    def __init__(self, text: str, *, model: str = "scripted") -> None:
        self._text = text
        self._model = model
        self.prompts: list[tuple[ChatMessage, ...]] = []

    @property
    def model(self) -> str:
        return self._model

    def complete(self, messages: Sequence[ChatMessage]) -> LLMResponse:
        self.prompts.append(tuple(messages))
        return LLMResponse(text=self._text, model=self._model, finish_reason="stop")


class ExplodingLLM:
    """Fails the test if it is called at all."""

    @property
    def model(self) -> str:
        return "exploding"

    def complete(self, messages: Sequence[ChatMessage]) -> LLMResponse:
        raise AssertionError("the model must not be called without evidence")


class OutageLLM:
    """A provider that is down. Distinct from a model that declines to answer."""

    @property
    def model(self) -> str:
        return "outage"

    def complete(self, messages: Sequence[ChatMessage]) -> LLMResponse:
        raise LLMError("generation request failed: TimeoutError: upstream")


class TestHappyPath:
    def test_an_answer_carries_citations_and_the_model(self) -> None:
        result = generate_answer("what is fusion?", HITS, llm=FakeLLM())
        assert result.refused is False
        assert result.refusal_reason is None
        assert result.citations
        assert result.model == FakeLLM().model
        assert result.hits_used == 3

    def test_citations_resolve_to_retrieved_chunks(self) -> None:
        result = generate_answer("q", HITS, llm=ScriptedLLM("Grounded [2]."))
        assert [citation.chunk_id for citation in result.citations] == ["b"]

    def test_the_hits_reach_the_prompt(self) -> None:
        llm = ScriptedLLM("Grounded [1].")
        generate_answer("what is fusion?", HITS, llm=llm)
        user = llm.prompts[0][1].content
        assert "what is fusion?" in user
        assert all(hit.text in user for hit in HITS)

    def test_uncited_sentences_are_reported_on_a_served_answer(self) -> None:
        answer = "Grounded [1]. This sentence has no support whatsoever."
        result = generate_answer("q", HITS, llm=ScriptedLLM(answer))
        assert result.refused is False
        assert result.uncited_claims == ("This sentence has no support whatsoever.",)

    def test_the_result_serialises(self) -> None:
        payload = generate_answer("q", HITS, llm=FakeLLM()).to_dict()
        assert payload["refused"] is False
        assert payload["citations"]
        assert "text" in payload["citations"][0]
        assert (
            "text"
            not in generate_answer("q", HITS, llm=FakeLLM()).to_dict(include_citation_text=False)[
                "citations"
            ][0]
        )


class TestRefusals:
    def test_no_hits_refuses_without_calling_the_model(self) -> None:
        result = generate_answer("q", [], llm=ExplodingLLM())
        assert result.refused is True
        assert result.refusal_reason == guardrails.REASON_NO_EVIDENCE
        assert result.citations == ()
        assert result.answer == CitationsConfig().refusal_message

    def test_an_abstaining_model_is_a_refusal(self) -> None:
        result = generate_answer("q", HITS, llm=ScriptedLLM(ABSTAIN_TOKEN))
        assert result.refused is True
        assert result.refusal_reason == guardrails.REASON_MODEL_ABSTAINED

    def test_an_uncited_answer_is_refused(self) -> None:
        result = generate_answer("q", HITS, llm=ScriptedLLM("Confident and unsourced."))
        assert result.refused is True
        assert result.refusal_reason == guardrails.REASON_NO_CITATIONS

    def test_an_answer_whose_every_marker_was_invented_is_refused(self) -> None:
        result = generate_answer("q", HITS, llm=ScriptedLLM("Grounded [9]."))
        assert result.refused is True
        assert result.refusal_reason == guardrails.REASON_NO_CITATIONS

    def test_the_citation_requirement_can_be_relaxed(self) -> None:
        config = GenerationConfig(citations=CitationsConfig(require_citation=False))
        result = generate_answer("q", HITS, llm=ScriptedLLM("Unsourced."), config=config)
        assert result.refused is False
        assert result.citations == ()

    def test_generating_without_evidence_can_be_allowed(self) -> None:
        config = GenerationConfig(
            citations=CitationsConfig(refuse_without_evidence=False, require_citation=False)
        )
        result = generate_answer("q", [], llm=ScriptedLLM("From memory."), config=config)
        assert result.refused is False
        assert result.hits_used == 0

    def test_a_refusal_reports_no_model_when_none_was_called(self) -> None:
        assert generate_answer("q", [], llm=ExplodingLLM()).model is None


class TestOutagesAreNotRefusals:
    def test_a_provider_failure_propagates(self) -> None:
        # An outage and "the documents do not say" are different facts; a caller
        # that cannot tell them apart retries the wrong one.
        with pytest.raises(LLMError):
            generate_answer("q", HITS, llm=OutageLLM())


class TestTruncation:
    def test_the_chunk_ceiling_bounds_what_the_model_sees(self) -> None:
        config = GenerationConfig(prompt=PromptConfig(max_chunks_in_prompt=2))
        llm = ScriptedLLM("Grounded [1].")
        result = generate_answer("q", HITS, llm=llm, config=config)
        assert result.hits_used == 2
        assert result.dropped_hits == 1
        assert HITS[2].text not in llm.prompts[0][1].content

    def test_a_marker_past_the_truncation_point_is_dropped(self) -> None:
        config = GenerationConfig(prompt=PromptConfig(max_chunks_in_prompt=1))
        result = generate_answer("q", HITS, llm=ScriptedLLM("Grounded [1] and [3]."), config=config)
        assert [citation.marker for citation in result.citations] == [1]
        assert result.invalid_markers == (3,)


class TestPromptOverride:
    def test_an_explicit_system_prompt_is_used(self) -> None:
        llm = ScriptedLLM("Grounded [1].")
        generate_answer("q", HITS, llm=llm, system_prompt="be terse")
        assert llm.prompts[0][0].content == "be terse"
