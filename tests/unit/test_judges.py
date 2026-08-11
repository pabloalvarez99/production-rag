"""Unit tests for the answer judges. Offline: no key, no network, no ragas."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from production_rag.evals.judges import (
    JUDGE_KINDS,
    AnswerJudge,
    FakeJudge,
    JudgeError,
    Judgement,
    OpenAIJudge,
    build_judge,
    tokenise,
)
from production_rag.generation.llm import LLMError, LLMResponse
from production_rag.generation.prompts import ChatMessage

PASSAGES = (
    "Reciprocal rank fusion merges two ranked lists by position.",
    "A cross-encoder scores the query and the passage together.",
)


class ScriptedJudgeLLM:
    """Returns a fixed reply and records the prompt it was given."""

    def __init__(self, text: str, *, model: str = "gpt-judge") -> None:
        self._text = text
        self._model = model
        self.prompts: list[tuple[ChatMessage, ...]] = []

    @property
    def model(self) -> str:
        return self._model

    def complete(self, messages: Sequence[ChatMessage]) -> LLMResponse:
        self.prompts.append(tuple(messages))
        return LLMResponse(text=self._text, model=self._model, finish_reason="stop")


class OutageJudgeLLM:
    """A judge model that is down."""

    @property
    def model(self) -> str:
        return "outage"

    def complete(self, messages: Sequence[ChatMessage]) -> LLMResponse:
        raise LLMError("generation request failed: TimeoutError: upstream")


class TestTokenise:
    def test_citation_markers_are_not_words(self) -> None:
        # Otherwise an answer scores faithfulness for the brackets it printed.
        assert "1" not in tokenise("Fusion merges lists [1].")

    def test_stopwords_are_dropped(self) -> None:
        assert tokenise("the and of") == set()

    def test_short_identifiers_survive(self) -> None:
        # The corpus really does contain `k1` and `b`; dropping them would make
        # the exact-token cases unscorable.
        assert tokenise("k1 and b control saturation") >= {"k1", "b"}

    def test_case_is_folded(self) -> None:
        assert tokenise("Qdrant") == tokenise("qdrant")


class TestFakeJudge:
    def test_it_satisfies_the_protocol(self) -> None:
        assert isinstance(FakeJudge(), AnswerJudge)

    def test_it_is_deterministic(self) -> None:
        first = FakeJudge().score("q", "Fusion merges lists [1].", PASSAGES)
        second = FakeJudge().score("q", "Fusion merges lists [1].", PASSAGES)
        assert first == second

    def test_a_fully_grounded_answer_scores_one(self) -> None:
        answer = "Reciprocal rank fusion merges two ranked lists by position [1]."
        assert FakeJudge().score("q", answer, PASSAGES).faithfulness == 1.0

    def test_an_answer_from_nowhere_scores_low(self) -> None:
        judgement = FakeJudge().score("q", "Kubernetes autoscaling uses metrics.", PASSAGES)
        assert judgement.faithfulness is not None
        assert judgement.faithfulness < 0.5

    def test_relevance_measures_the_question_not_the_passages(self) -> None:
        judgement = FakeJudge().score(
            "What does a cross-encoder do?",
            "A cross-encoder scores the query and the passage together [2].",
            PASSAGES,
        )
        assert judgement.relevance == 1.0

    def test_a_refusal_is_not_scored(self) -> None:
        # A refusal makes no claims, so scoring it zero would punish the system
        # for the behaviour refusal_accuracy exists to reward.
        judgement = FakeJudge().score("q", "I could not find support.", PASSAGES, refused=True)
        assert judgement.faithfulness is None
        assert judgement.relevance is None

    def test_no_passages_means_no_faithfulness_score(self) -> None:
        assert FakeJudge().score("q", "Something.", []).faithfulness == 0.0

    def test_an_empty_answer_declines_rather_than_scoring_zero(self) -> None:
        assert FakeJudge().score("q", "", PASSAGES).faithfulness is None

    def test_the_judge_name_is_versioned(self) -> None:
        # A scoring change must be visible as a different judge, not as a
        # quality movement.
        assert FakeJudge().score("q", "a", PASSAGES).judge == FakeJudge().name
        assert "v1" in FakeJudge().name


class TestOpenAIJudge:
    def test_it_parses_the_scores(self) -> None:
        llm = ScriptedJudgeLLM('{"faithfulness": 0.8, "relevance": 0.9, "rationale": "ok"}')
        judgement = OpenAIJudge(llm).score("q", "an answer", PASSAGES)
        assert judgement.faithfulness == 0.8
        assert judgement.relevance == 0.9
        assert judgement.rationale == "ok"

    def test_prose_around_the_json_is_tolerated(self) -> None:
        llm = ScriptedJudgeLLM('Sure!\n{"faithfulness": 1, "relevance": 1}\nHope that helps.')
        assert OpenAIJudge(llm).score("q", "a", PASSAGES).faithfulness == 1.0

    def test_the_question_answer_and_passages_reach_the_model(self) -> None:
        llm = ScriptedJudgeLLM('{"faithfulness": 1, "relevance": 1}')
        OpenAIJudge(llm).score("what is fusion?", "an answer", PASSAGES)
        user = llm.prompts[0][1].content
        assert "what is fusion?" in user
        assert "an answer" in user
        assert all(passage in user for passage in PASSAGES)

    def test_out_of_range_scores_are_clamped(self) -> None:
        llm = ScriptedJudgeLLM('{"faithfulness": 1.7, "relevance": -2}')
        judgement = OpenAIJudge(llm).score("q", "a", PASSAGES)
        assert judgement.faithfulness == 1.0
        assert judgement.relevance == 0.0

    def test_a_non_numeric_score_is_dropped_not_zeroed(self) -> None:
        llm = ScriptedJudgeLLM('{"faithfulness": "high", "relevance": 0.5}')
        assert OpenAIJudge(llm).score("q", "a", PASSAGES).faithfulness is None

    def test_an_unparseable_reply_is_an_error(self) -> None:
        # Not a zero: a judge that failed and an answer that is bad must not
        # look the same in the report.
        with pytest.raises(JudgeError, match="no JSON object"):
            OpenAIJudge(ScriptedJudgeLLM("I would rather not.")).score("q", "a", PASSAGES)

    def test_malformed_json_is_an_error(self) -> None:
        with pytest.raises(JudgeError, match="not valid JSON"):
            OpenAIJudge(ScriptedJudgeLLM('{"faithfulness": }')).score("q", "a", PASSAGES)

    def test_a_provider_outage_is_an_error(self) -> None:
        with pytest.raises(JudgeError, match="judge model failed"):
            OpenAIJudge(OutageJudgeLLM()).score("q", "a", PASSAGES)

    def test_a_refusal_is_not_sent_to_the_model(self) -> None:
        # No call, so no bill and no noise from grading a fixed string.
        llm = ScriptedJudgeLLM('{"faithfulness": 1, "relevance": 1}')
        judgement = OpenAIJudge(llm).score("q", "I could not find support.", PASSAGES, refused=True)
        assert llm.prompts == []
        assert judgement.faithfulness is None

    def test_the_name_carries_the_model(self) -> None:
        assert OpenAIJudge(ScriptedJudgeLLM("{}", model="gpt-4o")).name == "openai:gpt-4o"


class TestBuildJudge:
    def test_the_default_is_offline(self) -> None:
        assert isinstance(build_judge(), FakeJudge)

    def test_an_unknown_kind_is_an_error(self) -> None:
        with pytest.raises(JudgeError, match="unknown judge"):
            build_judge("vibes")

    def test_ragas_is_not_a_judge_kind(self) -> None:
        # ADR-0003 names Ragas and the `eval` extra declares it, but nothing here
        # runs it. A kind named "ragas" would put its label on numbers it never
        # computed.
        assert "ragas" not in JUDGE_KINDS
        with pytest.raises(JudgeError, match="unknown judge"):
            build_judge("ragas")

    def test_an_explicit_model_is_used_for_the_hosted_judge(self) -> None:
        judge = build_judge("openai", llm=ScriptedJudgeLLM("{}", model="gpt-4o"))
        assert judge.name == "openai:gpt-4o"

    def test_a_hosted_judge_without_a_key_fails_to_build(self) -> None:
        with pytest.raises(JudgeError, match="judge model unavailable"):
            build_judge("openai", api_key="")


class TestJudgement:
    def test_it_serialises(self) -> None:
        payload = Judgement(faithfulness=0.5, relevance=None, judge="j").to_dict()
        assert payload == {
            "faithfulness": 0.5,
            "relevance": None,
            "rationale": "",
            "judge": "j",
        }
