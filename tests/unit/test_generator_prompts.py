"""Unit tests for prompt assembly and the LLM seam. Offline: no key, no network."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from production_rag.config_loader import GenerationConfig, PromptConfig
from production_rag.generation.llm import (
    FAKE_MODEL,
    LLM_KINDS,
    FakeLLM,
    LLMError,
    OpenAILLM,
    build_llm,
)
from production_rag.generation.prompts import (
    ABSTAIN_TOKEN,
    DEFAULT_SYSTEM_PROMPT,
    ChatMessage,
    ContextBlock,
    build_prompt,
    estimate_tokens,
    load_system_prompt,
    parse_context_blocks,
    select_blocks,
)
from production_rag.retrieval.hybrid import RetrievalHit


def make_hit(chunk_id: str, text: str, rank: int = 1) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        source_path=f"data/raw/{chunk_id}.md",
        text=text,
        score=1.0 / rank,
        rank=rank,
        title=f"Doc {chunk_id}",
        heading_path="Guide > Section",
        point_id=f"point-{chunk_id}",
    )


HITS = [
    make_hit("a", "Qdrant stores dense and sparse vectors in one collection.", 1),
    make_hit("b", "Reciprocal rank fusion merges ranked lists by position.", 2),
    make_hit("c", "The reranker reads the query and the passage together.", 3),
]


class TestBlockSelection:
    def test_markers_are_one_based_and_contiguous(self) -> None:
        blocks, dropped = select_blocks(HITS)
        assert [block.marker for block in blocks] == [1, 2, 3]
        assert dropped == 0

    def test_max_chunks_truncates_from_the_tail(self) -> None:
        blocks, dropped = select_blocks(HITS, max_chunks=2)
        assert [block.hit.chunk_id for block in blocks] == ["a", "b"]
        assert dropped == 1

    def test_token_budget_truncates_from_the_tail(self) -> None:
        blocks, dropped = select_blocks(HITS, max_context_tokens=40)
        assert dropped == len(HITS) - len(blocks)
        assert [block.marker for block in blocks] == list(range(1, len(blocks) + 1))

    def test_the_first_block_survives_a_budget_it_cannot_fit(self) -> None:
        # Dropping everything would turn an oversized chunk into "no information".
        blocks, dropped = select_blocks(HITS, max_context_tokens=1)
        assert len(blocks) == 1
        assert dropped == len(HITS) - 1

    def test_no_hits_yields_no_blocks(self) -> None:
        assert select_blocks([]) == ((), 0)


class TestRendering:
    def test_a_block_carries_its_marker_provenance_and_text(self) -> None:
        rendered = ContextBlock(marker=2, hit=HITS[0]).render()
        assert rendered.startswith("[2] source=data/raw/a.md")
        assert "title=Doc a" in rendered
        assert "heading=Guide > Section" in rendered
        assert HITS[0].text in rendered

    def test_heading_path_can_be_left_out(self) -> None:
        rendered = ContextBlock(marker=1, hit=HITS[0]).render(include_heading_path=False)
        assert "heading=" not in rendered

    def test_the_prompt_is_system_then_user(self) -> None:
        prompt = build_prompt("what is fusion?", HITS)
        assert [message.role for message in prompt.messages] == ["system", "user"]
        assert prompt.hits_used == 3

    def test_the_question_is_carried_verbatim(self) -> None:
        prompt = build_prompt("  what is fusion?  ", HITS)
        assert prompt.messages[1].content.endswith("Question: what is fusion?")

    def test_an_empty_context_is_stated_rather_than_omitted(self) -> None:
        prompt = build_prompt("anything", [])
        assert "(no context blocks were retrieved)" in prompt.messages[1].content
        assert prompt.hits_used == 0

    def test_the_configured_chunk_ceiling_is_applied(self) -> None:
        config = GenerationConfig(prompt=PromptConfig(max_chunks_in_prompt=1))
        prompt = build_prompt("what is fusion?", HITS, config=config)
        assert prompt.hits_used == 1
        assert prompt.dropped_hits == 2

    def test_an_explicit_system_prompt_wins(self) -> None:
        prompt = build_prompt("q", HITS, system_prompt="be terse")
        assert prompt.messages[0].content == "be terse"

    def test_messages_serialise_to_provider_shape(self) -> None:
        assert ChatMessage(role="user", content="hi").as_dict() == {
            "role": "user",
            "content": "hi",
        }
        assert build_prompt("q", HITS).as_dicts()[0]["role"] == "system"

    def test_the_token_estimate_grows_with_the_context(self) -> None:
        small = build_prompt("q", HITS[:1]).estimated_context_tokens
        large = build_prompt("q", HITS).estimated_context_tokens
        assert 0 < small < large
        assert estimate_tokens("") == 1


class TestParsingBack:
    def test_blocks_round_trip(self) -> None:
        prompt = build_prompt("what is fusion?", HITS)
        parsed = parse_context_blocks(prompt.messages[1].content)
        assert [marker for marker, _ in parsed] == [1, 2, 3]
        assert parsed[0][1] == HITS[0].text

    def test_the_question_never_becomes_evidence(self) -> None:
        prompt = build_prompt("does the reranker read the passage?", HITS)
        parsed = parse_context_blocks(prompt.messages[1].content)
        assert all("does the reranker read the passage?" not in body for _, body in parsed)

    def test_an_empty_context_parses_to_nothing(self) -> None:
        prompt = build_prompt("q", [])
        assert parse_context_blocks(prompt.messages[1].content) == []


class TestSystemPromptFile:
    def test_a_missing_file_degrades_to_the_builtin(self, tmp_path: Path) -> None:
        assert load_system_prompt(tmp_path / "nope.md") == DEFAULT_SYSTEM_PROMPT

    def test_none_skips_the_lookup(self) -> None:
        assert load_system_prompt(None) == DEFAULT_SYSTEM_PROMPT

    def test_a_file_is_read(self, tmp_path: Path) -> None:
        path = tmp_path / "system.md"
        path.write_text("answer only from context", encoding="utf-8")
        assert load_system_prompt(path) == "answer only from context"

    def test_html_comments_are_not_sent_to_the_model(self, tmp_path: Path) -> None:
        path = tmp_path / "system.md"
        path.write_text("rules\n<!--\nnote to the next editor\n-->\n", encoding="utf-8")
        assert load_system_prompt(path) == "rules"

    def test_an_empty_file_degrades_to_the_builtin(self, tmp_path: Path) -> None:
        path = tmp_path / "system.md"
        path.write_text("   \n", encoding="utf-8")
        assert load_system_prompt(path) == DEFAULT_SYSTEM_PROMPT

    def test_the_tracked_prompt_file_agrees_with_the_builtin_sentinel(self) -> None:
        # The guardrails match the sentinel exactly; a prompt file that asks for a
        # different word would produce refusals served as answers.
        tracked = Path("configs/prompts/system.md")
        if not tracked.is_file():  # pragma: no cover - only when run outside the repo
            pytest.skip("configs/prompts/system.md is not present")
        assert ABSTAIN_TOKEN in load_system_prompt(tracked)


class TestFakeLLM:
    def test_it_answers_with_markers_from_the_prompt(self) -> None:
        prompt = build_prompt("what is fusion?", HITS)
        response = FakeLLM().complete(prompt.messages)
        assert "[1]" in response.text
        assert "[2]" in response.text
        assert response.model == FAKE_MODEL
        assert response.finish_reason == "stop"

    def test_it_quotes_only_as_many_blocks_as_configured(self) -> None:
        prompt = build_prompt("q", HITS)
        assert "[2]" not in FakeLLM(max_sentences=1).complete(prompt.messages).text

    def test_it_abstains_with_no_context(self) -> None:
        response = FakeLLM().complete(build_prompt("q", []).messages)
        assert response.text == ABSTAIN_TOKEN
        assert response.finish_reason == "abstain"

    def test_it_abstains_when_every_block_is_blank(self) -> None:
        prompt = build_prompt("q", [make_hit("empty", "   ", 1)])
        assert FakeLLM().complete(prompt.messages).text == ABSTAIN_TOKEN

    def test_it_abstains_when_context_has_no_substantive_query_term(self) -> None:
        prompt = build_prompt("Who won the Antarctic underwater chess championship?", HITS)
        response = FakeLLM().complete(prompt.messages)
        assert response.text == ABSTAIN_TOKEN
        assert response.finish_reason == "abstain"

    def test_it_is_deterministic(self) -> None:
        prompt = build_prompt("what is fusion?", HITS)
        assert FakeLLM().complete(prompt.messages).text == FakeLLM().complete(prompt.messages).text

    def test_the_marker_sits_inside_the_sentence(self) -> None:
        prompt = build_prompt("q", HITS[:1])
        assert FakeLLM().complete(prompt.messages).text.endswith("[1].")


class _StubMessage:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _StubChoice:
    def __init__(self, content: str | None, finish_reason: str = "stop") -> None:
        self.message = _StubMessage(content)
        self.finish_reason = finish_reason


class _StubUsage:
    prompt_tokens = 11
    completion_tokens = 7


class _StubResponse:
    def __init__(self) -> None:
        self.choices: list[Any] = []
        self.model = "gpt-4o-mini"
        self.usage = _StubUsage()


class StubOpenAIClient:
    """Stands in for ``openai.OpenAI``: records the call, returns a canned reply."""

    def __init__(self, *, content: str | None = "grounded [1].", choices: Any = None) -> None:
        self._content = content
        self._choices = choices
        self.calls: list[dict[str, Any]] = []
        self.chat = self

    @property
    def completions(self) -> StubOpenAIClient:
        return self

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        response = _StubResponse()
        response.choices = (
            self._choices if self._choices is not None else [_StubChoice(self._content)]
        )
        return response


class ExplodingOpenAIClient:
    def __init__(self) -> None:
        self.chat = self

    @property
    def completions(self) -> ExplodingOpenAIClient:
        return self

    def create(self, **kwargs: Any) -> Any:
        raise RuntimeError("upstream is down")


class TestOpenAILLM:
    def test_a_missing_key_is_a_usage_error_not_a_request_failure(self) -> None:
        with pytest.raises(LLMError, match="OPENAI_API_KEY"):
            OpenAILLM(api_key="")

    def test_it_sends_the_configured_model_and_temperature(self) -> None:
        client = StubOpenAIClient()
        llm = OpenAILLM(
            api_key="",
            model="gpt-4o-mini",
            temperature=0.1,
            max_output_tokens=256,
            client=client,
        )
        response = llm.complete(build_prompt("q", HITS).messages)
        assert response.text == "grounded [1]."
        assert response.prompt_tokens == 11
        assert client.calls[0]["model"] == "gpt-4o-mini"
        assert client.calls[0]["temperature"] == 0.1
        assert client.calls[0]["max_tokens"] == 256
        assert [message["role"] for message in client.calls[0]["messages"]] == ["system", "user"]

    def test_a_provider_failure_becomes_an_llm_error(self) -> None:
        llm = OpenAILLM(api_key="", client=ExplodingOpenAIClient())
        with pytest.raises(LLMError, match="generation request failed"):
            llm.complete(build_prompt("q", HITS).messages)

    def test_no_choices_is_an_error_not_an_empty_answer(self) -> None:
        llm = OpenAILLM(api_key="", client=StubOpenAIClient(choices=[]))
        with pytest.raises(LLMError, match="no choices"):
            llm.complete(build_prompt("q", HITS).messages)

    def test_a_message_without_content_is_an_error(self) -> None:
        llm = OpenAILLM(api_key="", client=StubOpenAIClient(content=None))
        with pytest.raises(LLMError, match="no content"):
            llm.complete(build_prompt("q", HITS).messages)

    def test_the_credential_never_appears_in_an_error(self) -> None:
        llm = OpenAILLM(api_key="test-credential-must-not-leak", client=ExplodingOpenAIClient())
        with pytest.raises(LLMError) as excinfo:
            llm.complete(build_prompt("q", HITS).messages)
        assert "test-credential-must-not-leak" not in str(excinfo.value)


class TestBuildLLM:
    def test_the_default_is_the_offline_double(self) -> None:
        assert isinstance(build_llm(), FakeLLM)

    def test_an_unknown_kind_names_the_valid_ones(self) -> None:
        with pytest.raises(LLMError, match="unknown llm"):
            build_llm("gemini")
        assert LLM_KINDS == ("fake", "openai")

    def test_openai_without_a_key_is_refused(self) -> None:
        with pytest.raises(LLMError, match="OPENAI_API_KEY"):
            build_llm("openai")

    def test_openai_reads_the_model_from_the_config(self) -> None:
        llm = build_llm("openai", config=GenerationConfig(model="gpt-4o"), api_key="k")
        assert llm.model == "gpt-4o"
