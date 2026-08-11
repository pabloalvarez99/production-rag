"""Unit tests for citation extraction: markers in, verifiable sources out."""

from __future__ import annotations

from production_rag.generation.citations import extract_citations
from production_rag.generation.prompts import ContextBlock, build_prompt
from production_rag.retrieval.hybrid import RetrievalHit


def make_hit(chunk_id: str, text: str, rank: int) -> RetrievalHit:
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


BLOCKS = (
    ContextBlock(marker=1, hit=make_hit("a", "Qdrant holds both vector kinds.", 1)),
    ContextBlock(marker=2, hit=make_hit("b", "RRF merges lists by position.", 2)),
    ContextBlock(marker=3, hit=make_hit("c", "A cross-encoder reads the pair.", 3)),
)


class TestMapping:
    def test_a_marker_resolves_to_the_chunk_it_points_at(self) -> None:
        result = extract_citations("Qdrant holds both kinds [1].", BLOCKS)
        assert [citation.marker for citation in result.citations] == [1]
        citation = result.citations[0]
        assert citation.chunk_id == "a"
        assert citation.source_path == "data/raw/a.md"
        assert citation.title == "Doc a"
        assert citation.heading_path == "Guide > Section"
        assert citation.text == "Qdrant holds both vector kinds."

    def test_citations_are_ordered_by_first_appearance(self) -> None:
        result = extract_citations("Second [2]. First [1].", BLOCKS)
        assert [citation.marker for citation in result.citations] == [2, 1]

    def test_a_repeated_marker_is_cited_once(self) -> None:
        result = extract_citations("One [1]. Again [1].", BLOCKS)
        assert [citation.marker for citation in result.citations] == [1]
        assert result.answer.count("[1]") == 2

    def test_a_grouped_marker_is_split(self) -> None:
        result = extract_citations("Both agree [1, 2].", BLOCKS)
        assert [citation.marker for citation in result.citations] == [1, 2]
        assert "[1][2]" in result.answer

    def test_an_answer_without_markers_cites_nothing(self) -> None:
        result = extract_citations("A confident, unsourced claim.", BLOCKS)
        assert result.citations == ()
        assert result.has_citations is False
        assert result.answer == "A confident, unsourced claim."


class TestInvalidMarkers:
    def test_a_marker_past_the_last_block_is_removed_and_reported(self) -> None:
        result = extract_citations("Invented support [7].", BLOCKS)
        assert result.invalid_markers == (7,)
        assert "[7]" not in result.answer
        assert result.answer == "Invented support."

    def test_a_mixed_group_keeps_the_valid_half(self) -> None:
        result = extract_citations("Partly grounded [2, 9].", BLOCKS)
        assert [citation.marker for citation in result.citations] == [2]
        assert result.invalid_markers == (9,)
        assert result.answer == "Partly grounded [2]."

    def test_marker_zero_is_invalid_because_blocks_are_one_based(self) -> None:
        result = extract_citations("Zero [0].", BLOCKS)
        assert result.citations == ()
        assert result.invalid_markers == (0,)

    def test_an_invalid_marker_is_reported_once(self) -> None:
        result = extract_citations("Here [9]. And here [9].", BLOCKS)
        assert result.invalid_markers == (9,)

    def test_no_blocks_means_every_marker_is_invalid(self) -> None:
        result = extract_citations("Grounded in nothing [1].", ())
        assert result.citations == ()
        assert result.invalid_markers == (1,)


class TestTextHygiene:
    def test_removing_a_marker_leaves_no_double_space(self) -> None:
        result = extract_citations("Before [9] after.", BLOCKS)
        assert result.answer == "Before after."

    def test_markers_are_not_renumbered(self) -> None:
        # Renumbering would stop the answer matching the prompt that produced it,
        # and lining those two up is where every debugging session starts.
        result = extract_citations("Only the third block says so [3].", BLOCKS)
        assert result.answer.endswith("[3].")
        assert result.citations[0].marker == 3


class TestSerialisation:
    def test_a_citation_carries_its_provenance(self) -> None:
        payload = extract_citations("Grounded [1].", BLOCKS).citations[0].to_dict()
        assert payload["marker"] == 1
        assert payload["chunk_id"] == "a"
        assert payload["source_path"] == "data/raw/a.md"
        assert payload["text"] == "Qdrant holds both vector kinds."

    def test_the_quoted_text_can_be_dropped(self) -> None:
        payload = (
            extract_citations("Grounded [1].", BLOCKS).citations[0].to_dict(include_text=False)
        )
        assert "text" not in payload


class TestAgreementWithThePrompt:
    def test_markers_map_against_the_prompt_not_the_hit_list(self) -> None:
        # Truncation is exactly the case where the two disagree: block 2 is the
        # second chunk *shown*, and a citation must resolve to that one.
        hits = [make_hit(name, f"Text about {name}.", rank) for rank, name in enumerate("abcd", 1)]
        prompt = build_prompt("q", hits, system_prompt="s")
        blocks = prompt.blocks
        result = extract_citations("Answer [2].", blocks)
        assert result.citations[0].chunk_id == blocks[1].hit.chunk_id
