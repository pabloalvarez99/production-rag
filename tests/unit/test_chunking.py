"""Chunking behaviour: sizes, overlap, heading segmentation, embed text.

Pure functions with no IO, so every case here is a direct statement about what
the chunker guarantees.
"""

from __future__ import annotations

from production_rag.config_loader import ChunkingConfig
from production_rag.ingest.chunking import (
    build_embed_text,
    chunk_document,
    split_into_sections,
    split_text,
)
from production_rag.ingest.models import Document

CONFIG = ChunkingConfig()


def _paragraph(word: str, count: int) -> str:
    return " ".join([word] * count)


def test_short_text_is_one_chunk() -> None:
    assert split_text("A single short sentence.", CONFIG) == ["A single short sentence."]


def test_blank_text_yields_nothing() -> None:
    assert split_text("   \n\n  ", CONFIG) == []


def test_long_text_is_split_within_the_size_limit() -> None:
    text = "\n\n".join(_paragraph("alpha", 40) for _ in range(12))
    chunks = split_text(text, CONFIG)
    assert len(chunks) > 1
    assert all(len(chunk) <= CONFIG.chunk_size for chunk in chunks)


def test_split_preserves_every_word_in_order() -> None:
    # Overlap means text repeats, but nothing may be lost: a dropped sentence is
    # a document that can never be retrieved and nothing reports it.
    text = "\n\n".join(
        f"sentence number {index} " + _paragraph("filler", 30) for index in range(15)
    )
    chunks = split_text(text, CONFIG)
    joined = " ".join(chunks)
    for index in range(15):
        assert f"sentence number {index}" in joined


def test_overlap_repeats_the_tail_of_the_previous_chunk() -> None:
    config = ChunkingConfig(chunk_size=300, chunk_overlap=100, min_chunk_chars=0)
    text = "\n\n".join(_paragraph(f"w{index}", 20) for index in range(12))
    chunks = split_text(text, config)
    assert len(chunks) >= 2
    shared = [
        index
        for index in range(1, len(chunks))
        if set(chunks[index].split()) & set(chunks[index - 1].split())
    ]
    assert shared, "consecutive chunks should share overlapping content"


def test_zero_overlap_produces_disjoint_chunks() -> None:
    config = ChunkingConfig(chunk_size=200, chunk_overlap=0, min_chunk_chars=0)
    text = "\n\n".join(f"para{index} " + _paragraph("x", 30) for index in range(8))
    chunks = split_text(text, config)
    markers = [f"para{index}" for index in range(8)]
    for marker in markers:
        assert sum(marker in chunk for chunk in chunks) == 1


def test_unbreakable_run_is_hard_cut_rather_than_dropped() -> None:
    config = ChunkingConfig(chunk_size=50, chunk_overlap=0, min_chunk_chars=0)
    chunks = split_text("x" * 200, config)
    assert chunks
    assert "".join(chunks) == "x" * 200


def test_sections_follow_headings() -> None:
    sections = split_into_sections(
        "intro text\n\n## First\nbody one\n\n### Nested\nbody two\n\n## Second\nbody three\n"
    )
    headings = [section.heading for section in sections]
    assert headings == [None, "First", "Nested", "Second"]
    assert sections[2].heading_path == ("First", "Nested")
    assert "body two" in sections[2].body
    assert "body three" not in sections[2].body


def test_headings_inside_a_fenced_block_are_not_headings() -> None:
    # A shell prompt or a Markdown example inside a code fence would otherwise
    # split the document at a line that is not structure at all.
    sections = split_into_sections("## Real\ntext\n\n```\n## Not a heading\n```\nmore text\n")
    assert [section.heading for section in sections] == ["Real"]
    assert "## Not a heading" in sections[0].body


def test_chunk_document_numbers_chunks_across_sections() -> None:
    document = Document(
        source_path="guide/intro.md",
        text="\n\n".join(f"## Section {index}\n" + _paragraph("word", 200) for index in range(3)),
        title="Guide",
        source="guide",
        tags=("a",),
    )
    chunks, dropped = chunk_document(document, CONFIG)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert dropped == 0
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    assert len({chunk.point_id for chunk in chunks}) == len(chunks)
    assert {chunk.doc_id for chunk in chunks} == {document.doc_id}


def test_chunk_document_reports_dropped_short_fragments() -> None:
    document = Document(
        source_path="notes.md",
        text="## A\ntiny\n\n## B\nalso tiny\n",
        title="Notes",
    )
    chunks, dropped = chunk_document(document, ChunkingConfig(min_chunk_chars=120))
    assert chunks == []
    assert dropped == 2


def test_chunk_carries_its_heading_ancestry() -> None:
    document = Document(
        source_path="guide/hybrid.md",
        text="## Hybrid search\n### Configuration\n" + _paragraph("detail", 60),
        title="Hybrid",
    )
    chunks, _ = chunk_document(document, CONFIG)
    assert chunks[0].heading == "Configuration"
    assert chunks[0].heading_path == "Hybrid search > Configuration"


def test_content_hash_matches_the_chunk_text() -> None:
    document = Document(source_path="a.md", text=_paragraph("token", 100), title="A")
    chunks, _ = chunk_document(document, CONFIG)
    from production_rag.ingest.hashing import sha256_text

    assert all(chunk.content_sha256 == sha256_text(chunk.text) for chunk in chunks)


def test_embed_text_prefixes_context_but_payload_text_does_not() -> None:
    document = Document(
        source_path="guide/hybrid.md",
        text="## Hybrid search\n" + _paragraph("detail", 60),
        title="Hybrid",
    )
    chunks, _ = chunk_document(document, CONFIG)
    chunk = chunks[0]
    assert chunk.embed_text.startswith("Hybrid > Hybrid search\n\n")
    # The prefix must never reach a citation.
    assert not chunk.text.startswith("Hybrid >")
    assert (
        "Hybrid > Hybrid search"
        not in chunk.to_payload(ingest_run_id="r", embedded_model="m")["text"]
    )


def test_embed_text_is_untouched_when_context_is_disabled() -> None:
    assert (
        build_embed_text("body", title="T", heading_path=("H",), prepend_heading_context=False)
        == "body"
    )


def test_embed_text_does_not_repeat_a_heading_equal_to_the_title() -> None:
    assert (
        build_embed_text(
            "body", title="Runbook", heading_path=("Runbook",), prepend_heading_context=True
        )
        == "Runbook\n\nbody"
    )


def test_chunking_is_deterministic() -> None:
    document = Document(source_path="a.md", text=_paragraph("word", 500), title="A")
    first, _ = chunk_document(document, CONFIG)
    second, _ = chunk_document(document, CONFIG)
    assert [chunk.point_id for chunk in first] == [chunk.point_id for chunk in second]
