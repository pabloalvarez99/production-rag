"""Identity rules for documents, chunks and points.

These functions decide whether re-ingesting a corpus is idempotent or whether it
silently doubles the collection, so they are tested as a contract rather than as
implementation detail.
"""

from __future__ import annotations

import subprocess
import sys
from uuid import UUID

import pytest

from production_rag.ingest.hashing import (
    CHUNK_INDEX_WIDTH,
    DOC_ID_LENGTH,
    chunk_id_for,
    doc_id_for,
    estimate_tokens,
    normalise_source_path,
    point_id_for,
    point_uuid_for,
    sha256_text,
)


def test_sha256_is_stable_and_hex() -> None:
    digest = sha256_text("hello world")
    assert digest == sha256_text("hello world")
    assert len(digest) == 64
    assert int(digest, 16) >= 0


def test_sha256_normalises_unicode_composition() -> None:
    # "é" as one code point vs "e" + combining acute: byte-different, identical
    # text. Without NFC these produce two point ids for the same paragraph, so
    # every re-ingest of a file edited on macOS would duplicate it.
    assert sha256_text("café") == sha256_text("café")


def test_sha256_distinguishes_different_text() -> None:
    assert sha256_text("a") != sha256_text("b")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("docs\\guide.md", "docs/guide.md"),
        ("/docs/guide.md", "docs/guide.md"),
        ("./docs/guide.md", "docs/guide.md"),
        ("docs/guide.md", "docs/guide.md"),
    ],
)
def test_normalise_source_path(raw: str, expected: str) -> None:
    assert normalise_source_path(raw) == expected


def test_doc_id_is_path_shaped_not_platform_shaped() -> None:
    # The same file ingested on Windows and in the Linux container must land on
    # the same doc_id, or the golden eval set only matches on one of them.
    assert doc_id_for("docs\\guide.md") == doc_id_for("docs/guide.md")
    assert len(doc_id_for("docs/guide.md")) == DOC_ID_LENGTH


def test_doc_id_differs_per_path() -> None:
    assert doc_id_for("a.md") != doc_id_for("b.md")


def test_chunk_id_is_zero_padded_and_citable() -> None:
    # data/eval/golden.jsonl references chunk ids in exactly this shape.
    assert chunk_id_for("9f2c1a7b3d4e5f60", 3) == "9f2c1a7b3d4e5f60:0003"
    assert chunk_id_for("9f2c1a7b3d4e5f60", 12345) == "9f2c1a7b3d4e5f60:12345"


def test_chunk_id_sorts_lexicographically_within_a_document() -> None:
    doc = doc_id_for("docs/guide.md")
    ids = [chunk_id_for(doc, index) for index in (0, 2, 10)]
    assert ids == sorted(ids), f"zero padding must be at least {CHUNK_INDEX_WIDTH} wide"


def test_point_id_is_a_uuid() -> None:
    point_id = point_id_for("docs/guide.md", 0, sha256_text("body"))
    assert str(UUID(point_id)) == point_id


def test_point_id_is_content_addressed() -> None:
    unchanged = point_id_for("docs/guide.md", 0, sha256_text("body"))
    assert unchanged == point_id_for("docs/guide.md", 0, sha256_text("body"))
    # Any of the three inputs changing must move the point.
    assert unchanged != point_id_for("docs/other.md", 0, sha256_text("body"))
    assert unchanged != point_id_for("docs/guide.md", 1, sha256_text("body"))
    assert unchanged != point_id_for("docs/guide.md", 0, sha256_text("edited"))


def test_point_uuid_matches_point_id() -> None:
    assert str(point_uuid_for("a.md", 0, sha256_text("x"))) == point_id_for(
        "a.md", 0, sha256_text("x")
    )


def test_point_id_is_stable_across_processes() -> None:
    # The regression this guards: str.__hash__ is salted per interpreter, so an
    # id built on hash() would be different in every process and every re-ingest
    # would write a second copy of the whole corpus.
    code = (
        "from production_rag.ingest.hashing import point_id_for, sha256_text;"
        "print(point_id_for('docs/guide.md', 7, sha256_text('body')))"
    )
    first = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert first == point_id_for("docs/guide.md", 7, sha256_text("body"))


@pytest.mark.parametrize(
    ("text", "expected"), [("", 0), ("a", 1), ("abcd", 1), ("abcde", 2), ("x" * 800, 200)]
)
def test_estimate_tokens_is_four_characters_per_token(text: str, expected: int) -> None:
    # An estimate, not a measurement: no tokeniser is loaded during ingest. Only
    # empty text scores zero, and no chunk is ever empty.
    assert estimate_tokens(text) == expected
