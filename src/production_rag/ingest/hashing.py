"""Content hashing and the three identifiers a chunk carries.

Three ids, because they answer three different questions:

``doc_id``
    *Which document is this?* Derived from the path relative to the corpus root,
    so it is stable across re-ingests and readable in a log line. Renaming a
    file changes its ``doc_id`` — that is the documented trade-off (see
    ``data/raw/README.md``): a rename orphans the old chunks until a full
    re-ingest.

``chunk_id``
    *Which chunk of that document?* ``<doc_id>:<index>`` with a zero-padded
    index. This is the id that appears in citations and in the golden evaluation
    set (``data/eval/README.md``), so it has to be human-readable and sortable.

``point_id``
    *Which row in Qdrant?* Qdrant only accepts an unsigned integer or a UUID as
    a point id, so the citable ``chunk_id`` cannot be used directly. A UUID5
    over ``source_path``, chunk index and content hash makes re-ingesting
    unchanged content an idempotent upsert instead of a duplicate, and makes
    edited content land on a new point.

All hashing goes through :func:`sha256_text`. Python's built-in ``hash()`` is
salted per process and would produce different ids on every run.
"""

from __future__ import annotations

import hashlib
import unicodedata
from uuid import NAMESPACE_URL, UUID, uuid5

DOC_ID_LENGTH = 16
"""Hex characters kept from the document digest.

64 bits of a SHA-256: collision-free for any corpus that fits on a disk, and
short enough to read in a log line or a citation.
"""

CHUNK_INDEX_WIDTH = 4
"""Zero-padding for the chunk index, so ``chunk_id`` sorts lexicographically."""


def sha256_text(text: str) -> str:
    """Return the SHA-256 hex digest of *text*.

    The text is NFC-normalised first: on macOS a filesystem can hand back a
    decomposed form of the same characters, and two byte sequences that render
    identically must not produce two different chunks.
    """
    normalised = unicodedata.normalize("NFC", text)
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def normalise_source_path(source_path: str) -> str:
    """Return *source_path* in the canonical form used for ids and payloads.

    Backslashes become forward slashes and any ``./`` prefix is dropped, so the
    same document ingested on Windows and in the Linux container gets the same
    ``doc_id``.
    """
    unified = source_path.replace("\\", "/").lstrip("/")
    while unified.startswith("./"):
        unified = unified[2:]
    return unified


def doc_id_for(source_path: str) -> str:
    """Return the stable document id for a corpus-relative path."""
    return sha256_text(normalise_source_path(source_path))[:DOC_ID_LENGTH]


def chunk_id_for(doc_id: str, chunk_index: int) -> str:
    """Return the citable chunk id, e.g. ``9f2c1a7b3d4e5f60:0003``."""
    if chunk_index < 0:
        raise ValueError(f"chunk_index must be non-negative, got {chunk_index}")
    return f"{doc_id}:{chunk_index:0{CHUNK_INDEX_WIDTH}d}"


def point_uuid_for(source_path: str, chunk_index: int, content_sha256: str) -> UUID:
    """Return the Qdrant point id for a chunk.

    Content-addressed on purpose: re-running ingest over an unchanged corpus
    upserts the same points (a no-op), while an edited paragraph produces a new
    point id, so a stale vector cannot survive under a fresh payload.
    """
    if chunk_index < 0:
        raise ValueError(f"chunk_index must be non-negative, got {chunk_index}")
    name = f"{normalise_source_path(source_path)}::{chunk_index}::{content_sha256}"
    return uuid5(NAMESPACE_URL, name)


def point_id_for(source_path: str, chunk_index: int, content_sha256: str) -> str:
    """Return :func:`point_uuid_for` as the string Qdrant is given."""
    return str(point_uuid_for(source_path, chunk_index, content_sha256))


def estimate_tokens(text: str) -> int:
    """Rough token count for a chunk: four characters per token.

    A deliberate estimate, not a measurement. Ingest has no reason to load a
    tokeniser — the number is used for budgeting and for spotting a chunker
    regression, and the exact count only matters at generation time.
    """
    return max(1, (len(text) + 3) // 4) if text else 0
