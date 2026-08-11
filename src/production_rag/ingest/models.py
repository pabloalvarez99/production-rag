"""The two data structures the ingest pipeline moves: documents and chunks.

Both are frozen Pydantic models. Frozen because a chunk's identity is derived
from its content — mutating the text after the id was computed would produce a
payload that lies about itself. Pydantic rather than a dataclass so the Qdrant
payload is a validated ``model_dump`` instead of a hand-rolled dict.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from production_rag.ingest.hashing import (
    chunk_id_for,
    doc_id_for,
    estimate_tokens,
    normalise_source_path,
    point_id_for,
    sha256_text,
)

ROOT_SOURCE = "root"
"""``source`` value for a document sitting directly in the corpus root.

``source`` is the first path segment under the corpus root and is keyword
indexed for filtering. A file with no enclosing folder has no segment to use, so
it gets an explicit sentinel rather than an empty string, which reads like a bug
in a filter expression.
"""


class Document(BaseModel):
    """One file from the corpus, already decoded and stripped of front matter."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_path: str = Field(description="Path relative to the corpus root, forward slashes.")
    text: str = Field(description="Body, with any YAML front matter removed.")
    title: str = Field(description="Front matter title, else the first H1, else the file stem.")
    source: str = Field(
        default=ROOT_SOURCE,
        description="First path segment under the corpus root; filterable.",
    )
    tags: tuple[str, ...] = Field(default=(), description="Front matter tags, if any.")

    @property
    def doc_id(self) -> str:
        """Stable id derived from :attr:`source_path`."""
        return doc_id_for(self.source_path)


class Chunk(BaseModel):
    """An embeddable slice of a document, carrying its own provenance.

    ``text`` is what gets cited and what a reader sees. ``embed_text`` is what
    the embedding model is given, and may carry a title and heading prefix when
    ``ingest.chunking.prepend_heading_context`` is on: retrieval quality
    improves on short queries, but the prefix must never leak into a citation,
    so the two are separate fields and only ``text`` reaches the payload.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    doc_id: str
    chunk_id: str
    point_id: str
    chunk_index: int = Field(ge=0)
    text: str
    embed_text: str
    content_sha256: str
    source_path: str
    source: str = ROOT_SOURCE
    title: str | None = None
    heading: str | None = Field(default=None, description="Nearest enclosing heading.")
    heading_path: str | None = Field(
        default=None, description="Heading ancestry joined with ' > ', for citations."
    )
    token_count_est: int = Field(ge=0)
    tags: tuple[str, ...] = ()

    @classmethod
    def build(
        cls,
        *,
        document: Document,
        chunk_index: int,
        text: str,
        embed_text: str,
        heading: str | None = None,
        heading_path: tuple[str, ...] = (),
    ) -> Chunk:
        """Create a chunk with every derived field computed from its content.

        This is the only sanctioned constructor: it is what guarantees that
        ``content_sha256``, ``chunk_id`` and ``point_id`` agree with ``text``.
        """
        source_path = normalise_source_path(document.source_path)
        content_sha256 = sha256_text(text)
        doc_id = doc_id_for(source_path)
        return cls(
            doc_id=doc_id,
            chunk_id=chunk_id_for(doc_id, chunk_index),
            point_id=point_id_for(source_path, chunk_index, content_sha256),
            chunk_index=chunk_index,
            text=text,
            embed_text=embed_text,
            content_sha256=content_sha256,
            source_path=source_path,
            source=document.source,
            title=document.title,
            heading=heading,
            heading_path=" > ".join(heading_path) if heading_path else None,
            token_count_est=estimate_tokens(text),
            tags=document.tags,
        )

    def to_payload(self, *, ingest_run_id: str, embedded_model: str) -> dict[str, Any]:
        """Return the Qdrant payload for this chunk.

        ``embed_text`` is deliberately excluded: it is derivable from ``text``
        plus the heading fields, and storing it would double payload size for
        every chunk in the collection.

        ``ingest_run_id`` and ``embedded_model`` are stamped on every point so a
        collection can be read back and audited — "which run wrote this, and
        with which model?" is the first question when retrieval quality moves
        and nobody remembers what was re-ingested.
        """
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source_path": self.source_path,
            "source": self.source,
            "title": self.title,
            "heading": self.heading,
            "heading_path": self.heading_path,
            "chunk_index": self.chunk_index,
            "content_sha256": self.content_sha256,
            "doc_id": self.doc_id,
            "token_count_est": self.token_count_est,
            "tags": list(self.tags),
            "ingest_run_id": ingest_run_id,
            "embedded_model": embedded_model,
        }
