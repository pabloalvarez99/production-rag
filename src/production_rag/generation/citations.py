"""Citation extraction: turning ``[2]`` in an answer into a verifiable source.

    answer text + the blocks that were in the prompt -> Citation[]

A citation is only worth anything if it resolves to a chunk the model was
actually shown. So the mapping is done against
:class:`~production_rag.generation.prompts.ContextBlock`, the numbered blocks in
the rendered prompt — never against the retrieval result, which may be longer
after truncation and would silently shift every marker by however many chunks
did not fit.

**Invalid markers are removed, not reported and kept.** A model that cites ``[7]``
when six blocks were shown has cited nothing; leaving the marker in the text
would show a user a footnote that goes nowhere, which reads as more grounded
than an uncited sentence rather than less. The count of dropped markers is
carried in the result, because it is a real quality signal about the model and
the prompt.

Renumbering is deliberately *not* done. If an answer cites only ``[3]``, the
citation list carries marker 3 and the text still says ``[3]``. Compacting to
``[1]`` would make the answer no longer match the prompt that produced it, and
every debugging session on this path starts by lining those two up.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import structlog

from production_rag.generation.prompts import ContextBlock

_log = structlog.get_logger(__name__)

_MARKER_GROUP = re.compile(r"\[\s*(\d+(?:\s*[,;]\s*\d+)*)\s*\]")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:!?])")
_REPEATED_SPACE = re.compile(r"[ \t]{2,}")


@dataclass(frozen=True, slots=True)
class Citation:
    """One resolved marker: the number, and the chunk it points at.

    Flat and JSON-safe, like :class:`~production_rag.retrieval.hybrid.RetrievalHit`:
    this is what an HTTP response renders and what a citation-precision eval
    scores, so it carries provenance (``source_path``, ``chunk_id``, heading
    ancestry) rather than a bare index into a list the caller may not have.
    """

    marker: int
    chunk_id: str
    source_path: str
    text: str
    score: float
    rank: int
    title: str | None = None
    heading_path: str | None = None
    point_id: str = ""

    def to_dict(self, *, include_text: bool = True) -> dict[str, Any]:
        """JSON-serialisable form.

        Args:
            include_text: Whether to carry the quoted chunk. A citation without
                its text is unverifiable by the person reading the answer, so it
                is included by default; a caller with its own copy of the hits
                can drop it to keep a payload small.
        """
        payload: dict[str, Any] = {
            "marker": self.marker,
            "chunk_id": self.chunk_id,
            "source_path": self.source_path,
            "title": self.title,
            "heading_path": self.heading_path,
            "point_id": self.point_id,
            "rank": self.rank,
            "score": round(self.score, 6),
        }
        if include_text:
            payload["text"] = self.text
        return payload


@dataclass(frozen=True, slots=True)
class CitationResult:
    """The cleaned answer and the citations it supports.

    ``answer`` is the text with unresolvable markers removed, so what a caller
    displays and what :attr:`citations` contains cannot disagree.
    """

    answer: str
    citations: tuple[Citation, ...]
    invalid_markers: tuple[int, ...] = ()

    @property
    def has_citations(self) -> bool:
        """Whether anything in the answer resolved to a retrieved chunk."""
        return bool(self.citations)


def extract_citations(answer: str, blocks: Sequence[ContextBlock]) -> CitationResult:
    """Resolve the bracketed markers in *answer* against the prompt's *blocks*.

    Args:
        answer: The model's raw text.
        blocks: The numbered blocks that were in the prompt, in prompt order.

    Returns:
        The answer with unresolvable markers stripped, the citations in order of
        first appearance in the text, and the markers that resolved to nothing.

    Note:
        Grouped markers (``[1, 2]``) are accepted and split. A group in which
        only some numbers are valid is rewritten to the valid ones rather than
        dropped whole, because the sentence really is supported by the surviving
        block.
    """
    by_marker = {block.marker: block for block in blocks}
    seen: list[int] = []
    invalid: list[int] = []

    def _replace(match: re.Match[str]) -> str:
        numbers = [int(part) for part in re.split(r"[,;]", match.group(1))]
        kept: list[int] = []
        for number in numbers:
            if number in by_marker:
                kept.append(number)
                if number not in seen:
                    seen.append(number)
            elif number not in invalid:
                invalid.append(number)
        return "".join(f"[{number}]" for number in kept)

    cleaned = _tidy(_MARKER_GROUP.sub(_replace, answer))
    if invalid:
        _log.warning(
            "citation_markers_dropped",
            invalid=invalid,
            blocks=len(by_marker),
        )
    citations = tuple(_to_citation(by_marker[marker]) for marker in seen)
    return CitationResult(
        answer=cleaned,
        citations=citations,
        invalid_markers=tuple(invalid),
    )


def _to_citation(block: ContextBlock) -> Citation:
    """Project a prompt block onto the citation a caller receives."""
    hit = block.hit
    return Citation(
        marker=block.marker,
        chunk_id=hit.chunk_id,
        source_path=hit.source_path,
        text=hit.text,
        score=hit.score,
        rank=hit.rank,
        title=hit.title,
        heading_path=hit.heading_path,
        point_id=hit.point_id,
    )


def _tidy(text: str) -> str:
    """Clean up the whitespace a removed marker leaves behind."""
    without_gaps = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    return _REPEATED_SPACE.sub(" ", without_gaps).strip()
