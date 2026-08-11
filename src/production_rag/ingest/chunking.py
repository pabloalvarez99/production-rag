r"""Recursive chunking: structure first, characters as the last resort.

The strategy in one line: **split on the most meaningful boundary that works,
and only cut mid-sentence when nothing else is left.** Concretely, a document is
first segmented by its Markdown headings, then each section body is split on
progressively weaker separators (blank line, line, sentence, space), and the
resulting pieces are merged greedily up to ``chunk_size`` with a trailing
overlap carried into the next chunk.

Two decisions worth stating because they are invisible in the output:

* **Headings are a segmentation layer, not just a separator.** Splitting on
  ``"\n## "`` inside a flat splitter loses track of *which* heading a chunk fell
  under, and that heading is the cheapest retrieval signal a Markdown corpus
  has. Segmenting first means every chunk knows its heading ancestry exactly.
* **Overlap never crosses a heading boundary.** Carrying the tail of one section
  into the next would attribute one section's sentences to another section's
  heading, which is a citation that points at the wrong place.

Everything in this module is a pure function of its inputs. No IO, no logging,
no configuration lookups — which is what makes the chunker cheap to test at the
character level.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from production_rag.config_loader import ChunkingConfig
from production_rag.ingest.models import Chunk, Document

_HEADING_RE = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<title>.*?)\s*#*\s*$")
"""ATX Markdown heading. Trailing hashes (``## Title ##``) are closing markers."""

_FENCE_RE = re.compile(r"^\s{0,3}(```|~~~)")
"""Start or end of a fenced code block. A ``#`` comment inside is not a heading."""


@dataclass(frozen=True, slots=True)
class Section:
    """A run of body text under one heading, with its full heading ancestry."""

    heading: str | None
    heading_path: tuple[str, ...]
    body: str


def split_into_sections(text: str) -> list[Section]:
    """Segment Markdown into sections by ATX heading, preserving ancestry.

    Text appearing before the first heading becomes a leading section with no
    heading, so a document with no headings at all still yields one section.
    Headings inside fenced code blocks are ignored — a shell comment is not a
    document structure.
    """
    sections: list[Section] = []
    ancestry: dict[int, str] = {}
    current_heading: str | None = None
    current_path: tuple[str, ...] = ()
    buffer: list[str] = []
    in_fence = False

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append(Section(heading=current_heading, heading_path=current_path, body=body))
        buffer.clear()

    for line in text.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            buffer.append(line)
            continue

        match = None if in_fence else _HEADING_RE.match(line)
        if match is None:
            buffer.append(line)
            continue

        flush()
        level = len(match.group("hashes"))
        heading = match.group("title").strip()
        # Drop any recorded heading at this level or deeper: a new ## ends the
        # previous ## and everything nested under it.
        ancestry = {lvl: name for lvl, name in ancestry.items() if lvl < level}
        ancestry[level] = heading
        current_heading = heading
        current_path = tuple(ancestry[lvl] for lvl in sorted(ancestry))

    flush()
    return sections


def _is_leading_separator(separator: str) -> bool:
    r"""Whether *separator* belongs to the text that follows it.

    ``"\n## "`` introduces what comes next, so it is kept at the front of the
    following piece. ``"\n\n"`` and ``". "`` terminate what came before and stay
    at the end of the preceding piece. Getting this wrong is how a splitter
    starts every chunk with a stray blank line.
    """
    return separator.startswith("\n") and separator.strip() != ""


def _split_keep(text: str, separator: str) -> list[str]:
    """Split on *separator* without losing a single character.

    ``"".join(_split_keep(text, sep)) == text`` always holds; that invariant is
    what lets the merge step below reconstruct readable prose.
    """
    pieces = text.split(separator)
    if len(pieces) == 1:
        return pieces
    if _is_leading_separator(separator):
        rebuilt = [pieces[0], *(separator + piece for piece in pieces[1:])]
    else:
        rebuilt = [*(piece + separator for piece in pieces[:-1]), pieces[-1]]
    return [piece for piece in rebuilt if piece]


def _hard_cut(text: str, limit: int) -> list[str]:
    """Last resort: fixed-width slices when no separator occurs in *text*."""
    return [text[start : start + limit] for start in range(0, len(text), limit)]


def _split_recursive(text: str, separators: tuple[str, ...], limit: int) -> list[str]:
    """Break *text* into pieces of at most *limit* characters.

    Each separator is tried in order; a piece still too long is re-split with the
    remaining, weaker separators. Concatenating the result reproduces *text*.
    """
    if len(text) <= limit:
        return [text] if text else []
    if not separators:
        return _hard_cut(text, limit)

    separator, *rest = separators
    remaining = tuple(rest)
    if separator not in text:
        return _split_recursive(text, remaining, limit)

    pieces: list[str] = []
    for piece in _split_keep(text, separator):
        if len(piece) <= limit:
            pieces.append(piece)
        else:
            pieces.extend(_split_recursive(piece, remaining, limit))
    return pieces


def _overlap_tail(pieces: list[str], overlap: int) -> list[str]:
    """Return the trailing pieces of an emitted chunk, up to *overlap* chars.

    When the final piece alone is longer than the overlap budget, its last
    *overlap* characters are used. Overlap exists so a sentence straddling a
    boundary is retrievable from both sides; it is not provenance, so cutting it
    mid-word costs nothing.
    """
    if overlap <= 0 or not pieces:
        return []
    tail: list[str] = []
    total = 0
    for piece in reversed(pieces):
        if total >= overlap:
            break
        tail.insert(0, piece)
        total += len(piece)
    if total > overlap and len(tail) == 1:
        return [tail[0][-overlap:]]
    return tail


def _merge_pieces(pieces: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    """Greedily pack *pieces* into chunks of at most *chunk_size* characters."""
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for piece in pieces:
        if current and current_len + len(piece) > chunk_size:
            emitted = "".join(current).strip()
            if emitted:
                chunks.append(emitted)
            current = _overlap_tail(current, chunk_overlap)
            current_len = sum(len(part) for part in current)
            # A recursively split piece may itself consume the whole budget.
            # Overlap is optional repeated context, so trim it to the space
            # left by that piece instead of letting overlap + piece exceed the
            # configured hard bound.
            available_overlap = chunk_size - len(piece)
            if current_len > available_overlap:
                overlap_text = "".join(current)
                current = [overlap_text[-available_overlap:]] if available_overlap > 0 else []
                current_len = available_overlap
        current.append(piece)
        current_len += len(piece)

    emitted = "".join(current).strip()
    if emitted:
        chunks.append(emitted)
    return chunks


def split_text(text: str, config: ChunkingConfig) -> list[str]:
    """Split one block of prose into overlapping chunks.

    Operates on a single section body; heading segmentation happens in
    :func:`split_into_sections`. Returns chunks in document order, none longer
    than ``config.chunk_size``. A run with no useful separator is hard-cut to
    preserve that provider-facing bound.
    """
    stripped = text.strip()
    if not stripped:
        return []
    if len(stripped) <= config.chunk_size:
        return [stripped]
    pieces = _split_recursive(stripped, tuple(config.separators), config.chunk_size)
    return _merge_pieces(pieces, config.chunk_size, config.chunk_overlap)


def build_embed_text(
    chunk_text: str,
    *,
    title: str | None,
    heading_path: tuple[str, ...],
    prepend_heading_context: bool,
) -> str:
    """Return the text handed to the embedding model.

    With ``prepend_heading_context`` on, the chunk is prefixed with its document
    title and heading ancestry. It is a cheap, measurable win on short queries —
    a chunk that says "it is enabled by default" is meaningless in isolation and
    unambiguous under "Hybrid search > Configuration".

    A heading equal to the title is not repeated: the prefix should orient the
    model, not spend its context on the same string twice.
    """
    if not prepend_heading_context:
        return chunk_text
    parts: list[str] = []
    if title:
        parts.append(title)
    parts.extend(head for head in heading_path if head != title)
    if not parts:
        return chunk_text
    return f"{' > '.join(parts)}\n\n{chunk_text}"


def chunk_document(document: Document, config: ChunkingConfig) -> tuple[list[Chunk], int]:
    """Split *document* into chunks, numbered across the whole document.

    Args:
        document: The loaded document, front matter already stripped.
        config: Chunk size, overlap and separators.

    Returns:
        The chunks in document order and the number of fragments dropped for
        being shorter than ``min_chunk_chars``. The count is returned rather
        than logged here so the caller can report it: a chunker that silently
        discards a fifth of a corpus looks exactly like a corpus that was
        smaller than expected.
    """
    chunks: list[Chunk] = []
    dropped = 0
    index = 0

    whole_text = document.text.strip()
    if not whole_text:
        return [], 0
    if len(whole_text) < config.min_chunk_chars:
        return [
            Chunk.build(
                document=document,
                chunk_index=0,
                text=whole_text,
                embed_text=build_embed_text(
                    whole_text,
                    title=document.title,
                    heading_path=(),
                    prepend_heading_context=config.prepend_heading_context,
                ),
            )
        ], 0

    for section in split_into_sections(document.text):
        for piece in split_text(section.body, config):
            if len(piece) < config.min_chunk_chars:
                dropped += 1
                continue
            chunks.append(
                Chunk.build(
                    document=document,
                    chunk_index=index,
                    text=piece,
                    embed_text=build_embed_text(
                        piece,
                        title=document.title,
                        heading_path=section.heading_path,
                        prepend_heading_context=config.prepend_heading_context,
                    ),
                    heading=section.heading,
                    heading_path=section.heading_path,
                )
            )
            index += 1

    return chunks, dropped
