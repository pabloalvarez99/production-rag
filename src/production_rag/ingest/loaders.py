"""Corpus discovery: filesystem to :class:`Document`.

Three properties this module is responsible for, all of which are the kind of
thing that quietly breaks a corpus:

* **Deterministic order.** Files are walked sorted, so two ingest runs over the
  same corpus produce the same chunk indices and therefore the same ``chunk_id``
  values. Golden evaluation sets reference those ids.
* **Loud skips.** A file with an unlisted extension is logged, not ignored. A
  silently skipped PDF looks exactly like an empty corpus.
* **Path canonicalisation.** ``source_path`` is always corpus-relative with
  forward slashes, so a document ingested on Windows and re-ingested in the
  Linux container keeps its ``doc_id``.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from fnmatch import fnmatch
from pathlib import Path

import structlog
import yaml

from production_rag.ingest.hashing import normalise_source_path
from production_rag.ingest.models import ROOT_SOURCE, Document

_log = structlog.get_logger(__name__)

_FRONT_MATTER_FENCE = "---"
_H1_PREFIX = "# "


class CorpusError(RuntimeError):
    """The corpus root is missing or is not a directory."""


_DOTFILE_PATTERNS = frozenset({".*", "*/.*", "**/.*"})
"""Patterns that mean "skip hidden files", at any depth."""


def _is_excluded(relative_posix: str, patterns: Sequence[str]) -> bool:
    """Whether a corpus-relative path matches any exclusion pattern.

    ``fnmatch`` is used rather than :meth:`pathlib.Path.match` because the
    patterns in ``configs/default.yaml`` use ``**``, which ``Path.match`` does
    not support on 3.12.

    A dotfile pattern is widened to every path segment: ``**/.*`` reads as "skip
    hidden files", but ``fnmatch`` requires the literal ``/`` and so would miss a
    dotfile sitting directly in the corpus root. Nothing is hidden unless a
    pattern says so — what is walked stays a property of the config file rather
    than of this function.
    """
    segments = relative_posix.split("/")
    for pattern in patterns:
        if pattern in _DOTFILE_PATTERNS and any(part.startswith(".") for part in segments):
            return True
        if fnmatch(relative_posix, pattern) or fnmatch(f"/{relative_posix}", pattern):
            return True
    return False


def split_front_matter(raw: str) -> tuple[dict[str, object], str]:
    """Split optional YAML front matter from the body.

    Returns ``({}, raw)`` when there is no front matter, or when the block is
    malformed: a typo in a metadata header should cost the metadata, not the
    document. The failure is logged so it is not invisible.
    """
    if not raw.startswith(_FRONT_MATTER_FENCE):
        return {}, raw

    lines = raw.splitlines(keepends=True)
    for position, line in enumerate(lines[1:], start=1):
        if line.strip() != _FRONT_MATTER_FENCE:
            continue
        block = "".join(lines[1:position])
        body = "".join(lines[position + 1 :])
        try:
            parsed = yaml.safe_load(block)
        except yaml.YAMLError as exc:
            _log.warning("front_matter_unparsable", error=str(exc))
            return {}, raw
        if not isinstance(parsed, dict):
            return {}, body
        return parsed, body
    return {}, raw


def _first_h1(text: str) -> str | None:
    """Return the first level-one Markdown heading, if the document has one."""
    for line in text.splitlines():
        if line.startswith(_H1_PREFIX):
            heading = line[len(_H1_PREFIX) :].strip().rstrip("#").strip()
            if heading:
                return heading
    return None


def _tags_from(front_matter: dict[str, object]) -> tuple[str, ...]:
    """Normalise the ``tags`` front matter field into a tuple of strings.

    Accepts a YAML list or a comma-separated string, because both forms show up
    in real corpora and rejecting one would be a pointless papercut.
    """
    raw = front_matter.get("tags")
    if raw is None:
        return ()
    if isinstance(raw, str):
        return tuple(tag.strip() for tag in raw.split(",") if tag.strip())
    if isinstance(raw, (list, tuple)):
        return tuple(str(tag).strip() for tag in raw if str(tag).strip())
    return ()


def _source_of(relative_posix: str) -> str:
    """First path segment under the corpus root; the filterable ``source``."""
    head, separator, _ = relative_posix.partition("/")
    return head if separator else ROOT_SOURCE


def load_document(path: Path, root: Path) -> Document:
    """Read one file into a :class:`Document`.

    The title is taken from front matter, then the first H1, then the file stem.
    Front matter is removed from the body: a YAML header embedded in an
    embedding input is noise that every chunk of the document would carry.
    """
    relative = normalise_source_path(str(path.relative_to(root)))
    raw = path.read_text(encoding="utf-8", errors="replace")
    front_matter, body = split_front_matter(raw)

    declared_title = front_matter.get("title")
    title = str(declared_title).strip() if declared_title else None
    return Document(
        source_path=relative,
        text=body.strip(),
        title=title or _first_h1(body) or path.stem,
        source=_source_of(relative),
        tags=_tags_from(front_matter),
    )


def iter_documents(
    root: str | Path,
    *,
    include_extensions: Sequence[str] = (".md", ".markdown", ".txt"),
    exclude_globs: Sequence[str] = (),
) -> Iterator[Document]:
    """Yield every ingestible document under *root*, in deterministic order.

    Args:
        root: Corpus root. ``source_path`` values are relative to it.
        include_extensions: Extensions to walk, matched case-insensitively.
        exclude_globs: Patterns to skip, as in ``ingest.exclude_globs``.

    Yields:
        Documents in sorted path order.

    Raises:
        CorpusError: *root* does not exist or is not a directory. This is a
            configuration mistake, and an empty result would look like an empty
            corpus — two problems with very different fixes.
    """
    corpus_root = Path(root)
    if not corpus_root.is_dir():
        raise CorpusError(f"corpus root does not exist or is not a directory: {corpus_root}")

    wanted = {extension.lower() for extension in include_extensions}
    empty_documents = 0

    for path in sorted(corpus_root.rglob("*")):
        if not path.is_file():
            continue
        relative = normalise_source_path(str(path.relative_to(corpus_root)))
        if _is_excluded(relative, exclude_globs):
            _log.debug("document_excluded", source_path=relative)
            continue
        if path.suffix.lower() not in wanted:
            _log.info("document_skipped_extension", source_path=relative, suffix=path.suffix)
            continue

        document = load_document(path, corpus_root)
        if not document.text:
            empty_documents += 1
            _log.warning("document_empty", source_path=relative)
            continue
        yield document

    if empty_documents:
        _log.warning("documents_empty_total", count=empty_documents)
