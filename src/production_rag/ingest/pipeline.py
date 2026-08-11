"""The ingest pipeline: corpus directory to points in a collection.

    documents -> sections -> chunks -> (skip unchanged) -> vectors -> upsert

Two properties are worth more than the code that implements them:

**Idempotent.** A point id is a UUID5 over path, chunk index and content hash, so
re-running over an unchanged corpus rewrites the same points. Nothing
duplicates, and an edited paragraph lands on a new id instead of leaving a stale
vector under a fresh payload.

**Incremental by default.** Embedding is the only step in this project that
spends real money. Chunks whose point id is already in the collection are never
embedded again, which makes "re-ingest after editing one file" cost one file.

Every count the run produces is reported. A pipeline that quietly drops a fifth
of a corpus looks exactly like a corpus that was smaller than expected, and the
difference is only visible if the numbers are printed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

import structlog

from production_rag.config_loader import YamlConfig
from production_rag.ingest.chunking import chunk_document
from production_rag.ingest.loaders import iter_documents
from production_rag.ingest.models import Chunk
from production_rag.retrieval.embeddings import EmbeddingProvider
from production_rag.retrieval.store import VectorStore

_log = structlog.get_logger(__name__)


class IngestError(RuntimeError):
    """The pipeline was asked to do something it cannot."""


@dataclass(frozen=True, slots=True)
class IngestResult:
    """Everything one ingest run did, in numbers.

    Returned rather than logged so callers can assert on it: the CLI prints it as
    JSON, the tests check it, and a future eval run records it next to the
    metrics it produced.
    """

    ingest_run_id: str
    collection: str
    source_dir: str
    embedded_model: str
    vector_size: int
    dry_run: bool = False
    documents_scanned: int = 0
    documents_ingested: int = 0
    documents_without_chunks: int = 0
    chunks_created: int = 0
    chunks_dropped_short: int = 0
    chunks_skipped_unchanged: int = 0
    chunks_embedded: int = 0
    chunks_upserted: int = 0
    points_in_collection: int = 0
    collection_created: bool = False
    duration_seconds: float = 0.0
    per_document: dict[str, int] = field(default_factory=dict)

    def to_summary(self) -> dict[str, Any]:
        """Machine-readable summary, the CLI's last line of stdout."""
        summary = asdict(self)
        summary["ok"] = True
        return summary


@dataclass(slots=True)
class _Counters:
    """Mutable tallies accumulated while walking the corpus."""

    documents_scanned: int = 0
    documents_ingested: int = 0
    documents_without_chunks: int = 0
    chunks_created: int = 0
    chunks_dropped_short: int = 0
    chunks_skipped_unchanged: int = 0
    chunks_embedded: int = 0
    chunks_upserted: int = 0
    per_document: dict[str, int] = field(default_factory=dict)


def run_ingest(
    *,
    source_dir: str | Path,
    config: YamlConfig,
    embedder: EmbeddingProvider,
    store: VectorStore | None = None,
    collection: str | None = None,
    recreate: bool = False,
    incremental: bool | None = None,
    dry_run: bool = False,
    ingest_run_id: str | None = None,
) -> IngestResult:
    """Ingest every document under *source_dir* into the vector store.

    Args:
        source_dir: Corpus root. ``source_path`` payload values are relative to
            it, so ingesting ``data/raw`` and ``data/raw/sample`` produces
            different — and both valid — provenance.
        config: The loaded YAML profile; supplies chunking, extensions and the
            embedding batch size.
        embedder: Any :class:`EmbeddingProvider`. Its ``dimensions`` decides the
            collection's vector size and its ``model`` is stamped on every point.
        store: Target store. Required unless *dry_run*.
        collection: Name to report in the result. Only useful for a dry run,
            where there is no store to ask; otherwise the store's own name wins.
        recreate: Drop the collection before writing. Destructive, and the only
            way to change embedding model or vector size.
        incremental: Skip chunks already present. Defaults to the YAML setting.
            Forced off by *recreate*, since nothing is present after a drop.
        dry_run: Walk and chunk only — no embedding call, no upsert, no
            connection to the store. This is the mode that costs nothing and
            answers "what would this corpus turn into?".
        ingest_run_id: Override the generated run id, for reproducible tests.

    Returns:
        An :class:`IngestResult` with every count from the run.

    Raises:
        IngestError: No store was given for a non-dry run.
        CorpusError: *source_dir* is missing or is not a directory.
        VectorStoreError: The collection could not be prepared or written.
        EmbeddingError: The embedding provider failed.
    """
    if store is None and not dry_run:
        raise IngestError("a vector store is required unless dry_run=True")

    run_id = ingest_run_id or uuid4().hex[:12]
    resolved_incremental = config.ingest.incremental if incremental is None else incremental
    if recreate:
        # Everything is new after a drop; checking for existing ids would be a
        # round-trip per batch that can only ever answer "no".
        resolved_incremental = False

    started = perf_counter()
    if store is not None:
        collection_name = store.collection
    else:
        collection_name = collection or config.qdrant.collection
    _log.info(
        "ingest_started",
        ingest_run_id=run_id,
        source_dir=str(source_dir),
        collection=collection_name,
        embedded_model=embedder.model,
        vector_size=embedder.dimensions,
        dry_run=dry_run,
        incremental=resolved_incremental,
        recreate=recreate,
    )

    created = False
    if store is not None and not dry_run:
        created = store.ensure_collection(
            vector_size=embedder.dimensions,
            distance=config.qdrant.vectors.dense.distance,
            recreate=recreate,
        )

    counters = _Counters()
    batch_size = config.ingest.embedding.batch_size
    pending: list[Chunk] = []

    def flush() -> None:
        """Embed and upsert the pending chunks, then clear them."""
        if not pending or store is None:
            pending.clear()
            return
        vectors = embedder.embed_documents([chunk.embed_text for chunk in pending])
        counters.chunks_embedded += len(vectors)
        counters.chunks_upserted += store.upsert_chunks(
            pending,
            vectors,
            ingest_run_id=run_id,
            embedded_model=embedder.model,
        )
        pending.clear()

    for document in iter_documents(
        source_dir,
        include_extensions=config.ingest.include_extensions,
        exclude_globs=config.ingest.exclude_globs,
    ):
        counters.documents_scanned += 1
        chunks, dropped = chunk_document(document, config.ingest.chunking)
        counters.chunks_dropped_short += dropped
        counters.chunks_created += len(chunks)
        counters.per_document[document.source_path] = len(chunks)

        if not chunks:
            counters.documents_without_chunks += 1
            # Not an error, but never silent: with the default 120-character
            # floor, a stub document legitimately produces nothing.
            _log.warning(
                "document_without_chunks",
                source_path=document.source_path,
                dropped_short=dropped,
            )
            continue

        counters.documents_ingested += 1
        _log.debug(
            "document_chunked",
            source_path=document.source_path,
            doc_id=document.doc_id,
            chunks=len(chunks),
            dropped_short=dropped,
        )

        if dry_run:
            continue

        fresh = chunks
        if resolved_incremental and store is not None:
            known = store.existing_point_ids([chunk.point_id for chunk in chunks])
            if known:
                fresh = [chunk for chunk in chunks if chunk.point_id not in known]
                counters.chunks_skipped_unchanged += len(chunks) - len(fresh)

        for chunk in fresh:
            pending.append(chunk)
            if len(pending) >= batch_size:
                flush()

    flush()

    points = 0 if (dry_run or store is None) else store.count()
    result = IngestResult(
        ingest_run_id=run_id,
        collection=collection_name,
        source_dir=str(source_dir),
        embedded_model=embedder.model,
        vector_size=embedder.dimensions,
        dry_run=dry_run,
        documents_scanned=counters.documents_scanned,
        documents_ingested=counters.documents_ingested,
        documents_without_chunks=counters.documents_without_chunks,
        chunks_created=counters.chunks_created,
        chunks_dropped_short=counters.chunks_dropped_short,
        chunks_skipped_unchanged=counters.chunks_skipped_unchanged,
        chunks_embedded=counters.chunks_embedded,
        chunks_upserted=counters.chunks_upserted,
        points_in_collection=points,
        collection_created=created,
        duration_seconds=round(perf_counter() - started, 3),
        per_document=dict(counters.per_document),
    )
    _log.info(
        "ingest_completed",
        **{key: value for key, value in asdict(result).items() if key != "per_document"},
    )
    return result
