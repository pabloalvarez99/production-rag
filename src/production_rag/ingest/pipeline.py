"""The ingest pipeline: corpus directory to points in a collection.

    documents -> sections -> chunks -> fit BM25 -> (skip unchanged)
              -> dense + sparse vectors -> upsert

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

**Why the corpus is materialised before anything is embedded.** BM25 weights
depend on corpus statistics — document frequency per term, average document
length — so they cannot be computed while streaming. The walk therefore collects
every chunk first, fits the encoder, and only then embeds and upserts in batches.
The cost is holding the chunk objects in memory for the length of the run; at a
few hundred bytes of text per chunk that is megabytes, and the alternative (two
full walks of the corpus, or per-shard statistics that make scores
incomparable) is worse. Turning ``ingest.sparse.enabled`` off skips the fit, not
the materialisation, so the two paths cannot diverge in behaviour.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
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
from production_rag.retrieval.sparse import build_sparse_encoder
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
    sparse_enabled: bool = False
    sparse_method: str | None = None
    # The statistics a later sparse score means. "Which corpus were the IDFs
    # computed over?" is the first question when lexical results drift, so the
    # answer travels with the run instead of being reconstructed from logs.
    sparse_stats: dict[str, float | int] | None = None
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

    sparse_config = config.ingest.sparse
    sparse_enabled = sparse_config.enabled and not dry_run
    encoder = (
        build_sparse_encoder(
            method=sparse_config.method,
            k1=sparse_config.k1,
            b=sparse_config.b,
            lowercase=sparse_config.lowercase,
            stopwords=sparse_config.stopwords,
        )
        if sparse_enabled
        else None
    )

    created = False
    if store is not None and not dry_run:
        created = store.ensure_collection(
            vector_size=embedder.dimensions,
            distance=config.qdrant.vectors.dense.distance,
            recreate=recreate,
            with_sparse=sparse_enabled,
        )

    counters = _Counters()
    batch_size = config.ingest.embedding.batch_size
    pending: list[Chunk] = []

    def flush() -> None:
        """Embed and upsert the pending chunks, then clear them."""
        if not pending or store is None:
            pending.clear()
            return
        texts = [chunk.embed_text for chunk in pending]
        vectors = embedder.embed_documents(texts)
        counters.chunks_embedded += len(vectors)
        # Sparse encodes the same text the embedder saw, so a heading a reader
        # would search for is findable lexically as well as semantically.
        sparse_vectors = None if encoder is None else encoder.encode_documents(texts)
        counters.chunks_upserted += store.upsert_chunks(
            pending,
            vectors,
            ingest_run_id=run_id,
            embedded_model=embedder.model,
            sparse_vectors=sparse_vectors,
        )
        pending.clear()

    collected: list[Chunk] = []
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

        collected.extend(chunks)

    # Statistics come from the whole walked corpus, including chunks an
    # incremental run will skip. Fitting on the delta instead would make today's
    # IDFs incomparable with the weights already in the collection, and the
    # damage would show up as gradually worse lexical ranking with no failure.
    if encoder is not None and collected:
        stats = encoder.fit([chunk.embed_text for chunk in collected])
        _log.info("sparse_fitted", method=encoder.method, **stats.as_dict())

    for document_chunks in _grouped_by_document(collected):
        fresh = document_chunks
        if resolved_incremental and store is not None:
            known = store.existing_point_ids([chunk.point_id for chunk in document_chunks])
            if known:
                fresh = [chunk for chunk in document_chunks if chunk.point_id not in known]
                counters.chunks_skipped_unchanged += len(document_chunks) - len(fresh)

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
        sparse_enabled=sparse_enabled,
        sparse_method=None if encoder is None else encoder.method,
        sparse_stats=encoder.stats.as_dict() if encoder is not None and encoder.fitted else None,
        duration_seconds=round(perf_counter() - started, 3),
        per_document=dict(counters.per_document),
    )
    if not dry_run:
        # Identity sidecar is free-path safe: pure disk hash, no Qdrant dial.
        # Cache keys and /ready read it so two corpora cannot cross-hit by name.
        try:
            from production_rag.corpus_identity import (
                build_corpus_identity,
                default_identity_path,
                write_identity_sidecar,
            )

            identity = build_corpus_identity(
                corpus_root=source_dir,
                embedder_id=embedder.model,
                collection=collection_name,
                chunking=config.ingest.chunking,
            )
            write_identity_sidecar(default_identity_path(collection_name), identity)
        except OSError as exc:  # pragma: no cover - best-effort side channel
            _log.warning("identity_sidecar_failed", error=str(exc))
    _log.info(
        "ingest_completed",
        **{
            key: value
            for key, value in asdict(result).items()
            if key not in {"per_document", "sparse_stats"}
        },
    )
    return result


def _grouped_by_document(chunks: Sequence[Chunk]) -> Iterator[list[Chunk]]:
    """Yield consecutive runs of chunks belonging to the same document.

    The skip-check is issued per document rather than per chunk or per corpus:
    one round-trip per file, and a single document's chunks are decided together,
    which is what makes "re-ingest after editing one file" cost one file.
    """
    group: list[Chunk] = []
    for chunk in chunks:
        if group and chunk.doc_id != group[0].doc_id:
            yield group
            group = []
        group.append(chunk)
    if group:
        yield group
