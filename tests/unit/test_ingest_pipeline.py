"""End-to-end ingest over a temporary corpus, with no Qdrant and no network.

The store is :class:`InMemoryVectorStore` — a real implementation of the same
protocol, not a mock — so these tests assert on what would actually be written:
point ids, payload keys, vector lengths, and the counts the CLI reports.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from production_rag.config_loader import ChunkingConfig, IngestConfig, QdrantConfig, YamlConfig
from production_rag.ingest.loaders import CorpusError, iter_documents, load_document
from production_rag.ingest.models import ROOT_SOURCE
from production_rag.ingest.pipeline import IngestError, IngestResult, run_ingest
from production_rag.retrieval.embeddings import FAKE_DIMENSIONS, FakeEmbeddingProvider
from production_rag.retrieval.store import (
    DENSE_VECTOR_NAME,
    CollectionMismatchError,
    InMemoryVectorStore,
)

BODY = " ".join(["Retrieval augmented generation grounds an answer in sources."] * 12)


def _config(**chunking: object) -> YamlConfig:
    """A YAML config with a small embedding batch, to exercise batching."""
    return YamlConfig(
        ingest=IngestConfig(
            chunking=ChunkingConfig(**chunking),  # type: ignore[arg-type]
            embedding=IngestConfig().embedding.model_copy(update={"batch_size": 2}),
        ),
        qdrant=QdrantConfig(),
    )


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """Three documents in two folders, plus files that must be ignored."""
    root = tmp_path / "raw"
    (root / "guide").mkdir(parents=True)
    (root / "notes").mkdir()
    (root / "guide" / "hybrid.md").write_text(
        "---\ntitle: Hybrid Search\ntags: [retrieval, qdrant]\n---\n\n"
        f"## Why hybrid\n{BODY}\n\n## Fusion\n{BODY}\n",
        encoding="utf-8",
    )
    (root / "notes" / "chunking.md").write_text(f"# Chunking\n\n{BODY}\n", encoding="utf-8")
    (root / "top.txt").write_text(BODY, encoding="utf-8")
    (root / "image.png").write_bytes(b"not text")
    (root / ".hidden.md").write_text(BODY, encoding="utf-8")
    return root


def _run(corpus: Path, store: InMemoryVectorStore | None = None, **kwargs: object) -> IngestResult:
    return run_ingest(
        source_dir=corpus,
        config=_config(),
        embedder=FakeEmbeddingProvider(),
        store=store,
        ingest_run_id="run-1",
        **kwargs,  # type: ignore[arg-type]
    )


# --- loading ---------------------------------------------------------------


EXCLUDES = IngestConfig().exclude_globs


def test_walk_finds_only_allowed_extensions(corpus: Path) -> None:
    # image.png is skipped on extension; .hidden.md on the exclude glob.
    paths = [document.source_path for document in iter_documents(corpus, exclude_globs=EXCLUDES)]
    assert sorted(paths) == ["guide/hybrid.md", "notes/chunking.md", "top.txt"]


def test_dotfiles_are_only_skipped_because_a_glob_says_so(corpus: Path) -> None:
    # Nothing is hidden by default: the exclusion is configuration, not a rule
    # buried in the walker.
    paths = [document.source_path for document in iter_documents(corpus)]
    assert ".hidden.md" in paths


def test_source_paths_use_forward_slashes(corpus: Path) -> None:
    assert all("\\" not in document.source_path for document in iter_documents(corpus))


def test_source_is_the_first_path_segment(corpus: Path) -> None:
    sources = {document.source_path: document.source for document in iter_documents(corpus)}
    assert sources["guide/hybrid.md"] == "guide"
    assert sources["top.txt"] == ROOT_SOURCE


def test_front_matter_becomes_title_and_tags_and_leaves_the_body(corpus: Path) -> None:
    document = load_document(corpus / "guide" / "hybrid.md", corpus)
    assert document.title == "Hybrid Search"
    assert document.tags == ("retrieval", "qdrant")
    assert not document.text.lstrip().startswith("---")


def test_first_h1_is_the_title_without_front_matter(corpus: Path) -> None:
    assert load_document(corpus / "notes" / "chunking.md", corpus).title == "Chunking"


def test_file_stem_is_the_last_resort_title(corpus: Path) -> None:
    assert load_document(corpus / "top.txt", corpus).title == "top"


def test_missing_corpus_root_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(CorpusError):
        list(iter_documents(tmp_path / "absent"))


# --- pipeline --------------------------------------------------------------


def test_ingest_writes_one_point_per_chunk(corpus: Path) -> None:
    store = InMemoryVectorStore()
    result = _run(corpus, store)
    assert result.documents_scanned == 3
    assert result.documents_ingested == 3
    assert result.chunks_created > 3
    assert result.chunks_embedded == result.chunks_created
    assert result.chunks_upserted == result.chunks_created
    assert result.points_in_collection == result.chunks_created
    assert len(store.points) == result.chunks_created


def test_vectors_are_stored_under_the_dense_name_at_the_right_size(corpus: Path) -> None:
    store = InMemoryVectorStore()
    _run(corpus, store)
    assert all(len(vector) == FAKE_DIMENSIONS for vector in store.vectors.values())
    assert DENSE_VECTOR_NAME == "dense"


def test_collection_is_created_on_the_first_run(corpus: Path) -> None:
    store = InMemoryVectorStore()
    assert _run(corpus, store).collection_created is True
    assert _run(corpus, store).collection_created is False


def test_payload_carries_provenance_and_never_the_embed_text(corpus: Path) -> None:
    store = InMemoryVectorStore()
    _run(corpus, store)
    payload = next(iter(store.points.values()))
    assert set(payload) == {
        "chunk_id",
        "text",
        "source_path",
        "source",
        "title",
        "heading",
        "heading_path",
        "chunk_index",
        "content_sha256",
        "doc_id",
        "token_count_est",
        "tags",
        "ingest_run_id",
        "embedded_model",
    }
    assert payload["ingest_run_id"] == "run-1"
    assert payload["embedded_model"] == FakeEmbeddingProvider().model
    assert "embed_text" not in payload


def test_rerun_is_idempotent(corpus: Path) -> None:
    # The property that makes ingest safe to run from a cron job: the same corpus
    # twice is the same collection, not a doubled one.
    store = InMemoryVectorStore()
    first = _run(corpus, store)
    second = _run(corpus, store)
    assert second.points_in_collection == first.points_in_collection


def test_rerun_skips_unchanged_chunks_instead_of_re_embedding(corpus: Path) -> None:
    store = InMemoryVectorStore()
    first = _run(corpus, store)
    second = _run(corpus, store)
    assert second.chunks_skipped_unchanged == first.chunks_created
    # The paid call is what matters: nothing was embedded the second time.
    assert second.chunks_embedded == 0


def test_no_incremental_re_embeds_everything(corpus: Path) -> None:
    store = InMemoryVectorStore()
    first = _run(corpus, store)
    second = _run(corpus, store, incremental=False)
    assert second.chunks_embedded == first.chunks_created
    assert second.chunks_skipped_unchanged == 0


def test_editing_a_document_only_re_embeds_that_document(corpus: Path) -> None:
    store = InMemoryVectorStore()
    _run(corpus, store)
    (corpus / "notes" / "chunking.md").write_text(
        f"# Chunking\n\nRewritten. {BODY}\n", encoding="utf-8"
    )
    second = _run(corpus, store)
    assert 0 < second.chunks_embedded < second.chunks_created


def test_recreate_clears_the_collection_first(corpus: Path) -> None:
    store = InMemoryVectorStore()
    first = _run(corpus, store)
    recreated = _run(corpus, store, recreate=True)
    assert recreated.chunks_embedded == recreated.chunks_created
    assert recreated.points_in_collection == first.points_in_collection


def test_dry_run_touches_no_store_and_reports_counts(corpus: Path) -> None:
    result = _run(corpus, None, dry_run=True, collection="whatever")
    assert result.dry_run is True
    assert result.chunks_created > 0
    assert result.chunks_embedded == 0
    assert result.chunks_upserted == 0
    assert result.points_in_collection == 0
    assert result.collection == "whatever"


def test_a_non_dry_run_without_a_store_is_refused() -> None:
    with pytest.raises(IngestError, match="vector store is required"):
        run_ingest(
            source_dir="data/raw",
            config=_config(),
            embedder=FakeEmbeddingProvider(),
            store=None,
        )


def test_a_document_that_yields_no_chunks_does_not_abort_the_run(corpus: Path) -> None:
    (corpus / "notes" / "stub.md").write_text("# Stub\n\ntoo short\n", encoding="utf-8")
    result = _run(corpus, InMemoryVectorStore())
    assert result.documents_scanned == 4
    assert result.documents_ingested == 3
    assert result.documents_without_chunks == 1


def test_dropped_short_fragments_are_counted(corpus: Path) -> None:
    result = run_ingest(
        source_dir=corpus,
        config=_config(min_chunk_chars=10_000),
        embedder=FakeEmbeddingProvider(),
        store=InMemoryVectorStore(),
    )
    assert result.chunks_created == 0
    assert result.chunks_dropped_short > 0


def test_changing_embedder_dimensions_is_refused_without_recreate(corpus: Path) -> None:
    store = InMemoryVectorStore()
    _run(corpus, store)
    with pytest.raises(CollectionMismatchError):
        run_ingest(
            source_dir=corpus,
            config=_config(),
            embedder=FakeEmbeddingProvider(dimensions=64),
            store=store,
        )


def test_summary_is_json_serialisable(corpus: Path) -> None:
    import json

    summary = _run(corpus, InMemoryVectorStore()).to_summary()
    assert json.loads(json.dumps(summary))["ok"] is True
    assert summary["ingest_run_id"] == "run-1"
