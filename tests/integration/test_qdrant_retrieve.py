"""Hybrid retrieval against a real Qdrant.

Skipped unless ``RUN_QDRANT_TESTS=1``, so the default suite stays offline. With a
container up (``make up``, or ``docker compose up -d qdrant``):

    RUN_QDRANT_TESTS=1 python -m pytest -m integration

What this covers that :class:`InMemoryVectorStore` cannot:

* Qdrant accepts a point carrying **both** named vectors in one upsert.
* ``query_points(using=...)`` selects the branch it is asked for.
* A sparse score coming back from Qdrant equals the BM25 dot product computed
  locally — which is the assertion that would catch a double-applied IDF, the one
  mistake this design is shaped to avoid.
* The migration guard fires against a live M1-shaped collection.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest

from production_rag.config_loader import PayloadIndexConfig, RetrievalConfig
from production_rag.ingest.models import Chunk, Document
from production_rag.retrieval.embeddings import FakeEmbeddingProvider
from production_rag.retrieval.hybrid import MODE_DENSE, MODE_HYBRID, MODE_SPARSE, Retriever
from production_rag.retrieval.sparse import Bm25Encoder
from production_rag.retrieval.store import CollectionMismatchError, QdrantStore

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_QDRANT_TESTS") != "1",
        reason="needs a reachable Qdrant; set RUN_QDRANT_TESTS=1",
    ),
]

CORPUS = {
    "sample/00-intro.md": "Production RAG combines retrieval with generation over a corpus.",
    "sample/01-hybrid.md": "Qdrant stores dense and sparse vectors in one collection.",
    "sample/02-flags.md": "Pass --recreate-collection to rebuild the index from scratch.",
}


@pytest.fixture
def store() -> Iterator[QdrantStore]:
    """A throwaway collection, deleted however the test ends."""
    collection = f"test_retrieve_{uuid.uuid4().hex[:8]}"
    subject = QdrantStore(
        url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        collection=collection,
        api_key=os.environ.get("QDRANT_API_KEY"),
        payload_indexes=(PayloadIndexConfig(field="doc_id"),),
    )
    try:
        yield subject
    finally:
        subject.client.delete_collection(collection)


def _chunks() -> list[Chunk]:
    chunks = []
    for index, (path, text) in enumerate(CORPUS.items()):
        document = Document(source_path=path, text=text, title=path, source="sample")
        chunks.append(Chunk.build(document=document, chunk_index=index, text=text, embed_text=text))
    return chunks


def _indexed(store: QdrantStore) -> tuple[FakeEmbeddingProvider, Bm25Encoder, list[Chunk]]:
    """Create the dual-vector collection and fill it."""
    embedder = FakeEmbeddingProvider()
    chunks = _chunks()
    texts = [chunk.embed_text for chunk in chunks]
    encoder = Bm25Encoder()
    encoder.fit(texts)
    store.ensure_collection(vector_size=embedder.dimensions, with_sparse=True)
    store.upsert_chunks(
        chunks,
        embedder.embed_documents(texts),
        ingest_run_id="integration",
        embedded_model=embedder.model,
        sparse_vectors=encoder.encode_documents(texts),
    )
    return embedder, encoder, chunks


def test_both_named_vectors_land_in_one_point(store: QdrantStore) -> None:
    _, _, chunks = _indexed(store)
    assert store.count() == len(chunks)


def test_dense_search_returns_scored_hits(store: QdrantStore) -> None:
    embedder, _, chunks = _indexed(store)
    hits = store.search_dense(embedder.embed_query("qdrant vectors"), limit=len(chunks))
    assert len(hits) == len(chunks)
    assert hits[0].score >= hits[-1].score
    assert hits[0].chunk_id


def test_sparse_search_matches_a_literal_flag(store: QdrantStore) -> None:
    _, encoder, _ = _indexed(store)
    hits = store.search_sparse(encoder.encode_query("--recreate-collection"), limit=5)
    assert hits
    assert hits[0].payload["source_path"] == "sample/02-flags.md"


def test_a_sparse_score_from_qdrant_equals_the_local_bm25_dot_product(
    store: QdrantStore,
) -> None:
    # The assertion that catches a double-applied IDF: if Qdrant were reweighting
    # the vector (modifier: idf), these two numbers would differ.
    _, encoder, chunks = _indexed(store)
    query = encoder.encode_query("qdrant vectors")
    remote = store.search_sparse(query, limit=len(chunks))
    local = {
        chunk.point_id: query.dot(vector)
        for chunk, vector in zip(
            chunks,
            encoder.encode_documents([chunk.embed_text for chunk in chunks]),
            strict=True,
        )
    }
    for hit in remote:
        assert hit.score == pytest.approx(local[hit.point_id], rel=1e-5)


def test_payload_projection_reaches_the_wire(store: QdrantStore) -> None:
    embedder, _, _ = _indexed(store)
    hit = store.search_dense(embedder.embed_query("qdrant"), limit=1, with_payload=["chunk_id"])[0]
    assert set(hit.payload) == {"chunk_id"}


def test_the_retriever_fuses_both_branches_end_to_end(store: QdrantStore) -> None:
    embedder, encoder, _ = _indexed(store)
    retriever = Retriever(
        store=store,
        embedder=embedder,
        config=RetrievalConfig(top_k=3),
        sparse_encoder=encoder,
    )
    result = retriever.retrieve("--recreate-collection", mode=MODE_HYBRID)
    assert result.hits[0].source_path == "sample/02-flags.md"
    assert result.dense_candidates > 0
    assert result.sparse_candidates > 0


@pytest.mark.parametrize("mode", [MODE_DENSE, MODE_SPARSE, MODE_HYBRID])
def test_every_mode_works_against_a_live_collection(store: QdrantStore, mode: str) -> None:
    embedder, encoder, _ = _indexed(store)
    retriever = Retriever(store=store, embedder=embedder, sparse_encoder=encoder)
    assert retriever.retrieve("qdrant vectors", mode=mode).hits


def test_a_dense_only_collection_refuses_the_sparse_migration(store: QdrantStore) -> None:
    # Qdrant cannot add a named vector in place, so an M1-shaped collection has to
    # be recreated. The error has to say that instead of failing at upsert.
    embedder = FakeEmbeddingProvider()
    store.ensure_collection(vector_size=embedder.dimensions)
    with pytest.raises(CollectionMismatchError, match="recreate-collection"):
        store.ensure_collection(vector_size=embedder.dimensions, with_sparse=True)


def test_recreate_performs_the_sparse_migration(store: QdrantStore) -> None:
    embedder = FakeEmbeddingProvider()
    store.ensure_collection(vector_size=embedder.dimensions)
    assert (
        store.ensure_collection(vector_size=embedder.dimensions, recreate=True, with_sparse=True)
        is True
    )
    _, encoder, _chunks_written = _indexed(store)
    assert store.search_sparse(encoder.encode_query("qdrant"), limit=5)
