"""Round-trip against a real Qdrant.

Skipped unless ``RUN_QDRANT_TESTS=1``, so the default suite stays offline. With a
container up (``make up``, or ``docker compose up -d qdrant``):

    RUN_QDRANT_TESTS=1 python -m pytest -m integration

What this covers that the in-memory store cannot: the named-vector shape Qdrant
actually accepts, that a UUID point id is a legal id, that ``wait=True`` makes a
count immediately after an upsert truthful, and that the vector-size guard fires
against a live collection.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest

from production_rag.config_loader import ChunkingConfig, PayloadIndexConfig
from production_rag.ingest.chunking import chunk_document
from production_rag.ingest.models import Chunk, Document
from production_rag.retrieval.embeddings import FAKE_DIMENSIONS, FakeEmbeddingProvider
from production_rag.retrieval.store import CollectionMismatchError, QdrantStore

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_QDRANT_TESTS") != "1",
        reason="needs a reachable Qdrant; set RUN_QDRANT_TESTS=1",
    ),
]

BODY = " ".join(["Qdrant stores the vector and the payload together."] * 12)


@pytest.fixture
def store() -> Iterator[QdrantStore]:
    """A throwaway collection, deleted however the test ends."""
    collection = f"test_ingest_{uuid.uuid4().hex[:8]}"
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
    document = Document(source_path="guide/qdrant.md", text=BODY, title="Qdrant", source="guide")
    chunks, _ = chunk_document(document, ChunkingConfig())
    return chunks


def test_upsert_then_count_is_immediately_truthful(store: QdrantStore) -> None:
    embedder = FakeEmbeddingProvider()
    chunks = _chunks()

    assert store.ensure_collection(vector_size=embedder.dimensions) is True
    written = store.upsert_chunks(
        chunks,
        embedder.embed_documents([chunk.embed_text for chunk in chunks]),
        ingest_run_id="integration",
        embedded_model=embedder.model,
    )

    assert written == len(chunks)
    assert store.count() == len(chunks)


def test_reupsert_overwrites_rather_than_duplicates(store: QdrantStore) -> None:
    embedder = FakeEmbeddingProvider()
    chunks = _chunks()
    vectors = embedder.embed_documents([chunk.embed_text for chunk in chunks])
    store.ensure_collection(vector_size=embedder.dimensions)

    for _ in range(2):
        store.upsert_chunks(
            chunks, vectors, ingest_run_id="integration", embedded_model=embedder.model
        )

    assert store.count() == len(chunks)
    assert store.existing_point_ids([chunk.point_id for chunk in chunks]) == {
        chunk.point_id for chunk in chunks
    }


def test_vector_size_mismatch_is_refused(store: QdrantStore) -> None:
    store.ensure_collection(vector_size=FAKE_DIMENSIONS)
    with pytest.raises(CollectionMismatchError, match="recreate"):
        store.ensure_collection(vector_size=FAKE_DIMENSIONS * 2)


def test_recreate_drops_the_existing_collection(store: QdrantStore) -> None:
    embedder = FakeEmbeddingProvider()
    chunks = _chunks()
    store.ensure_collection(vector_size=embedder.dimensions)
    store.upsert_chunks(
        chunks,
        embedder.embed_documents([chunk.embed_text for chunk in chunks]),
        ingest_run_id="integration",
        embedded_model=embedder.model,
    )

    assert store.ensure_collection(vector_size=embedder.dimensions, recreate=True) is True
    assert store.count() == 0
