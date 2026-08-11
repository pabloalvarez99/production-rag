"""The offline store: dual-vector writes and both search branches.

:class:`InMemoryVectorStore` is the substrate the retriever and the source-hit
eval run on, so its ranking behaviour is tested as behaviour, not as a stub.
"""

from __future__ import annotations

import pytest

from production_rag.ingest.models import Chunk, Document
from production_rag.retrieval.sparse import Bm25Encoder, SparseVector
from production_rag.retrieval.store import (
    CollectionMismatchError,
    InMemoryVectorStore,
    SearchableVectorStore,
    VectorStoreError,
)

RUN_ID = "run-test"
MODEL = "fake-deterministic-v1"


def _chunk(index: int, text: str, *, source_path: str = "sample/doc.md") -> Chunk:
    document = Document(source_path=source_path, text=text, title="Doc", source="sample")
    return Chunk.build(document=document, chunk_index=index, text=text, embed_text=text)


def _store(*, with_sparse: bool = True, size: int = 3) -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    store.ensure_collection(vector_size=size, with_sparse=with_sparse)
    return store


class TestDualWrite:
    def test_satisfies_the_searchable_protocol(self) -> None:
        assert isinstance(InMemoryVectorStore(), SearchableVectorStore)

    def test_stores_dense_and_sparse_for_the_same_point(self) -> None:
        store = _store()
        chunk = _chunk(0, "qdrant stores vectors")
        sparse = SparseVector.from_weights({1: 2.0})
        store.upsert_chunks(
            [chunk],
            [[1.0, 0.0, 0.0]],
            ingest_run_id=RUN_ID,
            embedded_model=MODEL,
            sparse_vectors=[sparse],
        )
        assert store.vectors[chunk.point_id] == [1.0, 0.0, 0.0]
        assert store.sparse_vectors[chunk.point_id] == sparse

    def test_sparse_write_without_a_sparse_collection_is_refused(self) -> None:
        # Same migration rule as Qdrant, enforced offline so a test catches it
        # before an operator does.
        store = _store(with_sparse=False)
        with pytest.raises(CollectionMismatchError, match="recreate-collection"):
            store.upsert_chunks(
                [_chunk(0, "text")],
                [[1.0, 0.0, 0.0]],
                ingest_run_id=RUN_ID,
                embedded_model=MODEL,
                sparse_vectors=[SparseVector.from_weights({1: 1.0})],
            )

    def test_adding_sparse_to_a_dense_only_collection_is_refused(self) -> None:
        store = _store(with_sparse=False)
        with pytest.raises(CollectionMismatchError, match="recreate-collection"):
            store.ensure_collection(vector_size=3, with_sparse=True)

    def test_recreate_allows_the_sparse_migration(self) -> None:
        store = _store(with_sparse=False)
        assert store.ensure_collection(vector_size=3, recreate=True, with_sparse=True) is True

    def test_a_mismatched_sparse_batch_length_is_refused(self) -> None:
        store = _store()
        with pytest.raises(VectorStoreError, match="sparse encoder returned"):
            store.upsert_chunks(
                [_chunk(0, "a"), _chunk(1, "b")],
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                ingest_run_id=RUN_ID,
                embedded_model=MODEL,
                sparse_vectors=[SparseVector.from_weights({1: 1.0})],
            )

    def test_dense_only_writes_still_work(self) -> None:
        # M1 behaviour: sparse is opt-in and its absence is not an error.
        store = _store(with_sparse=False)
        written = store.upsert_chunks(
            [_chunk(0, "text")], [[1.0, 0.0, 0.0]], ingest_run_id=RUN_ID, embedded_model=MODEL
        )
        assert written == 1
        assert store.sparse_vectors == {}


class TestDenseSearch:
    def test_ranks_by_cosine_similarity(self) -> None:
        store = _store()
        chunks = [_chunk(0, "a"), _chunk(1, "b"), _chunk(2, "c")]
        store.upsert_chunks(
            chunks,
            [[1.0, 0.0, 0.0], [0.7, 0.7, 0.0], [0.0, 0.0, 1.0]],
            ingest_run_id=RUN_ID,
            embedded_model=MODEL,
        )
        hits = store.search_dense([1.0, 0.0, 0.0], limit=3)
        assert [hit.point_id for hit in hits] == [chunk.point_id for chunk in chunks]
        assert hits[0].score == pytest.approx(1.0)

    def test_is_magnitude_independent(self) -> None:
        store = _store()
        store.upsert_chunks(
            [_chunk(0, "a")], [[3.0, 0.0, 0.0]], ingest_run_id=RUN_ID, embedded_model=MODEL
        )
        assert store.search_dense([0.1, 0.0, 0.0], limit=1)[0].score == pytest.approx(1.0)

    def test_a_zero_vector_scores_zero_instead_of_dividing_by_zero(self) -> None:
        store = _store()
        store.upsert_chunks(
            [_chunk(0, "a")], [[0.0, 0.0, 0.0]], ingest_run_id=RUN_ID, embedded_model=MODEL
        )
        assert store.search_dense([1.0, 0.0, 0.0], limit=1)[0].score == 0.0

    def test_limit_truncates_the_candidate_list(self) -> None:
        store = _store()
        store.upsert_chunks(
            [_chunk(0, "a"), _chunk(1, "b")],
            [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]],
            ingest_run_id=RUN_ID,
            embedded_model=MODEL,
        )
        assert len(store.search_dense([1.0, 0.0, 0.0], limit=1)) == 1

    def test_a_query_of_the_wrong_width_is_refused(self) -> None:
        # Silently scoring against a differently sized index would produce
        # plausible-looking nonsense.
        with pytest.raises(CollectionMismatchError, match="query vector of length"):
            _store().search_dense([1.0, 0.0], limit=1)

    @pytest.mark.parametrize("limit", [0, -1])
    def test_a_non_positive_limit_is_refused(self, limit: int) -> None:
        with pytest.raises(VectorStoreError, match="limit must be positive"):
            _store().search_dense([1.0, 0.0, 0.0], limit=limit)

    def test_payload_projection_keeps_only_the_named_fields(self) -> None:
        store = _store()
        store.upsert_chunks(
            [_chunk(0, "a")], [[1.0, 0.0, 0.0]], ingest_run_id=RUN_ID, embedded_model=MODEL
        )
        hit = store.search_dense([1.0, 0.0, 0.0], limit=1, with_payload=["chunk_id", "absent"])[0]
        assert set(hit.payload) == {"chunk_id"}
        assert hit.chunk_id == hit.payload["chunk_id"]

    def test_no_projection_returns_the_whole_payload(self) -> None:
        store = _store()
        store.upsert_chunks(
            [_chunk(0, "a")], [[1.0, 0.0, 0.0]], ingest_run_id=RUN_ID, embedded_model=MODEL
        )
        assert "source_path" in store.search_dense([1.0, 0.0, 0.0], limit=1)[0].payload


class TestSparseSearch:
    @staticmethod
    def _indexed() -> tuple[InMemoryVectorStore, Bm25Encoder, list[Chunk]]:
        texts = [
            "qdrant stores dense and sparse vectors",
            "reciprocal rank fusion combines ranked lists",
            "qdrant supports named vectors in one collection",
        ]
        encoder = Bm25Encoder()
        encoder.fit(texts)
        chunks = [_chunk(index, text) for index, text in enumerate(texts)]
        store = _store()
        store.upsert_chunks(
            chunks,
            [[float(index), 0.0, 0.0] for index in range(len(texts))],
            ingest_run_id=RUN_ID,
            embedded_model=MODEL,
            sparse_vectors=encoder.encode_documents(texts),
        )
        return store, encoder, chunks

    def test_matches_a_literal_term(self) -> None:
        store, encoder, chunks = self._indexed()
        hits = store.search_sparse(encoder.encode_query("fusion"), limit=5)
        assert [hit.point_id for hit in hits] == [chunks[1].point_id]

    def test_scores_are_bm25_dot_products(self) -> None:
        store, encoder, _ = self._indexed()
        query = encoder.encode_query("qdrant")
        hits = store.search_sparse(query, limit=5)
        expected = sorted(
            (query.dot(vector) for vector in store.sparse_vectors.values()), reverse=True
        )[: len(hits)]
        assert [hit.score for hit in hits] == pytest.approx(expected)

    def test_documents_sharing_no_term_are_not_returned(self) -> None:
        # Padding the candidate list with zero-score documents would only give
        # fusion noise to rank.
        store, encoder, _ = self._indexed()
        assert store.search_sparse(encoder.encode_query("kubernetes"), limit=5) == []

    def test_an_empty_query_vector_returns_nothing(self) -> None:
        store, encoder, _ = self._indexed()
        assert store.search_sparse(encoder.encode_query("the and of"), limit=5) == []

    def test_limit_truncates_the_candidate_list(self) -> None:
        store, encoder, _ = self._indexed()
        assert len(store.search_sparse(encoder.encode_query("qdrant"), limit=1)) == 1

    def test_a_dense_only_collection_has_nothing_sparse_to_find(self) -> None:
        store = _store(with_sparse=False)
        store.upsert_chunks(
            [_chunk(0, "qdrant")], [[1.0, 0.0, 0.0]], ingest_run_id=RUN_ID, embedded_model=MODEL
        )
        assert store.search_sparse(Bm25Encoder().encode_query("qdrant"), limit=5) == []

    @pytest.mark.parametrize("limit", [0, -1])
    def test_a_non_positive_limit_is_refused(self, limit: int) -> None:
        with pytest.raises(VectorStoreError, match="limit must be positive"):
            _store().search_sparse(SparseVector.from_weights({1: 1.0}), limit=limit)


class TestDeterminism:
    def test_equal_scores_break_on_the_point_id(self) -> None:
        # Qdrant does not specify an order among equal scores; a test that leans on
        # insertion order would fail later for the wrong reason.
        store = _store()
        chunks = [_chunk(index, f"text {index}") for index in range(3)]
        store.upsert_chunks(
            chunks,
            [[1.0, 0.0, 0.0]] * 3,
            ingest_run_id=RUN_ID,
            embedded_model=MODEL,
        )
        hits = store.search_dense([1.0, 0.0, 0.0], limit=3)
        assert [hit.point_id for hit in hits] == sorted(chunk.point_id for chunk in chunks)
