"""The retriever: modes, fusion wiring, candidate depth, and explainability.

Everything here runs against :class:`InMemoryVectorStore` and the fake embedder,
so the whole retrieval path is exercised with no network and no container.
"""

from __future__ import annotations

import pytest

from production_rag.config_loader import (
    FusionConfig,
    RetrievalConfig,
    SparseConfig,
    YamlConfig,
)
from production_rag.ingest.models import Chunk, Document
from production_rag.retrieval.embeddings import FakeEmbeddingProvider
from production_rag.retrieval.hybrid import (
    MODE_DENSE,
    MODE_HYBRID,
    MODE_SPARSE,
    RETRIEVAL_MODES,
    RetrievalError,
    RetrievalHit,
    Retriever,
)
from production_rag.retrieval.sparse import Bm25Encoder
from production_rag.retrieval.store import InMemoryVectorStore

RUN_ID = "run-test"

CORPUS = {
    "sample/00-intro.md": "Production RAG systems combine retrieval and generation.",
    "sample/01-qdrant.md": "Qdrant stores dense and sparse vectors in one collection.",
    "sample/02-fusion.md": "Reciprocal rank fusion merges two ranked lists by position.",
    "sample/03-chunking.md": "The chunker splits markdown documents on heading boundaries.",
    "sample/04-flags.md": "Pass --recreate-collection to rebuild the index from scratch.",
}


def _chunks() -> list[Chunk]:
    chunks = []
    for index, (path, text) in enumerate(CORPUS.items()):
        document = Document(source_path=path, text=text, title=path, source="sample")
        chunks.append(Chunk.build(document=document, chunk_index=index, text=text, embed_text=text))
    return chunks


def _indexed(*, with_sparse: bool = True) -> tuple[InMemoryVectorStore, FakeEmbeddingProvider]:
    embedder = FakeEmbeddingProvider()
    chunks = _chunks()
    texts = [chunk.embed_text for chunk in chunks]
    store = InMemoryVectorStore()
    store.ensure_collection(vector_size=embedder.dimensions, with_sparse=with_sparse)
    sparse_vectors = None
    if with_sparse:
        encoder = Bm25Encoder()
        encoder.fit(texts)
        sparse_vectors = encoder.encode_documents(texts)
    store.upsert_chunks(
        chunks,
        embedder.embed_documents(texts),
        ingest_run_id=RUN_ID,
        embedded_model=embedder.model,
        sparse_vectors=sparse_vectors,
    )
    return store, embedder


def _retriever(**overrides: object) -> Retriever:
    store, embedder = _indexed()
    config = RetrievalConfig(**overrides)  # type: ignore[arg-type]
    return Retriever(store=store, embedder=embedder, config=config)


class TestModes:
    def test_hybrid_is_the_default(self) -> None:
        assert RetrievalConfig().mode == MODE_HYBRID
        assert _retriever().retrieve("qdrant vectors").mode == MODE_HYBRID

    @pytest.mark.parametrize("mode", RETRIEVAL_MODES)
    def test_every_mode_returns_hits(self, mode: str) -> None:
        result = _retriever().retrieve("qdrant vectors", mode=mode)
        assert result.hits
        assert result.mode == mode

    def test_dense_mode_never_touches_the_sparse_branch(self) -> None:
        result = _retriever().retrieve("qdrant vectors", mode=MODE_DENSE)
        assert result.sparse_candidates == 0
        assert all(hit.branches == ("dense",) for hit in result.hits)

    def test_sparse_mode_never_touches_the_dense_branch(self) -> None:
        result = _retriever().retrieve("qdrant vectors", mode=MODE_SPARSE)
        assert result.dense_candidates == 0
        assert all(hit.branches == ("sparse",) for hit in result.hits)

    def test_mode_is_case_insensitive(self) -> None:
        assert _retriever().retrieve("qdrant", mode="HYBRID").mode == MODE_HYBRID

    def test_an_unknown_mode_names_the_valid_ones(self) -> None:
        with pytest.raises(RetrievalError, match="unknown retrieval mode"):
            _retriever().retrieve("qdrant", mode="rerank")


class TestSparseBehaviour:
    def test_a_literal_flag_is_found_by_the_sparse_branch(self) -> None:
        # The reason hybrid exists: the fake embedder has no semantics at all, so
        # only BM25 can put this document first.
        result = _retriever().retrieve("--recreate-collection", mode=MODE_SPARSE)
        assert result.hits[0].source_path == "sample/04-flags.md"

    def test_the_sparse_branch_survives_fusion(self) -> None:
        result = _retriever().retrieve("--recreate-collection", mode=MODE_HYBRID)
        assert result.hits[0].source_path == "sample/04-flags.md"
        assert "sparse" in result.hits[0].branches

    def test_query_terms_are_reported(self) -> None:
        assert _retriever().retrieve("qdrant vectors", mode=MODE_SPARSE).sparse_query_terms == 2

    def test_an_all_stopword_query_reports_zero_terms_and_no_sparse_hits(self) -> None:
        result = _retriever().retrieve("the and of", mode=MODE_SPARSE)
        assert result.sparse_query_terms == 0
        assert result.sparse_candidates == 0
        assert result.hits == ()

    def test_hybrid_still_answers_when_the_sparse_query_is_empty(self) -> None:
        # A stopword-only query halves a hybrid search; it must not empty it.
        result = _retriever().retrieve("the and of", mode=MODE_HYBRID)
        assert result.sparse_candidates == 0
        assert result.hits

    def test_a_dense_only_collection_degrades_instead_of_failing(self) -> None:
        store, embedder = _indexed(with_sparse=False)
        result = Retriever(store=store, embedder=embedder).retrieve("qdrant", mode=MODE_HYBRID)
        assert result.sparse_candidates == 0
        assert result.hits


class TestCandidateDepth:
    def test_top_k_bounds_the_returned_hits(self) -> None:
        assert len(_retriever(top_k=2).retrieve("qdrant vectors").hits) == 2

    def test_top_k_can_be_overridden_per_call(self) -> None:
        assert len(_retriever(top_k=5).retrieve("qdrant vectors", top_k=1).hits) == 1

    def test_candidate_depth_is_independent_of_top_k(self) -> None:
        # Under-retrieving per branch is a recall loss fusion cannot recover, so
        # the branches stay deep even when only one hit is wanted.
        result = _retriever(dense_top_k=4, top_k=1).retrieve("qdrant vectors")
        assert result.dense_candidates == 4
        assert len(result.hits) == 1

    def test_branch_limits_are_applied_separately(self) -> None:
        result = _retriever(dense_top_k=2, sparse_top_k=1, top_k=10).retrieve("qdrant vectors")
        assert result.dense_candidates == 2
        assert result.sparse_candidates == 1

    def test_a_non_positive_top_k_is_refused(self) -> None:
        with pytest.raises(RetrievalError, match="top_k must be positive"):
            _retriever().retrieve("qdrant", top_k=0)

    def test_ranks_are_one_based_and_contiguous(self) -> None:
        hits = _retriever(top_k=3).retrieve("qdrant vectors").hits
        assert [hit.rank for hit in hits] == [1, 2, 3]


class TestWeightsAndThreshold:
    def test_configured_weights_reach_fusion(self) -> None:
        result = _retriever(fusion=FusionConfig(dense_weight=2.0, sparse_weight=0.5)).retrieve(
            "qdrant vectors"
        )
        assert result.weights == {"dense": 2.0, "sparse": 0.5}

    def test_zero_sparse_weight_reduces_hybrid_to_the_dense_order(self) -> None:
        # An ablation must be a config change, and it must be exact.
        store, embedder = _indexed()
        ablated = Retriever(
            store=store,
            embedder=embedder,
            config=RetrievalConfig(fusion=FusionConfig(sparse_weight=0.0)),
        ).retrieve("--recreate-collection")
        dense_only = Retriever(store=store, embedder=embedder).retrieve(
            "--recreate-collection", mode=MODE_DENSE
        )
        assert [hit.chunk_id for hit in ablated.hits] == [hit.chunk_id for hit in dense_only.hits]

    def test_a_single_branch_mode_uses_weight_one(self) -> None:
        result = _retriever(fusion=FusionConfig(dense_weight=3.0)).retrieve(
            "qdrant", mode=MODE_DENSE
        )
        assert result.weights == {"dense": 1.0}

    def test_the_threshold_drops_hits_and_counts_them(self) -> None:
        result = _retriever(score_threshold=0.02, top_k=10).retrieve("qdrant vectors")
        assert result.dropped_below_threshold > 0
        assert all(hit.score >= 0.02 for hit in result.hits)

    def test_a_threshold_above_everything_returns_an_honest_empty_result(self) -> None:
        result = _retriever(score_threshold=1.0).retrieve("qdrant vectors")
        assert result.hits == ()
        assert result.dropped_below_threshold > 0

    def test_zero_threshold_keeps_every_fused_hit(self) -> None:
        result = _retriever(score_threshold=0.0, top_k=10).retrieve("qdrant vectors")
        assert result.dropped_below_threshold == 0

    def test_fusion_k_is_reported(self) -> None:
        assert _retriever(fusion=FusionConfig(k=10)).retrieve("qdrant").fusion_k == 10


class TestExplainability:
    def test_a_hit_carries_its_citation_fields(self) -> None:
        hit = _retriever().retrieve("qdrant vectors").hits[0]
        assert hit.source_path in CORPUS
        assert hit.text == CORPUS[hit.source_path]
        assert hit.chunk_id
        assert hit.point_id

    def test_a_hybrid_hit_explains_both_branches(self) -> None:
        result = _retriever().retrieve("qdrant vectors")
        agreed = next(hit for hit in result.hits if len(hit.branches) == 2)
        assert set(agreed.branch_ranks) == {"dense", "sparse"}
        assert set(agreed.branch_scores) == {"dense", "sparse"}
        assert all(rank >= 1 for rank in agreed.branch_ranks.values())

    def test_branch_scores_are_the_raw_store_scores(self) -> None:
        store, embedder = _indexed()
        retriever = Retriever(store=store, embedder=embedder)
        hit = retriever.retrieve("qdrant vectors", mode=MODE_SPARSE).hits[0]
        expected = store.search_sparse(Bm25Encoder().encode_query("qdrant vectors"), limit=40)[
            0
        ].score
        assert hit.branch_scores["sparse"] == pytest.approx(expected)

    def test_to_dict_is_json_ready(self) -> None:
        payload = _retriever().retrieve("qdrant vectors").hits[0].to_dict()
        assert set(payload) == {
            "rank",
            "score",
            "chunk_id",
            "source_path",
            "title",
            "heading",
            "heading_path",
            "point_id",
            "branches",
            "branch_ranks",
            "branch_scores",
            "text",
        }

    def test_summary_records_what_produced_the_numbers(self) -> None:
        summary = _retriever().retrieve("qdrant vectors").to_summary()
        assert summary["ok"] is True
        assert summary["mode"] == MODE_HYBRID
        assert summary["returned"] == len(summary["hits"])
        for key in ("fusion_k", "weights", "dense_candidates", "sparse_candidates"):
            assert key in summary

    def test_a_missing_payload_field_degrades_to_empty_not_a_crash(self) -> None:
        # return_payload_fields is configurable; trimming it must not break the
        # retriever, only the citation detail.
        result = _retriever(return_payload_fields=("chunk_id",)).retrieve("qdrant")
        assert result.hits[0].chunk_id
        assert result.hits[0].text == ""
        assert result.hits[0].title is None


class TestConstruction:
    def test_from_config_shares_the_ingest_tokenizer(self) -> None:
        # A query tokenised differently from the corpus is the classic lexical
        # search bug, so both sides must read one setting.
        store, embedder = _indexed()
        config = YamlConfig(ingest={"sparse": SparseConfig(stopwords="none")})  # type: ignore[arg-type]
        retriever = Retriever.from_config(store=store, embedder=embedder, config=config)
        assert retriever.retrieve("the collection", mode=MODE_SPARSE).sparse_query_terms == 2

    def test_from_config_carries_the_retrieval_block(self) -> None:
        store, embedder = _indexed()
        config = YamlConfig(retrieval=RetrievalConfig(top_k=1, mode=MODE_SPARSE))
        retriever = Retriever.from_config(store=store, embedder=embedder, config=config)
        assert retriever.config.top_k == 1
        assert retriever.retrieve("qdrant").mode == MODE_SPARSE

    @pytest.mark.parametrize("query", ["", "   ", "\n"])
    def test_an_empty_query_is_refused(self, query: str) -> None:
        with pytest.raises(RetrievalError, match="must not be empty"):
            _retriever().retrieve(query)

    def test_the_query_is_reported_stripped(self) -> None:
        assert _retriever().retrieve("  qdrant  ").query == "qdrant"

    def test_an_empty_collection_returns_no_hits(self) -> None:
        embedder = FakeEmbeddingProvider()
        store = InMemoryVectorStore()
        store.ensure_collection(vector_size=embedder.dimensions, with_sparse=True)
        result = Retriever(store=store, embedder=embedder).retrieve("qdrant")
        assert result.hits == ()
        assert result.to_summary()["returned"] == 0


def test_retrieval_hit_defaults_are_safe_to_construct() -> None:
    hit = RetrievalHit(chunk_id="c", source_path="p", text="t", score=1.0, rank=1)
    assert hit.branches == ()
    assert hit.branch_ranks == {}
