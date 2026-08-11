"""Unit tests for the rerank stage. Offline: no model download, no network."""

from __future__ import annotations

from typing import Any

import pytest

from production_rag.config_loader import RerankConfig
from production_rag.retrieval.hybrid import RetrievalHit
from production_rag.retrieval.rerank import (
    RERANK_KINDS,
    CohereReranker,
    FakeReranker,
    LocalCrossEncoderReranker,
    Reranker,
    RerankError,
    apply_rerank,
    build_reranker,
)


def make_hit(chunk_id: str, text: str, rank: int, score: float = 0.5) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        source_path=f"data/raw/{chunk_id}.md",
        text=text,
        score=score,
        rank=rank,
        point_id=f"point-{chunk_id}",
        branches=("dense", "sparse"),
        branch_ranks={"dense": rank, "sparse": rank},
        branch_scores={"dense": score, "sparse": score},
    )


class ExplodingReranker:
    """A reranker whose scoring always fails, to exercise the failure policy."""

    @property
    def name(self) -> str:
        return "exploding"

    def rerank(self, query: str, hits: Any, *, top_n: int) -> list[RetrievalHit]:
        raise RerankError("upstream refused the request")


class ReversingReranker:
    """Reverses the shortlist, so a test can see reranking took effect."""

    @property
    def name(self) -> str:
        return "reversing"

    def rerank(self, query: str, hits: Any, *, top_n: int) -> list[RetrievalHit]:
        from dataclasses import replace

        ordered = list(reversed(list(hits)))[:top_n]
        return [
            replace(hit, rank=index, rerank_score=1.0 / index, pre_rerank_rank=hit.rank)
            for index, hit in enumerate(ordered, start=1)
        ]


class TestFakeReranker:
    def test_satisfies_the_reranker_protocol(self) -> None:
        assert isinstance(FakeReranker(), Reranker)

    def test_promotes_the_hit_covering_more_query_terms(self) -> None:
        hits = [
            make_hit("a", "Qdrant stores payloads on disk.", rank=1),
            make_hit("b", "The hnsw ef parameter trades latency for recall.", rank=2),
        ]
        reranked = FakeReranker().rerank("hnsw ef recall", hits, top_n=2)
        assert [hit.chunk_id for hit in reranked] == ["b", "a"]

    def test_records_the_fusion_position_it_moved_a_hit_from(self) -> None:
        hits = [
            make_hit("a", "nothing relevant here", rank=1),
            make_hit("b", "sparse vectors and bm25 weights", rank=2),
        ]
        top = FakeReranker().rerank("bm25 weights", hits, top_n=1)[0]
        assert top.chunk_id == "b"
        assert top.rank == 1
        assert top.pre_rerank_rank == 2
        assert top.rerank_score == pytest.approx(1.0)

    def test_truncates_to_top_n(self) -> None:
        hits = [make_hit(str(index), "bm25", rank=index) for index in range(1, 6)]
        assert len(FakeReranker().rerank("bm25", hits, top_n=2)) == 2

    def test_renumbers_rank_from_one(self) -> None:
        hits = [
            make_hit("a", "unrelated", rank=1),
            make_hit("b", "chunking overlap", rank=2),
            make_hit("c", "chunking", rank=3),
        ]
        reranked = FakeReranker().rerank("chunking overlap", hits, top_n=3)
        assert [hit.rank for hit in reranked] == [1, 2, 3]

    def test_ties_keep_the_fusion_order(self) -> None:
        # Two hits the reranker cannot separate must not be shuffled: an eval
        # number depends on the ranking being deterministic.
        hits = [
            make_hit("a", "identical wording", rank=2),
            make_hit("b", "identical wording", rank=1),
        ]
        reranked = FakeReranker().rerank("identical wording", hits, top_n=2)
        assert [hit.chunk_id for hit in reranked] == ["b", "a"]

    def test_a_stopword_only_query_preserves_the_fusion_order(self) -> None:
        hits = [make_hit("a", "alpha", rank=1), make_hit("b", "beta", rank=2)]
        reranked = FakeReranker().rerank("the and of", hits, top_n=2)
        assert [hit.chunk_id for hit in reranked] == ["a", "b"]
        assert all(hit.rerank_score == 0.0 for hit in reranked)

    def test_empty_shortlist_returns_empty(self) -> None:
        assert FakeReranker().rerank("anything", [], top_n=5) == []

    def test_rejects_a_non_positive_cut(self) -> None:
        with pytest.raises(RerankError, match="top_n must be positive"):
            FakeReranker().rerank("bm25", [make_hit("a", "bm25", rank=1)], top_n=0)

    def test_is_deterministic_across_calls(self) -> None:
        hits = [make_hit(str(index), f"bm25 term {index}", rank=index) for index in range(1, 5)]
        first = [hit.chunk_id for hit in FakeReranker().rerank("bm25 term", hits, top_n=3)]
        second = [hit.chunk_id for hit in FakeReranker().rerank("bm25 term", hits, top_n=3)]
        assert first == second


class TestApplyRerank:
    def test_without_a_reranker_it_truncates_and_reports_not_applied(self) -> None:
        hits = [make_hit(str(index), "text", rank=index) for index in range(1, 6)]
        outcome = apply_rerank(None, "query", hits, top_n=2)
        assert outcome.applied is False
        assert outcome.reranker is None
        assert [hit.chunk_id for hit in outcome.hits] == ["1", "2"]

    def test_reports_the_reranker_and_the_candidate_depth(self) -> None:
        hits = [make_hit(str(index), "text", rank=index) for index in range(1, 5)]
        outcome = apply_rerank(ReversingReranker(), "query", hits, top_n=2)
        assert outcome.applied is True
        assert outcome.reranker == "reversing"
        assert outcome.candidates == 4
        assert [hit.chunk_id for hit in outcome.hits] == ["4", "3"]

    def test_fail_open_degrades_to_the_fusion_order_truncated(self) -> None:
        hits = [make_hit(str(index), "text", rank=index) for index in range(1, 6)]
        outcome = apply_rerank(ExplodingReranker(), "query", hits, top_n=3, fail_open=True)
        assert outcome.applied is False
        assert [hit.chunk_id for hit in outcome.hits] == ["1", "2", "3"]
        # The failure is carried in the response, not only in a log line.
        assert outcome.error is not None
        assert "upstream refused" in outcome.error
        assert outcome.reranker == "exploding"

    def test_fail_closed_raises(self) -> None:
        hits = [make_hit("a", "text", rank=1)]
        with pytest.raises(RerankError, match="upstream refused"):
            apply_rerank(ExplodingReranker(), "query", hits, top_n=1, fail_open=False)

    def test_summary_shape_is_stable(self) -> None:
        outcome = apply_rerank(None, "query", [make_hit("a", "t", rank=1)], top_n=1)
        assert outcome.as_dict() == {
            "applied": False,
            "reranker": None,
            "candidates": 1,
            "error": None,
        }


class TestBuildReranker:
    def test_off_is_the_default_and_returns_nothing(self) -> None:
        assert build_reranker() is None
        assert build_reranker("off") is None

    def test_fake_is_offline_and_needs_no_config(self) -> None:
        assert isinstance(build_reranker("fake"), FakeReranker)

    def test_local_is_constructed_without_loading_weights(self) -> None:
        # Construction must stay free: a CLI that fails argument parsing should not
        # first pay for a model download.
        config = RerankConfig(local_model="BAAI/bge-reranker-base")
        reranker = build_reranker("local", config=config)
        assert isinstance(reranker, LocalCrossEncoderReranker)
        assert reranker.name == "BAAI/bge-reranker-base"

    def test_cohere_without_a_key_is_a_clear_error(self) -> None:
        with pytest.raises(RerankError, match="needs an API key"):
            build_reranker("cohere")

    def test_auto_returns_nothing_when_rerank_is_disabled(self) -> None:
        assert build_reranker("auto", config=RerankConfig(enabled=False)) is None

    def test_auto_follows_the_configured_provider(self) -> None:
        config = RerankConfig(enabled=True, provider="local-cross-encoder")
        assert isinstance(build_reranker("auto", config=config), LocalCrossEncoderReranker)

    def test_auto_with_provider_none_returns_nothing(self) -> None:
        assert build_reranker("auto", config=RerankConfig(enabled=True, provider="none")) is None

    def test_auto_with_an_unknown_provider_is_an_error(self) -> None:
        config = RerankConfig(enabled=True, provider="magic")
        with pytest.raises(RerankError, match=r"unknown rerank\.provider"):
            build_reranker("auto", config=config)

    def test_an_unknown_kind_names_the_valid_ones(self) -> None:
        with pytest.raises(RerankError, match="unknown reranker"):
            build_reranker("cross-encoder")

    @pytest.mark.parametrize("kind", RERANK_KINDS)
    def test_every_advertised_kind_is_constructible_or_explains_itself(self, kind: str) -> None:
        # No advertised choice may fail obscurely. Cohere is the one that can
        # refuse here, and only for the documented reason.
        if kind == "cohere":
            with pytest.raises(RerankError, match="needs an API key"):
                build_reranker(kind)
        else:
            build_reranker(kind)


class FakeCrossEncoder:
    """Stand-in for sentence-transformers' CrossEncoder. No weights, no network."""

    def __init__(self, scores: list[float] | None = None, fail: bool = False) -> None:
        self.scores = scores
        self.fail = fail
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs: list[tuple[str, str]], batch_size: int = 32) -> list[float]:
        if self.fail:
            raise RuntimeError("CUDA out of memory")
        self.calls.append(pairs)
        return self.scores if self.scores is not None else [0.0] * len(pairs)


class TestLocalCrossEncoderReranker:
    def test_satisfies_the_reranker_protocol(self) -> None:
        assert isinstance(LocalCrossEncoderReranker(), Reranker)

    def test_orders_by_the_model_score(self) -> None:
        client = FakeCrossEncoder(scores=[0.1, 0.9, 0.4])
        reranker = LocalCrossEncoderReranker(client=client)
        hits = [make_hit(name, "text", rank=index) for index, name in enumerate("abc", start=1)]
        reranked = reranker.rerank("query", hits, top_n=3)
        assert [hit.chunk_id for hit in reranked] == ["b", "c", "a"]
        assert reranked[0].rerank_score == pytest.approx(0.9)
        assert reranked[0].pre_rerank_rank == 2

    def test_sends_query_and_chunk_text_as_pairs(self) -> None:
        client = FakeCrossEncoder(scores=[1.0])
        LocalCrossEncoderReranker(client=client).rerank(
            "what is rrf", [make_hit("a", "reciprocal rank fusion", rank=1)], top_n=1
        )
        assert client.calls == [[("what is rrf", "reciprocal rank fusion")]]

    def test_a_scorer_failure_becomes_a_rerank_error(self) -> None:
        reranker = LocalCrossEncoderReranker(client=FakeCrossEncoder(fail=True))
        with pytest.raises(RerankError, match="failed to score"):
            reranker.rerank("query", [make_hit("a", "text", rank=1)], top_n=1)

    def test_a_score_count_mismatch_is_refused_not_zipped(self) -> None:
        # Silently pairing the wrong score with a passage is worse than an error.
        reranker = LocalCrossEncoderReranker(client=FakeCrossEncoder(scores=[0.5]))
        hits = [make_hit("a", "t", rank=1), make_hit("b", "t", rank=2)]
        with pytest.raises(RerankError, match="1 scores for 2 hits"):
            reranker.rerank("query", hits, top_n=2)

    def test_an_empty_shortlist_never_touches_the_model(self) -> None:
        client = FakeCrossEncoder(fail=True)
        assert LocalCrossEncoderReranker(client=client).rerank("query", [], top_n=3) == []

    def test_a_missing_dependency_names_the_extra_to_install(self, monkeypatch: Any) -> None:
        import builtins

        real_import = builtins.__import__

        def blocked(name: str, *args: Any, **kwargs: Any) -> Any:
            if name.startswith("sentence_transformers"):
                raise ImportError("No module named 'sentence_transformers'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)
        with pytest.raises(RerankError, match=r"sentence-transformers"):
            LocalCrossEncoderReranker().rerank("query", [make_hit("a", "t", rank=1)], top_n=1)


class FakeCohereResult:
    def __init__(self, index: int, relevance_score: float) -> None:
        self.index = index
        self.relevance_score = relevance_score


class FakeCohereResponse:
    def __init__(self, results: list[FakeCohereResult]) -> None:
        self.results = results


class FakeCohereClient:
    def __init__(self, results: list[FakeCohereResult]) -> None:
        self._results = results
        self.kwargs: dict[str, Any] = {}

    def rerank(self, **kwargs: Any) -> FakeCohereResponse:
        self.kwargs = kwargs
        return FakeCohereResponse(self._results)


class TestCohereReranker:
    def test_orders_by_the_returned_relevance(self) -> None:
        client = FakeCohereClient([FakeCohereResult(1, 0.9), FakeCohereResult(0, 0.2)])
        reranker = CohereReranker(api_key="", client=client)
        hits = [make_hit("a", "t", rank=1), make_hit("b", "t", rank=2)]
        reranked = reranker.rerank("query", hits, top_n=2)
        assert [hit.chunk_id for hit in reranked] == ["b", "a"]
        assert client.kwargs["top_n"] == 2

    def test_an_out_of_range_index_is_refused(self) -> None:
        client = FakeCohereClient([FakeCohereResult(7, 0.9)])
        reranker = CohereReranker(api_key="", client=client)
        with pytest.raises(RerankError, match="index 7"):
            reranker.rerank("query", [make_hit("a", "t", rank=1)], top_n=1)

    def test_no_key_and_no_client_is_a_construction_error(self) -> None:
        with pytest.raises(RerankError, match="needs an API key"):
            CohereReranker(api_key="")
