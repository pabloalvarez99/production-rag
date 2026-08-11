"""Reciprocal rank fusion: arithmetic, determinism, and the ablation seam."""

from __future__ import annotations

import pytest

from production_rag.retrieval.rrf import (
    DEFAULT_RRF_K,
    rank_keys,
    reciprocal_rank_fusion,
)


def test_single_branch_preserves_its_order() -> None:
    fused = reciprocal_rank_fusion({"dense": ["a", "b", "c"]})
    assert [hit.key for hit in fused] == ["a", "b", "c"]


def test_score_matches_the_published_formula() -> None:
    fused = reciprocal_rank_fusion({"dense": ["a"]}, k=60)
    assert fused[0].score == pytest.approx(1 / 61)


def test_ranks_are_one_based() -> None:
    # 0-based ranks would silently change what the constant k means relative to
    # the paper and to configs/default.yaml.
    fused = reciprocal_rank_fusion({"dense": ["a", "b"]}, k=10)
    assert fused[0].ranks == {"dense": 1}
    assert fused[0].score == pytest.approx(1 / 11)
    assert fused[1].score == pytest.approx(1 / 12)


def test_agreement_between_branches_beats_a_single_first_place() -> None:
    # The property that makes hybrid retrieval work: a document both retrievers
    # like outranks one that only one retriever loves.
    fused = reciprocal_rank_fusion(
        {"dense": ["both", "dense_only"], "sparse": ["sparse_only", "both"]}, k=60
    )
    assert fused[0].key == "both"


def test_a_document_found_by_one_branch_still_appears() -> None:
    fused = reciprocal_rank_fusion({"dense": ["a"], "sparse": ["b"]})
    assert {hit.key for hit in fused} == {"a", "b"}


def test_contributions_explain_the_position() -> None:
    fused = reciprocal_rank_fusion({"dense": ["x"], "sparse": ["y", "x"]}, k=60)
    hit = next(item for item in fused if item.key == "x")
    assert hit.ranks == {"dense": 1, "sparse": 2}
    assert hit.branches == ("dense", "sparse")
    assert sum(hit.contributions.values()) == pytest.approx(hit.score)


def test_weights_scale_a_branch() -> None:
    fused = reciprocal_rank_fusion(
        {"dense": ["d"], "sparse": ["s"]}, weights={"dense": 2.0, "sparse": 1.0}
    )
    assert [hit.key for hit in fused] == ["d", "s"]
    assert fused[0].score == pytest.approx(2 / 61)


def test_zero_weight_disables_a_branch_without_hiding_it() -> None:
    # Ablations ("what does dense alone give us?") must be a config change, and
    # the disabled branch must still be visible in the explanation.
    fused = reciprocal_rank_fusion({"dense": ["a"], "sparse": ["b"]}, weights={"sparse": 0.0})
    assert [hit.key for hit in fused] == ["a", "b"]
    sparse_hit = next(hit for hit in fused if hit.key == "b")
    assert sparse_hit.score == 0.0
    assert sparse_hit.ranks == {"sparse": 1}


def test_a_branch_cannot_vote_twice_for_the_same_document() -> None:
    fused = reciprocal_rank_fusion({"dense": ["a", "a", "b"]}, k=60)
    assert [hit.key for hit in fused] == ["a", "b"]
    assert fused[0].score == pytest.approx(1 / 61)
    # The duplicate must not shift what follows it either.
    assert fused[1].score == pytest.approx(1 / 62)


def test_ties_break_on_the_key_for_determinism() -> None:
    first = reciprocal_rank_fusion({"dense": ["b"], "sparse": ["a"]})
    second = reciprocal_rank_fusion({"sparse": ["a"], "dense": ["b"]})
    assert [hit.key for hit in first] == [hit.key for hit in second] == ["a", "b"]


def test_limit_truncates_after_fusion_not_before() -> None:
    fused = reciprocal_rank_fusion({"dense": ["a", "b", "c"], "sparse": ["c", "b"]}, limit=2)
    # "a" leads the dense branch, so a pre-fusion cut would have kept it. Only
    # the fused scores drop it: "b" and "c" are in both branches, "a" in one.
    assert [hit.key for hit in fused] == ["c", "b"]


def test_empty_input_fuses_to_nothing() -> None:
    assert reciprocal_rank_fusion({}) == []
    assert reciprocal_rank_fusion({"dense": [], "sparse": []}) == []


def test_default_k_is_the_published_constant() -> None:
    assert DEFAULT_RRF_K == 60


@pytest.mark.parametrize("bad_k", [0, -1])
def test_non_positive_k_is_rejected(bad_k: int) -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        reciprocal_rank_fusion({"dense": ["a"]}, k=bad_k)


def test_negative_limit_is_rejected() -> None:
    with pytest.raises(ValueError, match="limit"):
        reciprocal_rank_fusion({"dense": ["a"]}, limit=-1)


def test_rank_keys_sorts_by_score_descending() -> None:
    assert rank_keys([("a", 0.1), ("b", 0.9), ("c", 0.5)]) == ["b", "c", "a"]


def test_rank_keys_breaks_score_ties_on_the_key() -> None:
    assert rank_keys([("b", 0.5), ("a", 0.5)]) == ["a", "b"]
