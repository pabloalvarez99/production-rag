"""The retriever: one query in, a fused, explainable ranking out.

    query -> dense branch  (embedding    -> cosine    -> ranks)
          -> sparse branch (BM25 encoder -> dot product -> ranks)
          -> RRF fusion    -> top_k RetrievalHit

Three modes — ``dense``, ``sparse``, ``hybrid`` — because a hybrid number means
nothing without the two single-branch numbers to compare it against. Running an
ablation is therefore a flag, not a code change, and the same object serves the
CLI, the eval harness and any later generation stage.

**Every hit explains its own position.** ``scores`` carries the fused score, the
per-branch ranks and the per-branch raw scores. "This chunk is second because
BM25 ranked it first and dense ranked it fourteenth" is a debuggable statement;
a bare similarity number is not.

The retriever holds no corpus state. The dense side needs the embedder that built
the index; the sparse side needs only a tokenizer, because ingest stored full BM25
weights and a query term weighs 1.0 (see :mod:`production_rag.retrieval.sparse`).
So a fresh process can query a collection it did not build, with no warm-up.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import structlog

from production_rag.config_loader import RerankConfig, RetrievalConfig, YamlConfig
from production_rag.retrieval.embeddings import EmbeddingProvider
from production_rag.retrieval.filters import (
    NO_FILTER_SUMMARY,
    FilterPolicy,
    QueryFilter,
)
from production_rag.retrieval.rerank import Reranker, RerankOutcome, apply_rerank
from production_rag.retrieval.rrf import rank_keys, reciprocal_rank_fusion
from production_rag.retrieval.sparse import SparseEncoder, build_sparse_encoder
from production_rag.retrieval.store import SearchableVectorStore, SearchHit

_log = structlog.get_logger(__name__)

DENSE_BRANCH = "dense"
SPARSE_BRANCH = "sparse"

MODE_DENSE = "dense"
MODE_SPARSE = "sparse"
MODE_HYBRID = "hybrid"
RETRIEVAL_MODES = (MODE_DENSE, MODE_SPARSE, MODE_HYBRID)
"""The modes a caller may ask for. ``hybrid`` is the default; the other two exist
so a single-branch baseline is measurable."""

NO_RERANK_SUMMARY: dict[str, Any] = {
    "applied": False,
    "reranker": None,
    "candidates": 0,
    "error": None,
}
"""Summary emitted when no reranker was configured. Present rather than omitted:
an eval row that lacks the key is indistinguishable from one written before the
key existed."""


class RetrievalError(RuntimeError):
    """The retriever was asked for something it cannot do."""


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    """One retrieved chunk, with provenance and the evidence for its rank.

    The fields are the ones a citation needs (``source_path``, ``chunk_id``,
    ``title``, heading ancestry) plus the text itself. Deliberately flat and
    JSON-safe: this is what the CLI prints, what the eval harness scores, and
    what a generation stage would later put in a prompt.
    """

    chunk_id: str
    source_path: str
    text: str
    score: float
    rank: int
    title: str | None = None
    heading: str | None = None
    heading_path: str | None = None
    point_id: str = ""
    branches: tuple[str, ...] = ()
    branch_ranks: dict[str, int] = field(default_factory=dict)
    branch_scores: dict[str, float] = field(default_factory=dict)
    # Set only when a reranker ran (M3). Keeping the fusion position instead of
    # overwriting ``rank`` is what makes "the cross-encoder pulled this from 27 to
    # 2" a statement the response itself can support.
    rerank_score: float | None = None
    pre_rerank_rank: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form, as printed by the retrieve CLI.

        The rerank keys appear only when a reranker actually ran, so a M2-era
        consumer of this payload sees exactly the M2 shape.
        """
        payload: dict[str, Any] = {
            "rank": self.rank,
            "score": round(self.score, 6),
            "chunk_id": self.chunk_id,
            "source_path": self.source_path,
            "title": self.title,
            "heading": self.heading,
            "heading_path": self.heading_path,
            "point_id": self.point_id,
            "branches": list(self.branches),
            "branch_ranks": dict(self.branch_ranks),
            "branch_scores": {
                branch: round(score, 6) for branch, score in self.branch_scores.items()
            },
            "text": self.text,
        }
        if self.rerank_score is not None:
            payload["rerank_score"] = round(self.rerank_score, 6)
        if self.pre_rerank_rank is not None:
            payload["pre_rerank_rank"] = self.pre_rerank_rank
        return payload


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """The hits plus what produced them.

    The metadata is not decoration: a retrieval number is only reproducible if the
    mode, the candidate depths and the fusion parameters that produced it are
    recorded next to it.
    """

    query: str
    mode: str
    hits: tuple[RetrievalHit, ...]
    collection: str
    embedded_model: str
    dense_candidates: int = 0
    sparse_candidates: int = 0
    sparse_query_terms: int = 0
    fusion_k: int = 0
    weights: dict[str, float] = field(default_factory=dict)
    score_threshold: float = 0.0
    dropped_below_threshold: int = 0
    # The metadata filter that was in force, if any. Reported for the same reason
    # as the mode and the candidate depths: a hit count measured under a filter is
    # not comparable with one measured without it, and nothing else in the result
    # says a filter was applied.
    query_filter: QueryFilter | None = None
    # What the rerank stage did. Reported even when nothing reranked, because "was
    # this ranking reranked?" must be answerable from the response a caller already
    # has — a rerank that quietly stopped happening looks exactly like a rerank
    # that is not helping.
    rerank: RerankOutcome | None = None

    def to_summary(self) -> dict[str, Any]:
        """Machine-readable summary, the CLI's last line of stdout."""
        return {
            "ok": True,
            "query": self.query,
            "mode": self.mode,
            "collection": self.collection,
            "embedded_model": self.embedded_model,
            "returned": len(self.hits),
            "dense_candidates": self.dense_candidates,
            "sparse_candidates": self.sparse_candidates,
            "sparse_query_terms": self.sparse_query_terms,
            "fusion_k": self.fusion_k,
            "weights": dict(self.weights),
            "score_threshold": self.score_threshold,
            "dropped_below_threshold": self.dropped_below_threshold,
            "filters": (
                self.query_filter.to_summary() if self.query_filter else NO_FILTER_SUMMARY
            ),
            "rerank": self.rerank.as_dict() if self.rerank else NO_RERANK_SUMMARY,
            "hits": [hit.to_dict() for hit in self.hits],
        }


class Retriever:
    """Queries a dual-vector collection in dense, sparse or hybrid mode."""

    def __init__(
        self,
        *,
        store: SearchableVectorStore,
        embedder: EmbeddingProvider,
        config: RetrievalConfig | None = None,
        sparse_encoder: SparseEncoder | None = None,
        reranker: Reranker | None = None,
        rerank_config: RerankConfig | None = None,
        filter_policy: FilterPolicy | None = None,
    ) -> None:
        """Wire the retriever to a store and an embedder.

        Args:
            store: A searchable store. The same collection the ingest wrote.
            embedder: Must be the model that built the index — a different one
                produces vectors in a different space, and the resulting ranking
                is meaningless rather than merely worse.
            config: Retrieval knobs; the documented defaults when omitted.
            sparse_encoder: Query-side encoder. Defaults to BM25 with the default
                tokenizer. Needs no fitted statistics, by design.
            reranker: Second-stage scorer applied after fusion. ``None`` — the
                default — is exactly the M2 pipeline, so adding this parameter
                changes no existing behaviour.
            rerank_config: Depth and failure policy for that stage.
            filter_policy: Which metadata fields a caller may filter on, and which
                of them are indexed. Defaults to the allowlist in *config* with no
                index information — honest for a caller that did not supply the
                Qdrant block, since claiming an index that may not exist would
                suppress the payload-scan warning.
        """
        self._store = store
        self._embedder = embedder
        self._config = config or RetrievalConfig()
        self._sparse = sparse_encoder or build_sparse_encoder()
        self._reranker = reranker
        self._rerank_config = rerank_config or RerankConfig()
        self._filter_policy = filter_policy or FilterPolicy.from_fields(
            self._config.filters.allowed_fields
        )

    @classmethod
    def from_config(
        cls,
        *,
        store: SearchableVectorStore,
        embedder: EmbeddingProvider,
        config: YamlConfig,
        reranker: Reranker | None = None,
    ) -> Retriever:
        """Build a retriever from a whole YAML profile.

        The query-side tokenizer is taken from ``ingest.sparse``, not from a
        retrieval block: a query tokenised differently from the corpus is the
        classic lexical-search bug, so there is one setting and both sides read it.

        The reranker is passed in rather than built here, because constructing one
        may need a credential from the environment, and this class should not be
        the place that reads secrets.
        """
        sparse_config = config.ingest.sparse
        return cls(
            store=store,
            embedder=embedder,
            config=config.retrieval,
            reranker=reranker,
            rerank_config=config.rerank,
            # Both halves of the filter contract come from the file: what a
            # deployment permits (`retrieval.filters`) and what it made cheap
            # (`qdrant.payload_indexes`). Reading only the first is how an
            # allowlisted field silently becomes a payload scan.
            filter_policy=FilterPolicy.from_fields(
                config.retrieval.filters.allowed_fields,
                [index.field for index in config.qdrant.payload_indexes],
            ),
            sparse_encoder=build_sparse_encoder(
                method=sparse_config.method,
                k1=sparse_config.k1,
                b=sparse_config.b,
                lowercase=sparse_config.lowercase,
                stopwords=sparse_config.stopwords,
            ),
        )

    @property
    def config(self) -> RetrievalConfig:
        """The retrieval configuration in force."""
        return self._config

    @property
    def store(self) -> SearchableVectorStore:
        """The store being queried. An eval report records which collection it was."""
        return self._store

    @property
    def embedder(self) -> EmbeddingProvider:
        """The embedder in use. An eval score is only meaningful next to its model."""
        return self._embedder

    @property
    def filter_policy(self) -> FilterPolicy:
        """Which metadata fields this retriever accepts a filter on."""
        return self._filter_policy

    def retrieve(
        self,
        query: str,
        *,
        mode: str | None = None,
        top_k: int | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> RetrievalResult:
        """Retrieve chunks for *query*.

        Args:
            query: The user's question, verbatim.
            mode: ``dense``, ``sparse`` or ``hybrid``. Defaults to the configured
                mode.
            top_k: Hits to return after fusion. Defaults to the configured value.
                Candidate depth per branch is *not* derived from this: fusion can
                only reorder what it was given, so the branches stay deep.
            filters: Metadata narrowing, e.g. ``{"source": "handbook"}``. Fields
                must be in ``retrieval.filters.allowed_fields``; an unknown one
                raises rather than being dropped. Omitted or empty is exactly the
                unfiltered path, byte for byte.

        Returns:
            A :class:`RetrievalResult`, possibly with no hits — an honest empty
            answer, with the counts that explain why.

        Raises:
            RetrievalError: Empty query, unknown mode, or non-positive *top_k*.
            FilterError: A filter field is not allowlisted, or its value is not
                expressible. Raised before the embedder is called, so a rejected
                filter never costs a provider round-trip.
            EmbeddingError: The embedding provider failed.
            VectorStoreError: The store could not be queried.
            RerankError: Reranking failed and ``rerank.fail_open`` is false.
        """
        text = query.strip()
        if not text:
            raise RetrievalError("query must not be empty")

        resolved_mode = (mode or self._config.mode).strip().lower()
        if resolved_mode not in RETRIEVAL_MODES:
            raise RetrievalError(
                f"unknown retrieval mode {resolved_mode!r}; expected one of "
                f"{', '.join(RETRIEVAL_MODES)}"
            )
        limit = self._config.top_k if top_k is None else top_k
        if limit <= 0:
            raise RetrievalError(f"top_k must be positive, got {limit}")

        # Before the embedder, deliberately: a filter this deployment does not
        # allow must not cost an embedding call to find out.
        query_filter = self._filter_policy.build(filters)

        # With a reranker in the pipeline, fusion is no longer the last word, so it
        # hands over a deeper shortlist: the reranker can only recover a relevant
        # chunk it was actually shown. Never fewer than the caller asked for.
        fuse_limit = max(limit, self._rerank_config.input_top_k) if self._reranker else limit

        fields = list(self._config.return_payload_fields)
        dense_hits: list[SearchHit] = []
        sparse_hits: list[SearchHit] = []
        sparse_terms = 0

        if resolved_mode in (MODE_DENSE, MODE_HYBRID):
            vector = self._embedder.embed_query(text)
            dense_hits = self._store.search_dense(
                vector,
                limit=self._config.dense_top_k,
                with_payload=fields,
                query_filter=query_filter,
            )

        if resolved_mode in (MODE_SPARSE, MODE_HYBRID):
            sparse_query = self._sparse.encode_query(text)
            sparse_terms = len(sparse_query.indices)
            if sparse_query.is_empty:
                # Every term was a stopword. Legitimate, but it silently halves a
                # hybrid query, so it is reported rather than inferred.
                _log.info("sparse_query_empty", query_length=len(text))
            else:
                sparse_hits = self._store.search_sparse(
                    sparse_query,
                    limit=self._config.sparse_top_k,
                    with_payload=fields,
                    query_filter=query_filter,
                )

        fused, dropped = self._fuse(dense_hits, sparse_hits, mode=resolved_mode, limit=fuse_limit)
        outcome = apply_rerank(
            self._reranker,
            text,
            fused,
            top_n=limit,
            fail_open=self._rerank_config.fail_open,
        )
        fusion = self._config.fusion
        result = RetrievalResult(
            query=text,
            mode=resolved_mode,
            hits=outcome.hits,
            collection=self._store.collection,
            embedded_model=self._embedder.model,
            dense_candidates=len(dense_hits),
            sparse_candidates=len(sparse_hits),
            sparse_query_terms=sparse_terms,
            fusion_k=fusion.k,
            weights=self._weights(resolved_mode),
            score_threshold=self._config.score_threshold,
            dropped_below_threshold=dropped,
            query_filter=query_filter,
            rerank=outcome if self._reranker else None,
        )
        _log.info(
            "retrieval_completed",
            mode=resolved_mode,
            returned=len(outcome.hits),
            filter_fields=list(query_filter.fields) if query_filter else [],
            reranker=outcome.reranker,
            rerank_applied=outcome.applied,
            dense_candidates=len(dense_hits),
            sparse_candidates=len(sparse_hits),
            dropped_below_threshold=dropped,
        )
        return result

    def _weights(self, mode: str) -> dict[str, float]:
        """Per-branch RRF weights for *mode*.

        A single-branch mode carries weight 1.0 so its fused scores stay on the
        same scale as a hybrid run's; only ``hybrid`` reads the configured
        weights, since weighting one branch against nothing is meaningless.
        """
        if mode == MODE_DENSE:
            return {DENSE_BRANCH: 1.0}
        if mode == MODE_SPARSE:
            return {SPARSE_BRANCH: 1.0}
        return {
            DENSE_BRANCH: self._config.fusion.dense_weight,
            SPARSE_BRANCH: self._config.fusion.sparse_weight,
        }

    def _fuse(
        self,
        dense_hits: list[SearchHit],
        sparse_hits: list[SearchHit],
        *,
        mode: str,
        limit: int,
    ) -> tuple[tuple[RetrievalHit, ...], int]:
        """Fuse the branches by RRF and project onto :class:`RetrievalHit`.

        Fusion runs even for a single branch: one code path, so a dense-only run
        and the dense half of a hybrid run cannot disagree about ordering, and the
        fused score is comparable across modes.
        """
        payloads: dict[str, dict[str, Any]] = {}
        raw_scores: dict[str, dict[str, float]] = {}
        rankings: dict[str, list[str]] = {}

        for branch, hits in ((DENSE_BRANCH, dense_hits), (SPARSE_BRANCH, sparse_hits)):
            if branch not in self._weights(mode):
                continue
            rankings[branch] = rank_keys([(hit.point_id, hit.score) for hit in hits])
            for hit in hits:
                payloads.setdefault(hit.point_id, hit.payload)
                raw_scores.setdefault(hit.point_id, {})[branch] = hit.score

        fused = reciprocal_rank_fusion(
            rankings,
            k=self._config.fusion.k,
            weights=self._weights(mode),
        )

        threshold = self._config.score_threshold
        kept: list[RetrievalHit] = []
        dropped = 0
        for item in fused:
            if threshold > 0.0 and item.score < threshold:
                dropped += 1
                continue
            if len(kept) >= limit:
                break
            payload = payloads.get(item.key, {})
            kept.append(
                RetrievalHit(
                    chunk_id=_text_field(payload, "chunk_id"),
                    source_path=_text_field(payload, "source_path"),
                    text=_text_field(payload, "text"),
                    score=item.score,
                    rank=len(kept) + 1,
                    title=_optional_field(payload, "title"),
                    heading=_optional_field(payload, "heading"),
                    heading_path=_optional_field(payload, "heading_path"),
                    point_id=item.key,
                    branches=item.branches,
                    branch_ranks=dict(item.ranks),
                    branch_scores=dict(raw_scores.get(item.key, {})),
                )
            )
        return tuple(kept), dropped


def _text_field(payload: dict[str, Any], key: str) -> str:
    """Read a required payload string, tolerating a projection that omitted it.

    An absent field yields ``""`` rather than a ``KeyError``: the payload
    projection is configurable, and a retriever that crashes because someone
    trimmed ``return_payload_fields`` would be a worse failure than an empty
    citation field.
    """
    value = payload.get(key)
    return value if isinstance(value, str) else ""


def _optional_field(payload: dict[str, Any], key: str) -> str | None:
    """Read an optional payload string; ``None`` when absent or null."""
    value = payload.get(key)
    return value if isinstance(value, str) else None
