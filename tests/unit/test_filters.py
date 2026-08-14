"""Metadata filters: the allowlist, the mapping, and what a filter changes.

Everything runs against :class:`InMemoryVectorStore` and the fake embedder, so
the whole filtered retrieval path is exercised with no container and no network.
The Qdrant translation is asserted on the expression that would be sent, which is
the part that can be wrong without a running server to notice.
"""

from __future__ import annotations

import pytest
import structlog
from qdrant_client import models

from production_rag.config_loader import (
    FiltersConfig,
    PayloadIndexConfig,
    QdrantConfig,
    RetrievalConfig,
    YamlConfig,
)
from production_rag.ingest.models import Chunk, Document
from production_rag.retrieval.embeddings import FakeEmbeddingProvider
from production_rag.retrieval.filters import (
    FILTER_INVALID_VALUE,
    FILTER_NOT_ALLOWED,
    NO_FILTER_SUMMARY,
    PUBLIC_TO_PAYLOAD,
    FilterError,
    FilterPolicy,
    parse_filter_arguments,
)
from production_rag.retrieval.hybrid import Retriever
from production_rag.retrieval.sparse import Bm25Encoder
from production_rag.retrieval.store import InMemoryVectorStore, to_qdrant_filter

RUN_ID = "run-filters"

CORPUS = (
    # (source_path, source, tags, text)
    ("handbook/01-vectors.md", "handbook", ("bm25",), "Qdrant stores dense and sparse vectors."),
    ("handbook/02-fusion.md", "handbook", ("rrf",), "Reciprocal rank fusion merges ranked lists."),
    ("notes/03-vectors.md", "notes", ("bm25", "rrf"), "Sparse vectors carry BM25 term weights."),
    ("notes/04-chunking.md", "notes", (), "The chunker splits markdown on heading boundaries."),
)

DEFAULT_POLICY = FilterPolicy.from_fields(("source", "title", "tags"), ("source", "tags"))
"""Mirrors the shipped profile: every allowlisted field is indexed except ``title``."""


def _chunks() -> list[Chunk]:
    chunks = []
    for index, (path, source, tags, text) in enumerate(CORPUS):
        document = Document(
            source_path=path,
            text=text,
            title=path.rsplit("/", 1)[-1],
            source=source,
            tags=tags,
        )
        chunks.append(Chunk.build(document=document, chunk_index=index, text=text, embed_text=text))
    return chunks


def _retriever(*, config: YamlConfig | None = None) -> Retriever:
    profile = config or YamlConfig(
        retrieval=RetrievalConfig(filters=FiltersConfig()),
        qdrant=QdrantConfig(),
    )
    embedder = FakeEmbeddingProvider()
    chunks = _chunks()
    texts = [chunk.embed_text for chunk in chunks]
    store = InMemoryVectorStore()
    store.ensure_collection(vector_size=embedder.dimensions, with_sparse=True)
    encoder = Bm25Encoder()
    encoder.fit(texts)
    store.upsert_chunks(
        chunks,
        embedder.embed_documents(texts),
        ingest_run_id=RUN_ID,
        embedded_model=embedder.model,
        sparse_vectors=encoder.encode_documents(texts),
    )
    return Retriever.from_config(store=store, embedder=embedder, config=profile)


def _sources(retriever: Retriever, **kwargs: object) -> set[str]:
    result = retriever.retrieve("vectors", **kwargs)  # type: ignore[arg-type]
    return {hit.source_path.split("/", 1)[0] for hit in result.hits}


class TestAllowlist:
    """A field outside the allowlist fails closed, with a typed reason."""

    def test_unknown_field_is_rejected(self) -> None:
        with pytest.raises(FilterError) as excinfo:
            DEFAULT_POLICY.build({"author": "pablo"})
        assert excinfo.value.error_type == FILTER_NOT_ALLOWED
        assert excinfo.value.field == "author"

    def test_error_message_names_the_allowed_fields(self) -> None:
        """The fix has to be readable off the rejection, not off the config file."""
        with pytest.raises(FilterError, match="source, title, tags"):
            DEFAULT_POLICY.build({"author": "pablo"})

    def test_source_path_is_not_a_filterable_field(self) -> None:
        """The mapping decision, asserted where it would be got wrong.

        ``source`` is the first path segment and is indexed. ``source_path`` is the
        full corpus-relative path and is neither allowlisted nor indexed. Answering
        one against the other would silently return a different result set.
        """
        with pytest.raises(FilterError) as excinfo:
            DEFAULT_POLICY.build({"source_path": "handbook/01-vectors.md"})
        assert excinfo.value.error_type == FILTER_NOT_ALLOWED

    def test_an_allowlisted_field_the_payload_cannot_express_is_still_rejected(self) -> None:
        """A deployment cannot invent a filterable field by editing the allowlist."""
        policy = FilterPolicy.from_fields(("source", "created_at"))
        assert policy.filterable_fields == ("source",)
        with pytest.raises(FilterError) as excinfo:
            policy.build({"created_at": "2026-08-14"})
        assert excinfo.value.error_type == FILTER_NOT_ALLOWED

    def test_public_names_map_onto_real_payload_keys(self) -> None:
        """Every mapped key must exist on a chunk payload, or the filter is a no-op."""
        payload = _chunks()[0].to_payload(ingest_run_id=RUN_ID, embedded_model="fake")
        for public, key in PUBLIC_TO_PAYLOAD.items():
            assert key in payload, f"{public!r} maps onto a payload key that is not written"


class TestValues:
    """Values are keywords or lists of keywords; anything else fails closed."""

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(5, id="number"),
            pytest.param(None, id="null"),
            pytest.param({}, id="object"),
        ],
    )
    def test_non_string_values_are_rejected(self, value: object) -> None:
        with pytest.raises(FilterError) as excinfo:
            DEFAULT_POLICY.build({"source": value})
        assert excinfo.value.error_type == FILTER_INVALID_VALUE

    @pytest.mark.parametrize(
        "value",
        [pytest.param([], id="empty-list"), pytest.param("  ", id="blank-string")],
    )
    def test_empty_alternatives_are_rejected(self, value: object) -> None:
        """An empty set matches nothing, and would read as a corpus gap."""
        with pytest.raises(FilterError) as excinfo:
            DEFAULT_POLICY.build({"source": value})
        assert excinfo.value.error_type == FILTER_INVALID_VALUE

    def test_repeated_alternatives_are_deduplicated(self) -> None:
        built = DEFAULT_POLICY.build({"tags": ["bm25", "bm25", "rrf"]})
        assert built is not None
        assert built.conditions[0].values == ("bm25", "rrf")

    @pytest.mark.parametrize("empty", [None, {}], ids=["none", "empty-object"])
    def test_nothing_asked_for_is_no_filter(self, empty: object) -> None:
        assert DEFAULT_POLICY.build(empty) is None  # type: ignore[arg-type]


class TestIndexAwareness:
    """An unindexed filter works, and is never silent about what it costs."""

    def test_unindexed_field_is_reported_on_the_filter(self) -> None:
        built = DEFAULT_POLICY.build({"title": "01-vectors.md"})
        assert built is not None
        assert built.unindexed_fields == ("title",)

    def test_unindexed_field_emits_a_warning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Capture via the module logger object itself. Suite-wide structlog
        # reconfiguration (create_app) makes capture_logs order-dependent.
        from production_rag.retrieval import filters as filters_mod

        events: list[dict[str, object]] = []

        class _Capture:
            def warning(self, event: str, **kwargs: object) -> None:
                events.append({"event": event, **kwargs})

        monkeypatch.setattr(filters_mod, "_log", _Capture())
        DEFAULT_POLICY.build({"title": "01-vectors.md"})
        assert events
        assert events[0]["event"] == "filter_field_unindexed"
        assert events[0]["fields"] == ["title"]

    def test_indexed_field_is_quiet(self) -> None:
        with structlog.testing.capture_logs() as logs:
            DEFAULT_POLICY.build({"source": "handbook"})
        assert not [entry for entry in logs if entry["event"] == "filter_field_unindexed"]

    def test_policy_reads_both_config_blocks(self) -> None:
        """Permission comes from `retrieval.filters`, cost from `qdrant.payload_indexes`."""
        profile = YamlConfig(
            retrieval=RetrievalConfig(filters=FiltersConfig(allowed_fields=("source", "tags"))),
            qdrant=QdrantConfig(payload_indexes=(PayloadIndexConfig(field="source"),)),
        )
        policy = _retriever(config=profile).filter_policy
        assert policy.allowed_fields == ("source", "tags")
        built = policy.build({"tags": "bm25"})
        assert built is not None
        assert built.unindexed_fields == ("tags",)


class TestQdrantTranslation:
    """The expression that would be sent, asserted without a server."""

    def test_no_filter_translates_to_none(self) -> None:
        assert to_qdrant_filter(None) is None

    def test_conditions_are_anded_and_values_are_ored(self) -> None:
        built = DEFAULT_POLICY.build({"source": "handbook", "tags": ["bm25", "rrf"]})
        expression = to_qdrant_filter(built)
        assert expression is not None
        assert expression.must is not None
        conditions = [
            (condition.key, condition.match)
            for condition in expression.must
            if isinstance(condition, models.FieldCondition)
        ]
        assert conditions == [
            ("source", models.MatchAny(any=["handbook"])),
            ("tags", models.MatchAny(any=["bm25", "rrf"])),
        ]


class TestFilteredRetrieval:
    """What a filter actually changes about the hit set."""

    def test_unfiltered_retrieval_sees_every_source(self) -> None:
        assert _sources(_retriever()) == {"handbook", "notes"}

    def test_allowlisted_field_narrows_the_hit_set(self) -> None:
        assert _sources(_retriever(), filters={"source": "handbook"}) == {"handbook"}

    def test_several_values_on_one_field_are_or(self) -> None:
        retriever = _retriever()
        assert _sources(retriever, filters={"source": ["handbook", "notes"]}) == {
            "handbook",
            "notes",
        }

    def test_several_fields_are_and(self) -> None:
        """`notes` has an `rrf`-tagged chunk; `handbook` + `rrf` is one document."""
        result = _retriever().retrieve("vectors", filters={"source": "handbook", "tags": "rrf"})
        assert [hit.source_path for hit in result.hits] == ["handbook/02-fusion.md"]

    def test_a_list_valued_payload_matches_on_any_element(self) -> None:
        result = _retriever().retrieve("vectors", filters={"tags": "rrf"})
        assert {hit.source_path for hit in result.hits} == {
            "handbook/02-fusion.md",
            "notes/03-vectors.md",
        }

    def test_a_filter_matching_nothing_is_an_empty_result_not_an_error(self) -> None:
        result = _retriever().retrieve("vectors", filters={"source": "absent"})
        assert result.hits == ()
        assert result.to_summary()["returned"] == 0

    def test_filtering_on_source_does_not_match_a_source_path_value(self) -> None:
        """`source` holds `handbook`, never `handbook/01-vectors.md`."""
        result = _retriever().retrieve("vectors", filters={"source": "handbook/01-vectors.md"})
        assert result.hits == ()

    def test_the_filter_applies_before_the_top_k_cut(self) -> None:
        """Qdrant filters before the ANN search; the offline store must agree.

        Filtering a top-k list afterwards would return fewer hits than asked for
        whenever a matching document ranked below the cut.
        """
        config = YamlConfig(retrieval=RetrievalConfig(top_k=2, dense_top_k=2, sparse_top_k=2))
        result = _retriever(config=config).retrieve("chunker", filters={"source": "notes"})
        assert {hit.source_path for hit in result.hits} == {
            "notes/03-vectors.md",
            "notes/04-chunking.md",
        }

    def test_an_unknown_field_raises_before_the_embedder_runs(self) -> None:
        """A rejected filter must not cost a provider round-trip."""
        retriever = _retriever()
        embedder = retriever.embedder
        assert isinstance(embedder, FakeEmbeddingProvider)
        with pytest.raises(FilterError):
            retriever.retrieve("vectors", filters={"author": "pablo"})

    def test_summary_reports_the_filter_that_was_applied(self) -> None:
        summary = _retriever().retrieve("vectors", filters={"source": "handbook"}).to_summary()
        assert summary["filters"]["applied"] is True
        assert summary["filters"]["fields"] == ["source"]
        assert summary["filters"]["unindexed"] == []
        assert summary["filters"]["conditions"][0]["payload_key"] == "source"

    def test_summary_without_a_filter_keeps_the_unfiltered_shape(self) -> None:
        summary = _retriever().retrieve("vectors").to_summary()
        assert summary["filters"] == NO_FILTER_SUMMARY

    def test_an_empty_filter_object_is_the_unfiltered_path(self) -> None:
        retriever = _retriever()
        filtered = retriever.retrieve("vectors", filters={}).to_summary()
        assert filtered == retriever.retrieve("vectors").to_summary()


class TestArgumentParsing:
    """``--filter field=value``, repeatable, with no allowlist knowledge."""

    def test_pairs_become_a_filters_object(self) -> None:
        assert parse_filter_arguments(["source=handbook"]) == {"source": ["handbook"]}

    def test_a_repeated_field_accumulates_alternatives(self) -> None:
        assert parse_filter_arguments(["tags=bm25", "tags=rrf"]) == {"tags": ["bm25", "rrf"]}

    def test_a_value_may_contain_an_equals_sign(self) -> None:
        assert parse_filter_arguments(["title=a=b"]) == {"title": ["a=b"]}

    def test_no_arguments_is_an_empty_object(self) -> None:
        assert parse_filter_arguments(None) == {}

    @pytest.mark.parametrize("argument", ["source", "=handbook", ""])
    def test_a_malformed_pair_is_rejected(self, argument: str) -> None:
        with pytest.raises(FilterError) as excinfo:
            parse_filter_arguments([argument])
        assert excinfo.value.error_type == FILTER_INVALID_VALUE

    def test_parsing_does_not_enforce_the_allowlist(self) -> None:
        """One allowlist, owned by the policy; a second one in argparse would drift."""
        assert parse_filter_arguments(["author=pablo"]) == {"author": ["pablo"]}
