"""BM25 sparse vectors: tokenisation, the ranking function, and the query seam."""

from __future__ import annotations

import math

import pytest

from production_rag.retrieval.sparse import (
    BM25_METHOD,
    Bm25Encoder,
    Bm25Tokenizer,
    SparseEncoder,
    SparseError,
    SparseVector,
    build_sparse_encoder,
    term_index,
)

CORPUS = [
    "Qdrant stores dense and sparse vectors in one collection.",
    "Reciprocal rank fusion combines two ranked lists.",
    "Qdrant supports named vectors.",
    "The chunker splits markdown documents by heading.",
]


def _encoder(**kwargs: object) -> Bm25Encoder:
    encoder = Bm25Encoder(**kwargs)  # type: ignore[arg-type]
    encoder.fit(CORPUS)
    return encoder


class TestTokenizer:
    def test_lowercases_and_splits_on_punctuation(self) -> None:
        assert Bm25Tokenizer(stopwords="none").tokenize("Qdrant, stores vectors!") == [
            "qdrant",
            "stores",
            "vectors",
        ]

    def test_keeps_hyphenated_and_dotted_literals_whole(self) -> None:
        # The whole reason the sparse branch exists: these literal tokens are what
        # dense retrieval misses, so splitting them would reopen the recall hole.
        assert Bm25Tokenizer(stopwords="none").tokenize(
            "run --recreate-collection with top_k 0.15 on text-embedding-3-small"
        ) == [
            "run",
            "recreate-collection",
            "with",
            "top_k",
            "0.15",
            "on",
            "text-embedding-3-small",
        ]

    def test_drops_english_stopwords(self) -> None:
        assert Bm25Tokenizer().tokenize("the store is in a collection") == [
            "store",
            "collection",
        ]

    def test_lowercase_off_keeps_case_distinct(self) -> None:
        tokenizer = Bm25Tokenizer(lowercase=False, stopwords="none")
        # The regex alphabet is lowercase, so uppercase text yields no terms at
        # all. Surprising enough to pin: casefolding is effectively mandatory.
        assert tokenizer.tokenize("Qdrant") == ["drant"]

    @pytest.mark.parametrize("name", ["none", ""])
    def test_stopwords_can_be_disabled_by_name(self, name: str) -> None:
        assert Bm25Tokenizer(stopwords=name).stopwords == frozenset()

    def test_explicit_stopword_iterable_is_casefolded(self) -> None:
        tokenizer = Bm25Tokenizer(stopwords=["Qdrant"])
        assert tokenizer.tokenize("qdrant stores vectors") == ["stores", "vectors"]

    def test_unknown_named_stopword_set_is_rejected(self) -> None:
        # Silently indexing with no stopwords would look like a quality problem
        # much later, in an eval number nobody can explain.
        with pytest.raises(SparseError, match="unknown stopword set"):
            Bm25Tokenizer(stopwords="spanish")


class TestTermIndex:
    def test_is_stable_across_calls(self) -> None:
        assert term_index("qdrant") == term_index("qdrant")

    def test_is_a_known_constant(self) -> None:
        # Pinning one value proves the hash is hashlib, not the per-process salted
        # builtin: a salted hash would make every re-ingest write a different
        # vector for the same term.
        assert term_index("qdrant") == 1_990_180_345

    def test_stays_inside_the_31_bit_range(self) -> None:
        for term in ("a", "qdrant", "text-embedding-3-small", "ñ", "0.15"):
            assert 0 <= term_index(term) <= 0x7FFF_FFFF

    def test_distinct_terms_get_distinct_indices(self) -> None:
        terms = {term for text in CORPUS for term in Bm25Tokenizer().tokenize(text)}
        assert len({term_index(term) for term in terms}) == len(terms)


class TestSparseVector:
    def test_from_weights_sorts_by_index(self) -> None:
        vector = SparseVector.from_weights({7: 0.5, 2: 1.5})
        assert vector.indices == (2, 7)
        assert vector.values == (1.5, 0.5)

    def test_equal_weights_produce_equal_vectors(self) -> None:
        assert SparseVector.from_weights({2: 1.0, 7: 0.5}) == SparseVector.from_weights(
            {7: 0.5, 2: 1.0}
        )

    def test_length_mismatch_is_rejected(self) -> None:
        with pytest.raises(SparseError, match="indices and"):
            SparseVector(indices=(1, 2), values=(1.0,))

    def test_duplicate_indices_are_rejected(self) -> None:
        with pytest.raises(SparseError, match="duplicate indices"):
            SparseVector(indices=(1, 1), values=(1.0, 2.0))

    def test_dot_sums_only_shared_terms(self) -> None:
        query = SparseVector.from_weights({1: 1.0, 2: 1.0})
        document = SparseVector.from_weights({2: 3.0, 9: 5.0})
        assert query.dot(document) == pytest.approx(3.0)

    def test_dot_with_an_empty_vector_is_zero(self) -> None:
        assert SparseVector((), ()).dot(SparseVector.from_weights({1: 1.0})) == 0.0
        assert SparseVector.from_weights({1: 1.0}).dot(SparseVector((), ())) == 0.0

    def test_is_empty_reports_no_terms(self) -> None:
        assert SparseVector((), ()).is_empty
        assert not SparseVector.from_weights({1: 1.0}).is_empty

    def test_as_dict_is_json_friendly(self) -> None:
        assert SparseVector.from_weights({4: 2.0}).as_dict() == {
            "indices": [4],
            "values": [2.0],
        }


class TestEncoder:
    def test_satisfies_the_encoder_protocol(self) -> None:
        assert isinstance(Bm25Encoder(), SparseEncoder)

    def test_method_is_reported_for_the_ingest_summary(self) -> None:
        assert Bm25Encoder().method == BM25_METHOD == "bm25"

    def test_encoding_documents_before_fitting_is_refused(self) -> None:
        with pytest.raises(SparseError, match="must be fitted"):
            Bm25Encoder().encode_documents(["anything"])

    def test_stats_before_fitting_is_refused(self) -> None:
        with pytest.raises(SparseError, match="call fit"):
            _ = Bm25Encoder().stats

    def test_fitting_an_empty_corpus_is_refused(self) -> None:
        # An empty fit yields undefined IDF, and vectors of zero weight are
        # indistinguishable from "nothing matched".
        with pytest.raises(SparseError, match="empty corpus"):
            Bm25Encoder().fit([])

    def test_stats_describe_the_fitted_corpus(self) -> None:
        encoder = _encoder()
        stats = encoder.stats
        assert stats.document_count == len(CORPUS)
        assert stats.average_length > 0
        assert stats.vocabulary_size == len(
            {term for text in CORPUS for term in encoder.tokenizer.tokenize(text)}
        )
        assert set(stats.as_dict()) == {
            "document_count",
            "average_length",
            "vocabulary_size",
        }

    def test_queries_need_no_statistics_at_all(self) -> None:
        # The property that keeps retrieval free of corpus state: the retrieve CLI
        # queries a collection it did not build.
        vector = Bm25Encoder().encode_query("qdrant vectors")
        assert vector.values == (1.0, 1.0)

    def test_query_weights_are_one_per_distinct_term(self) -> None:
        vector = Bm25Encoder().encode_query("qdrant qdrant qdrant vectors")
        assert sorted(vector.values) == [1.0, 1.0]

    def test_an_all_stopword_query_encodes_to_nothing(self) -> None:
        assert Bm25Encoder().encode_query("the and of").is_empty

    def test_query_dot_document_equals_the_bm25_score(self) -> None:
        encoder = _encoder(k1=1.2, b=0.75)
        (document,) = encoder.encode_documents([CORPUS[0]])
        query = encoder.encode_query("qdrant")

        terms = encoder.tokenizer.tokenize(CORPUS[0])
        length = len(terms)
        average = encoder.stats.average_length
        frequency = terms.count("qdrant")
        document_frequency = sum(
            1 for text in CORPUS if "qdrant" in encoder.tokenizer.tokenize(text)
        )
        idf = math.log(1 + (len(CORPUS) - document_frequency + 0.5) / (document_frequency + 0.5))
        expected = idf * frequency * 2.2 / (frequency + 1.2 * (1 - 0.75 + 0.75 * length / average))
        assert query.dot(document) == pytest.approx(expected)

    def test_a_rarer_term_outweighs_a_common_one(self) -> None:
        # IDF doing its job. "qdrant" is in 2 of 4 documents, "chunker" in 1.
        encoder = _encoder()
        (document,) = encoder.encode_documents(["qdrant chunker"])
        weights = dict(zip(document.indices, document.values, strict=True))
        assert weights[term_index("chunker")] > weights[term_index("qdrant")]

    def test_repeating_a_term_saturates_rather_than_scaling(self) -> None:
        encoder = _encoder()
        once, thrice = encoder.encode_documents(["chunker", "chunker chunker chunker"])
        single = dict(zip(once.indices, once.values, strict=True))[term_index("chunker")]
        triple = dict(zip(thrice.indices, thrice.values, strict=True))[term_index("chunker")]
        assert single < triple < 3 * single

    def test_length_normalisation_penalises_a_longer_document(self) -> None:
        encoder = _encoder()
        short, long = encoder.encode_documents(
            ["chunker", "chunker " + " ".join(f"filler{n}" for n in range(60))]
        )
        short_weight = dict(zip(short.indices, short.values, strict=True))[term_index("chunker")]
        long_weight = dict(zip(long.indices, long.values, strict=True))[term_index("chunker")]
        assert long_weight < short_weight

    def test_b_zero_disables_length_normalisation(self) -> None:
        encoder = _encoder(b=0.0)
        short, long = encoder.encode_documents(
            ["chunker", "chunker " + " ".join(f"filler{n}" for n in range(60))]
        )
        short_weight = dict(zip(short.indices, short.values, strict=True))[term_index("chunker")]
        long_weight = dict(zip(long.indices, long.values, strict=True))[term_index("chunker")]
        assert short_weight == pytest.approx(long_weight)

    def test_a_term_in_every_document_gets_a_small_weight(self) -> None:
        encoder = Bm25Encoder()
        encoder.fit(["chunker"] * 4)
        (document,) = encoder.encode_documents(["chunker"])
        # Non-negative, not zero-crossing: the +1 inside the log is what keeps a
        # ubiquitous term from scoring negatively and inverting a ranking.
        assert 0.0 < document.values[0] < 0.5

    def test_an_all_stopword_document_encodes_to_nothing(self) -> None:
        assert _encoder().encode_documents(["the and of"])[0].is_empty

    def test_encoding_is_deterministic_across_encoders(self) -> None:
        assert _encoder().encode_documents(CORPUS) == _encoder().encode_documents(CORPUS)

    def test_document_count_is_preserved(self) -> None:
        assert len(_encoder().encode_documents(CORPUS)) == len(CORPUS)

    @pytest.mark.parametrize(("k1", "b"), [(-0.1, 0.75), (1.2, -0.1), (1.2, 1.1)])
    def test_out_of_range_parameters_are_rejected(self, k1: float, b: float) -> None:
        with pytest.raises(SparseError):
            Bm25Encoder(k1=k1, b=b)


class TestBuildSparseEncoder:
    def test_builds_a_bm25_encoder_from_config_values(self) -> None:
        encoder = build_sparse_encoder(k1=1.5, b=0.5, lowercase=True, stopwords="none")
        assert encoder.method == BM25_METHOD
        assert encoder.tokenizer.stopwords == frozenset()

    def test_an_unimplemented_method_fails_loudly(self) -> None:
        # Declaring `method: splade` must not silently produce BM25 vectors an
        # eval would then attribute to the wrong retriever.
        with pytest.raises(SparseError, match="unsupported sparse method"):
            build_sparse_encoder(method="splade")
