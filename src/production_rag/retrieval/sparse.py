"""BM25 sparse vectors: the lexical half of hybrid retrieval.

Dense retrieval fails predictably on rare literal tokens — SKUs, error codes,
function names, flag spellings like ``--recreate-collection``. BM25 does not,
because it matches the token itself. This module produces the sparse vectors that
carry that behaviour, in pure Python with no model download and no service.

**Where the weighting happens.** Full BM25 weights are computed here, at ingest,
and stored as-is:

    weight(term, chunk) = idf(term) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
    idf(term)           = ln(1 + (N - df + 0.5) / (df + 0.5))

A query vector then carries **weight 1.0 per distinct query term**, so the dot
product of query and document vector is exactly the BM25 score of that chunk for
that query. Two consequences worth stating:

* Querying needs the tokenizer only, never the fitted statistics. Retrieval has
  no corpus-state dependency and no warm-up.
* Qdrant's own ``modifier: idf`` must **not** be enabled on this vector, or IDF
  would be applied twice. See :meth:`~production_rag.retrieval.store.QdrantStore.ensure_collection`
  and the note in :class:`~production_rag.config_loader.SparseVectorConfig`.

**What this costs.** IDF and average document length are corpus statistics, so
they are computed over the corpus walked in one ingest run. A corpus that grows
substantially shifts IDF and the sparse index goes mildly stale until the next
full ingest. That trade-off is stated in
[ADR 0001](../../../docs/adr/0001-hybrid-qdrant.md) and is the reason
:class:`Bm25Stats` is reported in the ingest summary rather than hidden.

Term identity is a hash, not a vocabulary: index = the first 31 bits of the
term's SHA-1. No vocabulary file to ship, keep in sync or version, and a query
can weight a term the ingest run never saw. The cost is that two distinct terms
could collide; at 2^31 slots and a corpus of this size the probability is
negligible, and a collision degrades ranking slightly rather than corrupting it.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from hashlib import sha1
from typing import Protocol, runtime_checkable

import structlog

_log = structlog.get_logger(__name__)

BM25_METHOD = "bm25"
"""The only sparse method implemented. ``ingest.sparse.method`` must match it."""

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")
"""Word characters, keeping internal ``-``, ``_`` and ``.`` joined.

Deliberate: ``text-embedding-3-small``, ``top_k`` and ``0.1`` are exactly the
literal tokens the sparse branch exists to match. Splitting them into fragments
would hand back the recall hole this module is here to close.
"""

_INDEX_MASK = 0x7FFF_FFFF
"""31 bits: Qdrant sparse indices are unsigned 32-bit, and staying under the sign
bit keeps the value safe through JSON and every client in between."""

ENGLISH_STOPWORDS = frozenset(
    # noqa on the split: the list-literal form is one 1,500-character line that no
    # reviewer will read, and reviewing this list is the point — it is a
    # retrieval-quality decision, not an implementation detail.
    (  # noqa: SIM905
        "a about above after again against all am an and any are aren as at be because been "
        "before being below between both but by can cannot could couldn did didn do does "
        "doesn doing don down during each few for from further had hadn has hasn have haven "
        "having he her here hers herself him himself his how i if in into is isn it its "
        "itself just me more most mustn my myself no nor not now of off on once only or "
        "other ought our ours ourselves out over own same shan she should shouldn so some "
        "such than that the their theirs them themselves then there these they this those "
        "through to too under until up very was wasn we were weren what when where which "
        "while who whom why will with won would wouldn you your yours yourself yourselves"
    ).split(" ")
)
"""Conservative English stopword list.

Kept small and inlined rather than pulled from NLTK: a stopword list is a
retrieval-quality decision that belongs in the repository where a reviewer can
read it and an eval can measure it, not behind a download.
"""

STOPWORD_SETS = {"english": ENGLISH_STOPWORDS, "": frozenset(), "none": frozenset()}
"""``ingest.sparse.stopwords`` values this module understands."""


class SparseError(RuntimeError):
    """The sparse encoder cannot be used as configured."""


@dataclass(frozen=True, slots=True)
class SparseVector:
    """A sparse vector as Qdrant models it: parallel indices and values.

    Immutable, and sorted by index at construction, so equality is meaningful and
    two encoders that agree on weights produce byte-identical vectors.
    """

    indices: tuple[int, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        """Reject a malformed vector at the boundary rather than at upsert."""
        if len(self.indices) != len(self.values):
            raise SparseError(
                f"sparse vector has {len(self.indices)} indices and {len(self.values)} values"
            )
        if len(set(self.indices)) != len(self.indices):
            raise SparseError("sparse vector has duplicate indices")

    @classmethod
    def from_weights(cls, weights: dict[int, float]) -> SparseVector:
        """Build a vector from an index-to-weight mapping, sorted by index."""
        ordered = sorted(weights.items())
        return cls(
            indices=tuple(index for index, _ in ordered),
            values=tuple(float(value) for _, value in ordered),
        )

    @property
    def is_empty(self) -> bool:
        """Whether the vector carries no terms.

        An empty query vector matches nothing, which is the correct behaviour for
        a query made entirely of stopwords — but the caller has to know, so the
        retriever reports it instead of returning a silent zero-hit result.
        """
        return not self.indices

    def dot(self, other: SparseVector) -> float:
        """Dot product with *other*: the BM25 score when *self* is the query."""
        if self.is_empty or other.is_empty:
            return 0.0
        theirs = dict(zip(other.indices, other.values, strict=True))
        return sum(
            value * theirs[index]
            for index, value in zip(self.indices, self.values, strict=True)
            if index in theirs
        )

    def as_dict(self) -> dict[str, list[float] | list[int]]:
        """JSON-friendly form, for a CLI summary or a debug log."""
        return {"indices": list(self.indices), "values": list(self.values)}


@runtime_checkable
class SparseEncoder(Protocol):
    """Turns text into sparse vectors.

    Two sides with different requirements, which is why they are separate
    methods: documents need corpus statistics, queries must not.
    """

    @property
    def method(self) -> str:
        """Identifier recorded in the ingest summary, e.g. ``"bm25"``."""

    def encode_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        """Encode chunk texts. Requires fitted statistics."""

    def encode_query(self, text: str) -> SparseVector:
        """Encode a query. Must work with no fitted statistics."""


@dataclass(frozen=True, slots=True)
class Bm25Stats:
    """Corpus statistics BM25 needs, and the honest record of their scope.

    Reported in the ingest summary: the numbers here are what a later retrieval
    score means, and "which corpus were the IDFs computed over?" is the first
    question when sparse results drift.
    """

    document_count: int
    average_length: float
    vocabulary_size: int

    def as_dict(self) -> dict[str, float | int]:
        """Summary form for the ingest report."""
        return {
            "document_count": self.document_count,
            "average_length": round(self.average_length, 3),
            "vocabulary_size": self.vocabulary_size,
        }


class Bm25Tokenizer:
    """Text to terms: casefold, split, drop stopwords.

    Shared by both sides of the index. A query tokenised differently from the
    corpus is the classic lexical-search bug, so there is exactly one tokenizer
    and both paths take it.
    """

    def __init__(
        self, *, lowercase: bool = True, stopwords: Iterable[str] | str = "english"
    ) -> None:
        """Configure tokenisation.

        Args:
            lowercase: Casefold before matching. With this off, ``Qdrant`` and
                ``qdrant`` become different terms.
            stopwords: A named set (``"english"``, ``"none"``, ``""``) or an
                explicit iterable of words.

        Raises:
            SparseError: A named stopword set that is not known — failing loudly
                beats silently indexing with no stopwords at all.
        """
        self._lowercase = lowercase
        if isinstance(stopwords, str):
            key = stopwords.strip().lower()
            if key not in STOPWORD_SETS:
                known = ", ".join(sorted(name for name in STOPWORD_SETS if name))
                raise SparseError(f"unknown stopword set {stopwords!r}; expected one of {known}")
            self._stopwords = STOPWORD_SETS[key]
        else:
            self._stopwords = frozenset(word.lower() for word in stopwords)

    @property
    def stopwords(self) -> frozenset[str]:
        """The active stopword set."""
        return self._stopwords

    def tokenize(self, text: str) -> list[str]:
        """Return the terms of *text*, in order, stopwords removed."""
        source = text.casefold() if self._lowercase else text
        return [token for token in _TOKEN_RE.findall(source) if token not in self._stopwords]


def term_index(term: str) -> int:
    """Map a term to its sparse-vector index.

    A hash rather than a vocabulary lookup: nothing to persist, nothing to keep
    in sync between ingest and query, and a term the ingest run never saw still
    lands somewhere deterministic. Stable across processes and releases because it
    is :mod:`hashlib`, never ``hash()``.
    """
    digest = sha1(term.encode("utf-8"), usedforsecurity=False).digest()
    return int.from_bytes(digest[:4], "big") & _INDEX_MASK


class Bm25Encoder:
    """BM25 encoder: fit over a corpus, then encode documents and queries.

    Not fitted at construction. :meth:`fit` is a separate, explicit step because
    it is a full pass over the corpus, and a class that hides an O(corpus) call
    inside a constructor is a class that gets called in a loop.
    """

    def __init__(
        self,
        *,
        k1: float = 1.2,
        b: float = 0.75,
        tokenizer: Bm25Tokenizer | None = None,
    ) -> None:
        """Configure the ranking function.

        Args:
            k1: Term-frequency saturation. Higher means repetition keeps helping.
            b: Length normalisation, 0 (off) to 1 (full).
            tokenizer: Shared tokenizer; a default English one when omitted.

        Raises:
            SparseError: A parameter is outside its valid range.
        """
        if k1 < 0:
            raise SparseError(f"bm25 k1 must not be negative, got {k1}")
        if not 0.0 <= b <= 1.0:
            raise SparseError(f"bm25 b must be within [0, 1], got {b}")
        self._k1 = k1
        self._b = b
        self._tokenizer = tokenizer or Bm25Tokenizer()
        self._document_frequency: Counter[str] = Counter()
        self._document_count = 0
        self._average_length = 0.0
        self._fitted = False

    @property
    def method(self) -> str:
        """The sparse method identifier."""
        return BM25_METHOD

    @property
    def tokenizer(self) -> Bm25Tokenizer:
        """The tokenizer both sides of the index share."""
        return self._tokenizer

    @property
    def fitted(self) -> bool:
        """Whether corpus statistics have been computed."""
        return self._fitted

    @property
    def stats(self) -> Bm25Stats:
        """Corpus statistics from the last :meth:`fit`.

        Raises:
            SparseError: Nothing has been fitted yet.
        """
        if not self._fitted:
            raise SparseError("bm25 encoder has no statistics; call fit() first")
        return Bm25Stats(
            document_count=self._document_count,
            average_length=self._average_length,
            vocabulary_size=len(self._document_frequency),
        )

    def fit(self, texts: Sequence[str]) -> Bm25Stats:
        """Compute document frequencies and average length over *texts*.

        Returns:
            The resulting statistics, so a caller can report them.

        Raises:
            SparseError: The corpus is empty. An empty fit would make every IDF
                undefined, and encoding against it would emit zero-weight vectors
                that look exactly like "nothing matched".
        """
        if not texts:
            raise SparseError("cannot fit bm25 on an empty corpus")

        self._document_frequency = Counter()
        total_length = 0
        for text in texts:
            terms = self._tokenizer.tokenize(text)
            total_length += len(terms)
            self._document_frequency.update(set(terms))

        self._document_count = len(texts)
        self._average_length = total_length / len(texts)
        self._fitted = True
        stats = self.stats
        _log.info("bm25_fitted", **stats.as_dict())
        return stats

    def encode_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        """Encode chunk texts as full BM25 weight vectors.

        Raises:
            SparseError: The encoder has not been fitted.
        """
        if not self._fitted:
            raise SparseError("bm25 encoder must be fitted before encoding documents")
        return [self._encode_document(text) for text in texts]

    def encode_query(self, text: str) -> SparseVector:
        """Encode a query as weight 1.0 per distinct term.

        Needs no statistics: the document side already carries IDF and term
        saturation, so the dot product is the BM25 score. This is what lets the
        retrieve CLI run against a collection it did not build.
        """
        terms = self._tokenizer.tokenize(text)
        return SparseVector.from_weights({term_index(term): 1.0 for term in terms})

    def _encode_document(self, text: str) -> SparseVector:
        """BM25 weight per distinct term in one chunk."""
        terms = self._tokenizer.tokenize(text)
        if not terms:
            return SparseVector((), ())
        length = len(terms)
        # Guard against a degenerate fit (all-stopword corpus) so a zero average
        # length cannot turn the normaliser into a division by zero.
        average = self._average_length or float(length)
        weights: dict[int, float] = {}
        for term, frequency in Counter(terms).items():
            denominator = frequency + self._k1 * (1 - self._b + self._b * length / average)
            saturated = frequency * (self._k1 + 1) / denominator
            weights[term_index(term)] = self._idf(term) * saturated
        return SparseVector.from_weights(weights)

    def _idf(self, term: str) -> float:
        """Probabilistic IDF with the +1 that keeps it non-negative.

        A term appearing in every document scores near zero rather than negative;
        an unseen term (possible when encoding a document outside the fitted
        corpus) is treated as maximally rare, which is the same behaviour as a
        term seen exactly once.
        """
        frequency = self._document_frequency.get(term, 0)
        numerator = self._document_count - frequency + 0.5
        return math.log(1 + numerator / (frequency + 0.5))


def build_sparse_encoder(
    *,
    method: str = BM25_METHOD,
    k1: float = 1.2,
    b: float = 0.75,
    lowercase: bool = True,
    stopwords: str = "english",
) -> Bm25Encoder:
    """Construct the sparse encoder named by ``ingest.sparse``.

    Raises:
        SparseError: An unimplemented method. Declaring ``method: splade`` in the
            config file must fail loudly rather than silently produce BM25
            vectors that an eval would then attribute to the wrong retriever.
    """
    if method.strip().lower() != BM25_METHOD:
        raise SparseError(
            f"unsupported sparse method {method!r}; only {BM25_METHOD!r} is implemented"
        )
    return Bm25Encoder(
        k1=k1,
        b=b,
        tokenizer=Bm25Tokenizer(lowercase=lowercase, stopwords=stopwords),
    )
