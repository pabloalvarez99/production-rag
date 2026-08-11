"""Vector store: the Qdrant collection, plus an in-memory stand-in.

Callers depend on a protocol, never on Qdrant directly, and there are two of
them: ingest needs :class:`VectorStore` (write only), the retriever needs
:class:`SearchableVectorStore` (write plus the two search branches). Keeping them
apart means a write-side double never has to grow a search implementation to
satisfy a type.

:class:`InMemoryVectorStore` implements the searchable one end to end — ensure,
skip-check, dual upsert, count, cosine search, sparse dot-product search — with
no container running. That is how the unit suite, the retriever tests and the
source-hit eval all stay offline. (``--dry-run`` stops earlier still: it walks and
chunks, and never reaches a store or an embedding call.)

**Two named vectors: "dense" (cosine) and "sparse" (BM25 weights), one point
each.** Dense was named from the start, in M1, precisely so that M2 could add
sparse; a collection built with an unnamed default vector cannot gain a second
one without a full rebuild. M1 collections still need recreating to gain
``sparse``, because Qdrant cannot add a named vector in place either —
:meth:`QdrantStore.ensure_collection` says so instead of failing at upsert.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import structlog
from qdrant_client import QdrantClient, models

from production_rag.config_loader import HnswConfig, PayloadIndexConfig
from production_rag.ingest.models import Chunk
from production_rag.retrieval.sparse import SparseVector

_log = structlog.get_logger(__name__)

DENSE_VECTOR_NAME = "dense"
"""Named vector for embeddings. Written since M1."""

SPARSE_VECTOR_NAME = "sparse"
"""Named vector for BM25 term weights. Written since M2.

A collection created by M1 carries only ``dense``. Adding a named vector to a
live Qdrant collection is not an in-place operation, so such a collection has to
be recreated — :meth:`QdrantStore.ensure_collection` says so explicitly rather
than failing later at upsert.
"""

DEFAULT_UPSERT_BATCH = 128
"""Points per upsert call. Large enough to amortise the round-trip, small enough
that a failure is retryable without re-sending the whole corpus."""

DEFAULT_SEARCH_EF = 128
"""HNSW ``ef`` at query time; mirrors ``qdrant.search.hnsw_ef`` in the profile."""


class VectorStoreError(RuntimeError):
    """The vector store cannot be used as configured."""


class CollectionMismatchError(VectorStoreError):
    """The live collection does not match what the caller is about to write.

    Raised instead of letting the upsert fail per-point, because the fix is a
    decision only the operator can make: re-embed the corpus with the other
    model, or recreate the collection and lose what is in it.
    """


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One point returned by a search, before fusion.

    Deliberately not a Qdrant type: fusion, the retriever and the eval harness all
    consume this, and none of them should import a client model.
    """

    point_id: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        """Citable chunk id from the payload, empty when absent."""
        value = self.payload.get("chunk_id", "")
        return value if isinstance(value, str) else str(value)


@runtime_checkable
class VectorStore(Protocol):
    """The contract the ingest pipeline needs from a vector store.

    Runtime-checkable so a test can assert an implementation still satisfies the
    contract. That check verifies method *names* only, not signatures — mypy is
    what enforces the rest.
    """

    @property
    def collection(self) -> str:
        """Name of the collection being written to."""

    def ensure_collection(
        self,
        *,
        vector_size: int,
        distance: str = "Cosine",
        recreate: bool = False,
        with_sparse: bool = False,
    ) -> bool:
        """Create the collection when absent; return whether it was created."""

    def existing_point_ids(self, point_ids: Sequence[str]) -> set[str]:
        """Return the subset of *point_ids* already present in the collection."""

    def upsert_chunks(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        *,
        ingest_run_id: str,
        embedded_model: str,
        sparse_vectors: Sequence[SparseVector] | None = None,
    ) -> int:
        """Write chunks and their vectors; return the number of points written."""

    def count(self) -> int:
        """Exact number of points in the collection."""


@runtime_checkable
class SearchableVectorStore(VectorStore, Protocol):
    """A store that can also be queried.

    Separate from :class:`VectorStore` on purpose: ingest needs write methods and
    nothing else, and a write-only double should not have to grow a search
    implementation to satisfy a type. The retriever asks for this one.
    """

    def search_dense(
        self, vector: Sequence[float], *, limit: int, with_payload: Sequence[str] | None = None
    ) -> list[SearchHit]:
        """Nearest neighbours of *vector* under the dense named vector."""

    def search_sparse(
        self, vector: SparseVector, *, limit: int, with_payload: Sequence[str] | None = None
    ) -> list[SearchHit]:
        """Highest BM25 dot products against the sparse named vector."""


def _build_points(
    chunks: Sequence[Chunk],
    vectors: Sequence[Sequence[float]],
    *,
    ingest_run_id: str,
    embedded_model: str,
    sparse_vectors: Sequence[SparseVector] | None = None,
) -> list[models.PointStruct]:
    """Pair chunks with their vectors into Qdrant points.

    The length checks are not defensive noise: a provider that returns fewer
    vectors than inputs, or reorders them, would silently attach one chunk's
    vector to another chunk's payload. That defect is invisible in the store and
    only shows up as inexplicably bad retrieval.

    Dense and sparse travel in the same ``PointStruct`` — one upsert, one point —
    which is what makes it structurally impossible for the two indexes to drift
    out of sync.
    """
    if len(chunks) != len(vectors):
        raise VectorStoreError(
            f"embedder returned {len(vectors)} vectors for {len(chunks)} chunks; refusing to upsert"
        )
    if sparse_vectors is not None and len(sparse_vectors) != len(chunks):
        raise VectorStoreError(
            f"sparse encoder returned {len(sparse_vectors)} vectors for {len(chunks)} chunks; "
            "refusing to upsert"
        )

    points: list[models.PointStruct] = []
    for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
        named: dict[str, Any] = {DENSE_VECTOR_NAME: list(vector)}
        if sparse_vectors is not None:
            sparse = sparse_vectors[index]
            named[SPARSE_VECTOR_NAME] = models.SparseVector(
                indices=list(sparse.indices), values=list(sparse.values)
            )
        points.append(
            models.PointStruct(
                id=chunk.point_id,
                vector=named,
                payload=chunk.to_payload(
                    ingest_run_id=ingest_run_id, embedded_model=embedded_model
                ),
            )
        )
    return points


class QdrantStore:
    """Qdrant-backed implementation of :class:`VectorStore`."""

    def __init__(
        self,
        *,
        url: str,
        collection: str,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        hnsw: HnswConfig | None = None,
        payload_indexes: Sequence[PayloadIndexConfig] = (),
        on_disk_payload: bool = True,
        wait_for_writes: bool = True,
        upsert_batch_size: int = DEFAULT_UPSERT_BATCH,
        search_hnsw_ef: int = DEFAULT_SEARCH_EF,
        client: QdrantClient | None = None,
    ) -> None:
        """Configure the store. No connection is opened until a call is made."""
        self._collection = collection
        self._hnsw = hnsw or HnswConfig()
        self._search_hnsw_ef = search_hnsw_ef
        self._payload_indexes = tuple(payload_indexes)
        self._on_disk_payload = on_disk_payload
        self._wait = wait_for_writes
        self._batch = upsert_batch_size
        self._client = client or QdrantClient(
            url=url,
            api_key=api_key or None,
            timeout=int(timeout_seconds),
        )

    @property
    def collection(self) -> str:
        """Name of the target collection."""
        return self._collection

    @property
    def client(self) -> QdrantClient:
        """The underlying client, for integration tests and one-off inspection."""
        return self._client

    def ensure_collection(
        self,
        *,
        vector_size: int,
        distance: str = "Cosine",
        recreate: bool = False,
        with_sparse: bool = False,
    ) -> bool:
        """Make the collection exist with the right vector configuration.

        Args:
            vector_size: Dimensionality of the embedder in use.
            distance: Qdrant distance name; cosine for normalised embeddings.
            recreate: Drop an existing collection first. Destructive, and the
                only way to change vector size, switch embedding model, or add a
                named vector to a collection that predates it.
            with_sparse: Declare the ``sparse`` named vector. Required before any
                BM25 weights can be written.

        Returns:
            ``True`` when the collection was created by this call.

        Raises:
            CollectionMismatchError: The collection exists with a different dense
                vector size, or without the sparse vector this run needs.
            VectorStoreError: Qdrant is unreachable.
        """
        try:
            exists = self._client.collection_exists(self._collection)
            if exists and recreate:
                _log.warning("collection_recreating", collection=self._collection)
                self._client.delete_collection(self._collection)
                exists = False
            if exists:
                self._assert_vector_size(vector_size)
                if with_sparse:
                    self._assert_sparse_declared()
                _log.info(
                    "collection_present",
                    collection=self._collection,
                    vector_size=vector_size,
                    with_sparse=with_sparse,
                )
                return False

            self._client.create_collection(
                collection_name=self._collection,
                vectors_config={
                    DENSE_VECTOR_NAME: models.VectorParams(
                        size=vector_size,
                        distance=models.Distance(distance),
                    )
                },
                # Sparse is declared only when it will be written. Declaring an
                # empty sparse vector would make search_sparse return nothing and
                # look like a ranking bug rather than a missing backfill.
                sparse_vectors_config=(
                    {SPARSE_VECTOR_NAME: models.SparseVectorParams()} if with_sparse else None
                ),
                hnsw_config=models.HnswConfigDiff(
                    m=self._hnsw.m,
                    ef_construct=self._hnsw.ef_construct,
                    full_scan_threshold=self._hnsw.full_scan_threshold,
                ),
                on_disk_payload=self._on_disk_payload,
            )
            self._create_payload_indexes()
            _log.info(
                "collection_created",
                collection=self._collection,
                vector_size=vector_size,
                distance=distance,
                with_sparse=with_sparse,
            )
            return True
        except (CollectionMismatchError, VectorStoreError):
            raise
        except Exception as exc:  # noqa: BLE001 - client raises a wide family
            raise VectorStoreError(
                f"could not prepare collection {self._collection!r}: {type(exc).__name__}: {exc}"
            ) from exc

    def _assert_vector_size(self, expected: int) -> None:
        """Fail loudly when the live collection was built for another model."""
        info = self._client.get_collection(self._collection)
        vectors = info.config.params.vectors
        if not isinstance(vectors, dict) or DENSE_VECTOR_NAME not in vectors:
            raise CollectionMismatchError(
                f"collection {self._collection!r} has no named vector "
                f"{DENSE_VECTOR_NAME!r}; recreate it with --recreate-collection"
            )
        actual = vectors[DENSE_VECTOR_NAME].size
        if actual != expected:
            raise CollectionMismatchError(
                f"collection {self._collection!r} stores {actual}-dimensional vectors but the "
                f"embedder produces {expected}; re-run with --recreate-collection to rebuild it "
                "(this deletes the current index)"
            )

    def _assert_sparse_declared(self) -> None:
        """Fail loudly when a pre-M2 collection is asked to hold sparse vectors.

        Qdrant cannot add a named vector to an existing collection, so this is a
        migration, not a retry. The message says exactly which flag performs it
        and what that costs — a re-embed of the whole corpus.
        """
        info = self._client.get_collection(self._collection)
        sparse = info.config.params.sparse_vectors
        if not sparse or SPARSE_VECTOR_NAME not in sparse:
            raise CollectionMismatchError(
                f"collection {self._collection!r} was created without the "
                f"{SPARSE_VECTOR_NAME!r} named vector (M1 layout). Qdrant cannot add one in "
                "place: re-run the ingest with --recreate-collection to rebuild it, which "
                "deletes the current index and re-embeds the corpus"
            )

    def _create_payload_indexes(self) -> None:
        """Add the keyword indexes that keep metadata filters cheap."""
        for index in self._payload_indexes:
            self._client.create_payload_index(
                collection_name=self._collection,
                field_name=index.field,
                field_schema=models.PayloadSchemaType(index.schema_),
            )
            _log.debug("payload_index_created", field=index.field, schema=index.schema_)

    def existing_point_ids(self, point_ids: Sequence[str]) -> set[str]:
        """Return which of *point_ids* Qdrant already holds.

        Used by incremental ingest. Point ids are content-addressed, so a hit
        means "this exact text was already embedded by a previous run" — the
        expensive call can be skipped entirely.
        """
        if not point_ids:
            return set()
        found: set[str] = set()
        for batch in _batched(point_ids, self._batch):
            records = self._client.retrieve(
                collection_name=self._collection,
                ids=list(batch),
                with_payload=False,
                with_vectors=False,
            )
            found.update(str(record.id) for record in records)
        return found

    def upsert_chunks(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        *,
        ingest_run_id: str,
        embedded_model: str,
        sparse_vectors: Sequence[SparseVector] | None = None,
    ) -> int:
        """Upsert chunks in batches, returning how many points were written."""
        points = _build_points(
            chunks,
            vectors,
            ingest_run_id=ingest_run_id,
            embedded_model=embedded_model,
            sparse_vectors=sparse_vectors,
        )
        written = 0
        try:
            for batch in _batched(points, self._batch):
                self._client.upsert(
                    collection_name=self._collection,
                    points=list(batch),
                    wait=self._wait,
                )
                written += len(batch)
                _log.debug("points_upserted", count=len(batch), collection=self._collection)
        except Exception as exc:  # noqa: BLE001 - client raises a wide family
            raise VectorStoreError(
                f"upsert into {self._collection!r} failed after {written} points: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return written

    def count(self) -> int:
        """Exact point count, so a post-ingest assertion means something."""
        try:
            return self._client.count(collection_name=self._collection, exact=True).count
        except Exception as exc:  # noqa: BLE001 - client raises a wide family
            raise VectorStoreError(
                f"could not count points in {self._collection!r}: {type(exc).__name__}: {exc}"
            ) from exc

    def search_dense(
        self, vector: Sequence[float], *, limit: int, with_payload: Sequence[str] | None = None
    ) -> list[SearchHit]:
        """Nearest neighbours under the ``dense`` named vector.

        Args:
            vector: Query embedding, from the same model that built the index.
            limit: Candidates to return. For hybrid this is ``dense_top_k``, not
                the final ``top_k``: fusion can only reorder what it receives.
            with_payload: Payload fields to fetch. ``None`` fetches all; naming
                the fields keeps the response small on a corpus whose payload
                carries the full chunk text.

        Returns:
            Hits ordered by cosine similarity, descending.

        Raises:
            VectorStoreError: Qdrant refused or is unreachable.
        """
        return self._query(list(vector), using=DENSE_VECTOR_NAME, limit=limit, fields=with_payload)

    def search_sparse(
        self, vector: SparseVector, *, limit: int, with_payload: Sequence[str] | None = None
    ) -> list[SearchHit]:
        """Highest BM25 dot products under the ``sparse`` named vector.

        The returned score *is* the BM25 score, because ingest stored complete
        BM25 weights and the query vector carries 1.0 per distinct term. Nothing
        in Qdrant reweights it — see :mod:`production_rag.retrieval.sparse`.

        Raises:
            VectorStoreError: Qdrant refused or is unreachable.
        """
        if vector.is_empty:
            # Every query term was a stopword or out of the tokenizer's alphabet.
            # An empty sparse query is a legitimate outcome, not an error; the
            # dense branch still has something to say.
            _log.info("sparse_query_empty", collection=self._collection)
            return []
        query = models.SparseVector(indices=list(vector.indices), values=list(vector.values))
        return self._query(query, using=SPARSE_VECTOR_NAME, limit=limit, fields=with_payload)

    def _query(
        self,
        query: Any,
        *,
        using: str,
        limit: int,
        fields: Sequence[str] | None,
    ) -> list[SearchHit]:
        """Run one branch of the search and normalise the response.

        Both branches share this so the payload projection, the ``ef`` setting and
        the error wrapping cannot drift apart between dense and sparse.
        """
        if limit <= 0:
            raise VectorStoreError(f"search limit must be positive, got {limit}")
        try:
            response = self._client.query_points(
                collection_name=self._collection,
                query=query,
                using=using,
                limit=limit,
                with_payload=list(fields) if fields else True,
                # ef applies to the HNSW graph only; Qdrant ignores it for the
                # sparse inverted index, so passing it unconditionally is safe.
                search_params=models.SearchParams(hnsw_ef=self._search_hnsw_ef),
            )
        except Exception as exc:  # noqa: BLE001 - client raises a wide family
            raise VectorStoreError(
                f"search on {self._collection!r} vector {using!r} failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return [
            SearchHit(
                point_id=str(point.id),
                score=float(point.score),
                payload=dict(point.payload or {}),
            )
            for point in response.points
        ]


class InMemoryVectorStore:
    """A :class:`SearchableVectorStore` that keeps points in dicts.

    Not a mock: it enforces the same invariants as the real store (vector size
    must match the collection, vectors and chunks must line up, ids overwrite
    rather than duplicate, sparse must be declared before it is written) and it
    ranks with the same two similarity functions — cosine for dense, dot product
    for sparse. That is what lets the retriever, RRF and the source-hit eval be
    exercised end to end with no container running.

    Brute force on purpose: an approximate index here would introduce a second
    source of ranking differences and make a failing offline test ambiguous.
    """

    def __init__(self, collection: str = "production_rag") -> None:
        self._collection = collection
        self._points: dict[str, dict[str, object]] = {}
        self._vectors: dict[str, list[float]] = {}
        self._sparse: dict[str, SparseVector] = {}
        self._vector_size: int | None = None
        self._with_sparse = False
        self.created = False

    @property
    def collection(self) -> str:
        """Name of the in-memory collection."""
        return self._collection

    @property
    def points(self) -> dict[str, dict[str, object]]:
        """Stored payloads by point id, for assertions."""
        return self._points

    @property
    def vectors(self) -> dict[str, list[float]]:
        """Stored vectors by point id, for assertions."""
        return self._vectors

    @property
    def sparse_vectors(self) -> dict[str, SparseVector]:
        """Stored sparse vectors by point id, for assertions."""
        return self._sparse

    def ensure_collection(
        self,
        *,
        vector_size: int,
        distance: str = "Cosine",
        recreate: bool = False,
        with_sparse: bool = False,
    ) -> bool:
        """Record the vector configuration, mirroring the real mismatch checks."""
        del distance
        if recreate:
            self._points.clear()
            self._vectors.clear()
            self._sparse.clear()
            self._vector_size = None
            self._with_sparse = False
        if self._vector_size is None:
            self._vector_size = vector_size
            self._with_sparse = with_sparse
            self.created = True
            return True
        if self._vector_size != vector_size:
            raise CollectionMismatchError(
                f"collection {self._collection!r} stores {self._vector_size}-dimensional vectors "
                f"but the embedder produces {vector_size}"
            )
        if with_sparse and not self._with_sparse:
            # Same migration rule as Qdrant, enforced offline so a test catches
            # the mistake before an operator meets it.
            raise CollectionMismatchError(
                f"collection {self._collection!r} was created without the "
                f"{SPARSE_VECTOR_NAME!r} named vector; recreate it with --recreate-collection"
            )
        return False

    def existing_point_ids(self, point_ids: Sequence[str]) -> set[str]:
        """Return which ids are already stored."""
        return {point_id for point_id in point_ids if point_id in self._points}

    def upsert_chunks(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        *,
        ingest_run_id: str,
        embedded_model: str,
        sparse_vectors: Sequence[SparseVector] | None = None,
    ) -> int:
        """Store payloads and vectors, overwriting any point with the same id."""
        # Built through _build_points for the length checks, then written from the
        # inputs: reading the vector back out of a PointStruct means unpacking the
        # client's wide vector union for no gain.
        _build_points(
            chunks,
            vectors,
            ingest_run_id=ingest_run_id,
            embedded_model=embedded_model,
            sparse_vectors=sparse_vectors,
        )
        if sparse_vectors is not None and not self._with_sparse:
            raise CollectionMismatchError(
                f"collection {self._collection!r} was created without the "
                f"{SPARSE_VECTOR_NAME!r} named vector; recreate it with --recreate-collection"
            )
        for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
            if self._vector_size is not None and len(vector) != self._vector_size:
                raise CollectionMismatchError(
                    f"vector of length {len(vector)} does not fit a "
                    f"{self._vector_size}-dimensional collection"
                )
            self._points[chunk.point_id] = chunk.to_payload(
                ingest_run_id=ingest_run_id, embedded_model=embedded_model
            )
            self._vectors[chunk.point_id] = list(vector)
            if sparse_vectors is not None:
                self._sparse[chunk.point_id] = sparse_vectors[index]
        return len(chunks)

    def count(self) -> int:
        """Number of stored points."""
        return len(self._points)

    def search_dense(
        self, vector: Sequence[float], *, limit: int, with_payload: Sequence[str] | None = None
    ) -> list[SearchHit]:
        """Brute-force cosine search, matching the real store's distance."""
        if limit <= 0:
            raise VectorStoreError(f"search limit must be positive, got {limit}")
        if self._vector_size is not None and len(vector) != self._vector_size:
            raise CollectionMismatchError(
                f"query vector of length {len(vector)} does not fit a "
                f"{self._vector_size}-dimensional collection"
            )
        query = list(vector)
        scored = [(point_id, _cosine(query, stored)) for point_id, stored in self._vectors.items()]
        return self._top(scored, limit=limit, fields=with_payload)

    def search_sparse(
        self, vector: SparseVector, *, limit: int, with_payload: Sequence[str] | None = None
    ) -> list[SearchHit]:
        """Brute-force dot product, which for these vectors is the BM25 score."""
        if limit <= 0:
            raise VectorStoreError(f"search limit must be positive, got {limit}")
        if vector.is_empty:
            return []
        scored = [
            (point_id, vector.dot(stored))
            for point_id, stored in self._sparse.items()
            # A document sharing no term with the query scores 0 and would only
            # pad the candidate list with noise for fusion to rank.
            if vector.dot(stored) > 0.0
        ]
        return self._top(scored, limit=limit, fields=with_payload)

    def _top(
        self,
        scored: Sequence[tuple[str, float]],
        *,
        limit: int,
        fields: Sequence[str] | None,
    ) -> list[SearchHit]:
        """Sort, truncate and project — the shared tail of both searches.

        Ties break on the point id: Qdrant's order among equal scores is not
        specified, and a test that depends on dict insertion order is a test that
        will fail for the wrong reason later.
        """
        ordered = sorted(scored, key=lambda pair: (-pair[1], pair[0]))[:limit]
        return [
            SearchHit(
                point_id=point_id,
                score=score,
                payload=_project(self._points[point_id], fields),
            )
            for point_id, score in ordered
        ]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity, with a zero vector scoring 0 instead of dividing by 0."""
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return 0.0 if norm == 0.0 else dot / norm


def _project(payload: Mapping[str, object], fields: Sequence[str] | None) -> dict[str, Any]:
    """Copy *payload*, keeping only *fields* when a projection was requested."""
    if not fields:
        return dict(payload)
    return {key: payload[key] for key in fields if key in payload}


def _batched[T](items: Sequence[T], size: int) -> Iterable[Sequence[T]]:
    """Yield consecutive slices of *items* of at most *size* elements."""
    if size <= 0:
        raise ValueError(f"batch size must be positive, got {size}")
    for start in range(0, len(items), size):
        yield items[start : start + size]
