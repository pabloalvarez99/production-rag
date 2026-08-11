"""Vector store: the Qdrant collection, plus an in-memory stand-in.

The pipeline depends on the :class:`VectorStore` protocol, never on Qdrant
directly. That buys two concrete things:

* :class:`InMemoryVectorStore` runs the entire write path — ensure, skip-check,
  upsert, count — with no container running, which is how the unit suite stays
  offline. (``--dry-run`` stops earlier still: it walks and chunks, and never
  reaches a store or an embedding call.)
* M2 adds a named sparse vector to the same collection. Whatever changes, the
  pipeline keeps talking to the same five methods.

**One named vector, "dense", cosine distance.** Named from the start even though
M1 writes only one: adding a second named vector to a collection created with an
unnamed default vector requires recreating the collection and re-embedding the
corpus, which is the expensive mistake this avoids.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol

import structlog
from qdrant_client import QdrantClient, models

from production_rag.config_loader import HnswConfig, PayloadIndexConfig
from production_rag.ingest.models import Chunk

_log = structlog.get_logger(__name__)

DENSE_VECTOR_NAME = "dense"
"""The only named vector M1 writes. Sparse joins it in M2."""

DEFAULT_UPSERT_BATCH = 128
"""Points per upsert call. Large enough to amortise the round-trip, small enough
that a failure is retryable without re-sending the whole corpus."""


class VectorStoreError(RuntimeError):
    """The vector store cannot be used as configured."""


class CollectionMismatchError(VectorStoreError):
    """The live collection's vector size differs from the embedder's.

    Raised instead of letting the upsert fail per-point, because the fix is a
    decision only the operator can make: re-embed the corpus with the other
    model, or recreate the collection and lose what is in it.
    """


class VectorStore(Protocol):
    """The contract the ingest pipeline needs from a vector store."""

    @property
    def collection(self) -> str:
        """Name of the collection being written to."""

    def ensure_collection(
        self, *, vector_size: int, distance: str = "Cosine", recreate: bool = False
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
    ) -> int:
        """Write chunks and their vectors; return the number of points written."""

    def count(self) -> int:
        """Exact number of points in the collection."""


def _build_points(
    chunks: Sequence[Chunk],
    vectors: Sequence[Sequence[float]],
    *,
    ingest_run_id: str,
    embedded_model: str,
) -> list[models.PointStruct]:
    """Pair chunks with their vectors into Qdrant points.

    The length check is not defensive noise: a provider that returns fewer
    vectors than inputs, or reorders them, would silently attach one chunk's
    vector to another chunk's payload. That defect is invisible in the store and
    only shows up as inexplicably bad retrieval.
    """
    if len(chunks) != len(vectors):
        raise VectorStoreError(
            f"embedder returned {len(vectors)} vectors for {len(chunks)} chunks; refusing to upsert"
        )
    return [
        models.PointStruct(
            id=chunk.point_id,
            vector={DENSE_VECTOR_NAME: list(vector)},
            payload=chunk.to_payload(ingest_run_id=ingest_run_id, embedded_model=embedded_model),
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]


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
        client: QdrantClient | None = None,
    ) -> None:
        """Configure the store. No connection is opened until a call is made."""
        self._collection = collection
        self._hnsw = hnsw or HnswConfig()
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
        self, *, vector_size: int, distance: str = "Cosine", recreate: bool = False
    ) -> bool:
        """Make the collection exist with the right dense vector configuration.

        Args:
            vector_size: Dimensionality of the embedder in use.
            distance: Qdrant distance name; cosine for normalised embeddings.
            recreate: Drop an existing collection first. Destructive, and the
                only way to change vector size or switch embedding model.

        Returns:
            ``True`` when the collection was created by this call.

        Raises:
            CollectionMismatchError: The collection exists with a different dense
                vector size, so the corpus in it was embedded by another model.
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
                _log.info(
                    "collection_present", collection=self._collection, vector_size=vector_size
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
    ) -> int:
        """Upsert chunks in batches, returning how many points were written."""
        points = _build_points(
            chunks, vectors, ingest_run_id=ingest_run_id, embedded_model=embedded_model
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


class InMemoryVectorStore:
    """A :class:`VectorStore` that keeps points in a dict.

    Not a mock: it enforces the same invariants as the real store (vector size
    must match the collection, vectors and chunks must line up, ids overwrite
    rather than duplicate), which is what makes it usable both as the
    ``--dry-run`` backend and as the unit suite's store.
    """

    def __init__(self, collection: str = "production_rag") -> None:
        self._collection = collection
        self._points: dict[str, dict[str, object]] = {}
        self._vectors: dict[str, list[float]] = {}
        self._vector_size: int | None = None
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

    def ensure_collection(
        self, *, vector_size: int, distance: str = "Cosine", recreate: bool = False
    ) -> bool:
        """Record the vector size, mirroring the real mismatch check."""
        del distance
        if recreate:
            self._points.clear()
            self._vectors.clear()
            self._vector_size = None
        if self._vector_size is None:
            self._vector_size = vector_size
            self.created = True
            return True
        if self._vector_size != vector_size:
            raise CollectionMismatchError(
                f"collection {self._collection!r} stores {self._vector_size}-dimensional vectors "
                f"but the embedder produces {vector_size}"
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
    ) -> int:
        """Store payloads and vectors, overwriting any point with the same id."""
        # Built through _build_points for the length check, then written from the
        # inputs: reading the vector back out of a PointStruct means unpacking the
        # client's wide vector union for no gain.
        _build_points(chunks, vectors, ingest_run_id=ingest_run_id, embedded_model=embedded_model)
        for chunk, vector in zip(chunks, vectors, strict=True):
            if self._vector_size is not None and len(vector) != self._vector_size:
                raise CollectionMismatchError(
                    f"vector of length {len(vector)} does not fit a "
                    f"{self._vector_size}-dimensional collection"
                )
            self._points[chunk.point_id] = chunk.to_payload(
                ingest_run_id=ingest_run_id, embedded_model=embedded_model
            )
            self._vectors[chunk.point_id] = list(vector)
        return len(chunks)

    def count(self) -> int:
        """Number of stored points."""
        return len(self._points)


def _batched[T](items: Sequence[T], size: int) -> Iterable[Sequence[T]]:
    """Yield consecutive slices of *items* of at most *size* elements."""
    if size <= 0:
        raise ValueError(f"batch size must be positive, got {size}")
    for start in range(0, len(items), size):
        yield items[start : start + size]
