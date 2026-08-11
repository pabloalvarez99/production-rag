"""Embedding providers behind one protocol.

Two implementations, and the default is the fake one on purpose:

:class:`FakeEmbeddingProvider`
    32 dimensions, derived deterministically from the SHA-256 of the text. It
    makes the whole ingest path — chunking, ids, payload, upsert, count —
    runnable and testable with no credential, no network and no cost. A
    portfolio repository that cannot be run by a reader is a screenshot.

:class:`OpenAIEmbeddingProvider`
    ``text-embedding-3-small`` by default. Real vectors, real money, and the
    only path that needs ``OPENAI_API_KEY``.

Both satisfy :class:`EmbeddingProvider`, so everything downstream — the
pipeline, the store, M2's hybrid retrieval — depends on the protocol and never
on which one is in use. The API key is read from the environment, is never
logged, and never appears in an error message.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import structlog

from production_rag.config_loader import EmbeddingConfig

_log = structlog.get_logger(__name__)

FAKE_EMBEDDER = "fake"
OPENAI_EMBEDDER = "openai"
EMBEDDER_KINDS = (FAKE_EMBEDDER, OPENAI_EMBEDDER)

FAKE_DIMENSIONS = 32
"""Dimensionality of the deterministic test embedder.

Small enough that a whole vector fits in a log line or an assertion, large
enough that unrelated texts are not accidentally near-parallel.
"""

FAKE_MODEL_NAME = "fake-deterministic-v1"
"""Recorded in every payload, so a fake-embedded collection is never mistaken
for a real one after the fact."""


class EmbeddingError(RuntimeError):
    """An embedding call failed or the provider is not usable as configured."""


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turns text into vectors of a fixed dimensionality."""

    @property
    def model(self) -> str:
        """Model identifier, stamped into every Qdrant payload."""

    @property
    def dimensions(self) -> int:
        """Vector length. Must match the collection's dense vector size."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of documents, returning one vector per input in order."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query.

        Separate from :meth:`embed_documents` because asymmetric models
        (and OpenAI's future ones) may prefix the two differently. Same
        implementation for both providers today; the seam is what matters.
        """


class FakeEmbeddingProvider:
    """Deterministic pseudo-embeddings derived from the text's SHA-256.

    Properties that make it useful rather than merely cheap:

    * **Deterministic across processes.** Built on :mod:`hashlib`, never on
      ``hash()``, which is salted per interpreter — an id or vector derived from
      ``hash()`` changes on every run and quietly breaks idempotent upserts.
    * **Unit length.** The collection uses cosine distance, where magnitude is
      irrelevant but a zero vector is undefined. Normalising here keeps the
      store honest.
    * **Same text, same vector; different text, different vector.** Enough to
      exercise identity, batching and upsert semantics end to end.

    It carries no semantics whatsoever. Retrieval *quality* can only be measured
    with a real embedder; this one measures that the plumbing is correct.
    """

    def __init__(self, dimensions: int = FAKE_DIMENSIONS) -> None:
        if dimensions <= 0:
            raise ValueError(f"dimensions must be positive, got {dimensions}")
        self._dimensions = dimensions

    @property
    def model(self) -> str:
        """Fixed identifier for the fake provider."""
        return FAKE_MODEL_NAME

    @property
    def dimensions(self) -> int:
        """Configured vector length."""
        return self._dimensions

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed every input deterministically."""
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed a query with the same function used for documents."""
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        """Expand the digest of *text* into a unit vector of the right length."""
        payload = text.encode("utf-8")
        raw = bytearray()
        counter = 0
        needed = self._dimensions * 4
        while len(raw) < needed:
            raw.extend(hashlib.sha256(counter.to_bytes(4, "big") + payload).digest())
            counter += 1

        values = [
            # Map each 4-byte word into [-1, 1); the exact mapping is arbitrary,
            # only its determinism matters.
            struct.unpack_from(">I", raw, offset * 4)[0] / 2**31 - 1.0
            for offset in range(self._dimensions)
        ]
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0.0:  # pragma: no cover - needs a 256-bit digest of all zeros
            values[0] = 1.0
            norm = 1.0
        return [value / norm for value in values]


class OpenAIEmbeddingProvider:
    """Embeddings from the OpenAI API.

    The client is constructed lazily so importing this module — which the CLI and
    the tests always do — never requires the ``openai`` package to be importable
    or a credential to be present.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
        batch_size: int = 128,
        max_retries: int = 5,
        timeout_seconds: float = 30.0,
    ) -> None:
        if not api_key:
            raise EmbeddingError(
                "the OpenAI embedder needs a credential: set OPENAI_API_KEY, "
                "or run with --embedder fake"
            )
        self._api_key = api_key
        self._model = model
        self._dimensions = dimensions
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._timeout_seconds = timeout_seconds
        self._client: object | None = None

    @property
    def model(self) -> str:
        """The configured OpenAI embedding model."""
        return self._model

    @property
    def dimensions(self) -> int:
        """Requested output dimensionality."""
        return self._dimensions

    def _get_client(self) -> object:
        """Build the OpenAI client on first use."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - dependency is declared
                raise EmbeddingError(
                    "the openai package is not installed; run with --embedder fake "
                    'or reinstall with pip install -e "."'
                ) from exc
            # Retries and timeout are handled by the SDK: it already knows which
            # status codes are retryable and applies jittered backoff.
            self._client = OpenAI(
                api_key=self._api_key,
                max_retries=self._max_retries,
                timeout=self._timeout_seconds,
            )
        return self._client

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed *texts* in batches, preserving input order."""
        if not texts:
            return []
        client = self._get_client()
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            _log.debug("embedding_batch", size=len(batch), model=self._model)
            vectors.extend(self._embed_batch(client, batch))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self.embed_documents([text])[0]

    def _embed_batch(self, client: object, batch: list[str]) -> list[list[float]]:
        """Call the embeddings endpoint once and unpack the response in order."""
        try:
            response = client.embeddings.create(  # type: ignore[attr-defined]
                model=self._model,
                input=batch,
                dimensions=self._dimensions,
            )
        except Exception as exc:  # noqa: BLE001 - provider errors are opaque by design
            # The message is included; the credential never is, and the SDK does
            # not put it in exception text.
            raise EmbeddingError(f"embedding request failed: {type(exc).__name__}: {exc}") from exc

        ordered = sorted(response.data, key=lambda item: item.index)
        return [list(item.embedding) for item in ordered]


def build_embedder(
    kind: str,
    *,
    config: EmbeddingConfig,
    api_key: str | None = None,
    fake_dimensions: int = FAKE_DIMENSIONS,
) -> EmbeddingProvider:
    """Construct an embedding provider by name.

    Args:
        kind: ``"fake"`` or ``"openai"``.
        config: The ``ingest.embedding`` block, for model name and batching.
        api_key: Credential for the real provider. Ignored by the fake one.
        fake_dimensions: Vector length for the fake provider.

    Returns:
        A provider satisfying :class:`EmbeddingProvider`.

    Raises:
        EmbeddingError: Unknown *kind*, or the real provider has no credential.
    """
    if kind == FAKE_EMBEDDER:
        return FakeEmbeddingProvider(dimensions=fake_dimensions)
    if kind == OPENAI_EMBEDDER:
        return OpenAIEmbeddingProvider(
            api_key=api_key or "",
            model=config.model,
            dimensions=config.dimensions,
            batch_size=config.batch_size,
            max_retries=config.max_retries,
            timeout_seconds=config.timeout_seconds,
        )
    raise EmbeddingError(f"unknown embedder {kind!r}; expected one of {', '.join(EMBEDDER_KINDS)}")
