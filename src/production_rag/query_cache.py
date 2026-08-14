"""Optional in-process cache for grounded query results.

A cache that forgets filters is worse than no cache: it would serve an
unfiltered answer to a filtered question (or the reverse) and look correct.
The key therefore includes every input that can change which evidence the
answer is allowed to rest on — collection identity, the question text, the
canonical filter expression, embedder id, llm id, and a fingerprint of the
retrieval knobs that decide ranking.

This is deliberately **not** Redis. A single-process demo and the unit suite
need a seam a hiring manager can read in one file; a distributed cache needs a
deployment, a TTL policy, and a reason to share answers across tenants. None of
those apply on the free path. When the process restarts the cache is empty, and
that is fine: cold-start cost is one query, not a correctness bug.

Default off. The local demo turns it on via ``CACHE_ENABLED=true`` (compose)
or ``cache.enabled`` in the YAML profile. Production-shaped configs leave both
false so a multi-worker deploy cannot accidentally share answers it never keyed
for concurrency.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from production_rag.api.schemas import QueryResponse

CacheStatus = Literal["hit", "miss"]


def canonical_filters(filters: Mapping[str, Any] | None) -> str:
    """Stable string form of *filters* for keying.

    Field order and list order must not create two keys for one request. Empty
    and omitted both mean "unfiltered" and share one sentinel, so a miss under
    a filter can never be filled by an earlier unfiltered hit that used a
    different empty representation.
    """
    if not filters:
        return ""
    normalized: dict[str, Any] = {}
    for key in sorted(filters):
        value = filters[key]
        if isinstance(value, list | tuple):
            normalized[key] = sorted(str(item) for item in value)
        else:
            normalized[key] = str(value)
    return json.dumps(normalized, separators=(",", ":"), ensure_ascii=False)


def retrieval_fingerprint(
    *,
    mode: str | None,
    top_k: int | None,
    dense_top_k: int | None = None,
    sparse_top_k: int | None = None,
    rrf_k: int | None = None,
    rerank: str | None = None,
) -> str:
    """Compact fingerprint of ranking knobs that change which hits are returned."""
    payload = {
        "mode": mode or "",
        "top_k": top_k,
        "dense_top_k": dense_top_k,
        "sparse_top_k": sparse_top_k,
        "rrf_k": rrf_k,
        "rerank": rerank or "",
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


@dataclass(frozen=True, slots=True)
class CacheKey:
    """Everything that must match before a stored answer may be reused."""

    collection: str
    query: str
    filters: str
    embedder_id: str
    llm_id: str
    retrieval: str
    # Corpus hash (and friends) so two corpora that share a collection *name*
    # still cannot cross-hit. Empty string means "identity not productized yet"
    # and remains backward-compatible with pre-season keys.
    corpus_identity: str = ""

    def digest(self) -> str:
        """Fixed-length key for the map. SHA-256 of the structured fields."""
        material = "\n".join(
            (
                self.collection,
                self.query,
                self.filters,
                self.embedder_id,
                self.llm_id,
                self.retrieval,
                self.corpus_identity,
            )
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()


class QueryResultCache:
    """Thread-safe LRU of :class:`~production_rag.api.schemas.QueryResponse`.

    Entries store the public response only — never prompts, never credentials.
    """

    def __init__(self, *, max_entries: int = 256) -> None:
        """Bound the map so a hot demo cannot grow without limit."""
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self._max_entries = max_entries
        self._entries: OrderedDict[str, QueryResponse] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @property
    def size(self) -> int:
        """Current entry count."""
        with self._lock:
            return len(self._entries)

    def get(self, key: CacheKey) -> tuple[QueryResponse | None, CacheStatus]:
        """Return a stored response and whether it was a hit."""
        digest = key.digest()
        with self._lock:
            response = self._entries.get(digest)
            if response is None:
                self.misses += 1
                return None, "miss"
            self._entries.move_to_end(digest)
            self.hits += 1
            # model_copy so a caller cannot mutate the stored entry through a
            # shared pydantic model instance.
            return response.model_copy(deep=True), "hit"

    def put(self, key: CacheKey, response: QueryResponse) -> None:
        """Store *response* under *key*, evicting the least-recent entry if full."""
        digest = key.digest()
        with self._lock:
            if digest in self._entries:
                self._entries.move_to_end(digest)
            self._entries[digest] = response.model_copy(deep=True)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        """Drop every entry. Used by tests and by operators after re-ingest."""
        with self._lock:
            self._entries.clear()
            self.hits = 0
            self.misses = 0


# Process-wide cache instance. Built lazily so importing the module never
# allocates state a test would have to clean up when the feature is off.
_CACHE: QueryResultCache | None = None
_CACHE_LOCK = threading.Lock()


def get_query_cache(*, max_entries: int = 256) -> QueryResultCache:
    """Return the process cache, creating it on first use."""
    global _CACHE
    with _CACHE_LOCK:
        if _CACHE is None:
            _CACHE = QueryResultCache(max_entries=max_entries)
        return _CACHE


def reset_query_cache() -> None:
    """Drop the process cache entirely. Tests call this between cases."""
    global _CACHE
    with _CACHE_LOCK:
        if _CACHE is not None:
            _CACHE.clear()
        _CACHE = None


__all__ = [
    "CacheKey",
    "CacheStatus",
    "QueryResultCache",
    "canonical_filters",
    "get_query_cache",
    "reset_query_cache",
    "retrieval_fingerprint",
]
