"""Metadata filters: an allowlist, translated into store conditions.

    {"source": "sample", "tags": ["bm25", "rrf"]}
        -> QueryFilter(conditions=(source in {sample}, tags in {bm25, rrf}))

One module owns the whole contract, and it imports nothing from a vector store
client: the allowlist check, the public-name to payload-key mapping and the value
normalisation are plain data transformations, so they are unit-testable offline
and the same rules apply to the HTTP route, the CLIs and the library. Translation
into a Qdrant expression lives in :mod:`production_rag.retrieval.store`, next to
the client that needs it.

**Fail closed.** A field that is not in ``retrieval.filters.allowed_fields`` is a
:class:`FilterError`, never a dropped condition. Silently ignoring a filter would
answer a different question than the one asked, and the caller would have no way
to tell — which is the same failure class as an unrequested provider bill: an
answer that looks right and is not.

**Exact keyword matching only.** A condition is "this field equals one of these
values"; several values on one field are OR, several fields are AND. There are no
ranges, no substring matching and no negation, because every allowlisted field is
a keyword payload and a range predicate over an unindexed field is exactly the
table scan the allowlist exists to prevent. Widening this is an ADR, not a patch.

**An unindexed field is reported, never silent.** ``qdrant.payload_indexes``
decides which filters stay O(log n). Filtering on a field without an index is
allowed — it is still the honest answer to the question asked — but it degrades to
a payload scan, so it is logged at warning level and named in the retrieval
summary rather than left for someone to discover as unexplained latency.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import structlog

_log = structlog.get_logger(__name__)

FILTER_NOT_ALLOWED = "filter_not_allowed"
"""``error_type`` for a field outside the allowlist. Part of the public contract:
a client branches on this string, never on the message."""

FILTER_INVALID_VALUE = "filter_invalid_value"
"""``error_type`` for an allowlisted field carrying a value this contract cannot
express — a number, a null, an object, or an empty list of alternatives."""

PUBLIC_TO_PAYLOAD: dict[str, str] = {
    "source": "source",
    "title": "title",
    "tags": "tags",
}
"""Public API field name -> Qdrant payload key.

The mapping is identity today and is written out anyway, because the one pair a
reader will get wrong is ``source``. **The public ``source`` field is the payload
``source`` key — the first path segment under the corpus root (``sample``), which
is keyword-indexed.** It is *not* ``source_path``, the full corpus-relative file
path (``sample/01-hybrid-search.md``), which carries no index and is not
filterable. Asking for ``source_path`` is rejected as an unknown field rather than
quietly answered against ``source``.

A public name absent from this table cannot be filtered on even if a deployment
adds it to ``retrieval.filters.allowed_fields``: the allowlist says what a
deployment permits, this table says what the payload can actually express, and a
filter needs both.
"""


class FilterError(ValueError):
    """A filter was asked for something the allowlist does not permit.

    Carries :attr:`error_type` — a stable slug — because the HTTP layer returns it
    in a 422 body and the CLIs put it in their last stdout line. An exception
    class name is an implementation detail; this string is the contract.
    """

    def __init__(self, message: str, *, error_type: str, field: str | None = None) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.field = field


@dataclass(frozen=True, slots=True)
class FilterCondition:
    """One resolved condition: *payload_key* must equal one of *values*."""

    field: str
    """The public field name, as the caller wrote it. What an error message quotes."""
    payload_key: str
    """The Qdrant payload key it maps onto. What the store actually queries."""
    values: tuple[str, ...]
    """Alternatives, OR-ed. Never empty — an empty set matches nothing, which is a
    request that can only be a mistake, so it is rejected at build time."""
    indexed: bool = False
    """Whether ``qdrant.payload_indexes`` covers ``payload_key``."""

    def matches(self, payload: Mapping[str, Any]) -> bool:
        """Whether *payload* satisfies this condition.

        Used by :class:`~production_rag.retrieval.store.InMemoryVectorStore`, and
        deliberately written to the same rule Qdrant applies to a keyword match:
        a list-valued payload (``tags``) matches when **any** element matches, a
        scalar matches on equality. An absent key never matches — a filter is a
        narrowing, so a document that cannot answer the question is out.
        """
        value = payload.get(self.payload_key)
        if value is None:
            return False
        if isinstance(value, (list, tuple, set)):
            return any(_as_text(item) in self.values for item in value)
        return _as_text(value) in self.values

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form, as reported in the retrieval summary."""
        return {
            "field": self.field,
            "payload_key": self.payload_key,
            "values": list(self.values),
            "indexed": self.indexed,
        }


@dataclass(frozen=True, slots=True)
class QueryFilter:
    """A validated conjunction of conditions, ready for a store to translate."""

    conditions: tuple[FilterCondition, ...]

    @property
    def is_empty(self) -> bool:
        """Whether this filter narrows nothing."""
        return not self.conditions

    @property
    def fields(self) -> tuple[str, ...]:
        """Public field names carried, in the order they were built."""
        return tuple(condition.field for condition in self.conditions)

    @property
    def unindexed_fields(self) -> tuple[str, ...]:
        """Fields that will cost a payload scan on Qdrant."""
        return tuple(
            condition.field for condition in self.conditions if not condition.indexed
        )

    def matches(self, payload: Mapping[str, Any]) -> bool:
        """Whether *payload* satisfies every condition."""
        return all(condition.matches(payload) for condition in self.conditions)

    def to_summary(self) -> dict[str, Any]:
        """What the retrieval result reports about the filter that was applied."""
        return {
            "applied": not self.is_empty,
            "fields": list(self.fields),
            "unindexed": list(self.unindexed_fields),
            "conditions": [condition.to_dict() for condition in self.conditions],
        }


NO_FILTER_SUMMARY: dict[str, Any] = {
    "applied": False,
    "fields": [],
    "unindexed": [],
    "conditions": [],
}
"""Summary emitted when no filter was requested. Present rather than omitted, for
the same reason as the rerank summary: a result that lacks the key is
indistinguishable from one produced before the key existed."""


@dataclass(frozen=True, slots=True)
class FilterPolicy:
    """What a deployment permits a query to filter on.

    Two independent lists, and the difference matters. ``allowed_fields`` is a
    *permission*: a field missing from it is rejected. ``indexed_fields`` is a
    *cost*: a field missing from it still works and is reported as a scan.
    """

    allowed_fields: tuple[str, ...] = ()
    indexed_fields: tuple[str, ...] = ()

    @classmethod
    def from_fields(
        cls,
        allowed: Iterable[str],
        indexed: Iterable[str] = (),
    ) -> FilterPolicy:
        """Build a policy from the two configured lists."""
        return cls(allowed_fields=tuple(allowed), indexed_fields=tuple(indexed))

    @property
    def filterable_fields(self) -> tuple[str, ...]:
        """Allowlisted names that also map onto a payload key.

        The intersection, because an allowlisted field the payload cannot express
        is a configuration mistake rather than a usable filter — and quoting the
        full allowlist in an error message would send a caller after a field that
        would never work.
        """
        return tuple(name for name in self.allowed_fields if name in PUBLIC_TO_PAYLOAD)

    def build(self, raw: Mapping[str, Any] | None) -> QueryFilter | None:
        """Validate *raw* and resolve it into a :class:`QueryFilter`.

        Args:
            raw: The caller's ``filters`` object, or ``None``.

        Returns:
            ``None`` when nothing was asked for — including for an empty mapping,
            so "no filters" is one code path rather than two. Otherwise a filter
            whose conditions are all allowlisted and all mapped.

        Raises:
            FilterError: A field is outside the allowlist, is not expressible
                against the payload, or carries a value this contract cannot
                represent. Never a partially applied filter.
        """
        if not raw:
            return None
        if not isinstance(raw, Mapping):
            raise FilterError(
                "filters must be an object mapping field names to values",
                error_type=FILTER_INVALID_VALUE,
            )

        allowed = self.filterable_fields
        indexed = frozenset(self.indexed_fields)
        conditions: list[FilterCondition] = []
        for field_name, value in raw.items():
            name = str(field_name).strip()
            if name not in allowed:
                raise FilterError(
                    f"filtering on {name!r} is not allowed; allowed fields: "
                    f"{', '.join(allowed) if allowed else '(none)'}",
                    error_type=FILTER_NOT_ALLOWED,
                    field=name,
                )
            payload_key = PUBLIC_TO_PAYLOAD[name]
            conditions.append(
                FilterCondition(
                    field=name,
                    payload_key=payload_key,
                    values=_normalise_values(name, value),
                    indexed=payload_key in indexed,
                )
            )

        query_filter = QueryFilter(conditions=tuple(conditions))
        unindexed = query_filter.unindexed_fields
        if unindexed:
            # Allowed, and never silent: without this line an unindexed filter is
            # only visible as latency nobody can attribute.
            _log.warning(
                "filter_field_unindexed",
                fields=list(unindexed),
                indexed_fields=sorted(indexed),
                note="filter degrades to a payload scan; add a payload index or drop the field",
            )
        return query_filter


def _normalise_values(field_name: str, value: Any) -> tuple[str, ...]:
    """Coerce one field's value into a non-empty tuple of alternatives.

    Strings only, on purpose. Every allowlisted field is a keyword payload, and
    accepting a number or a boolean here would produce a condition that silently
    never matches, because Qdrant compares a keyword index against keywords.
    """
    if isinstance(value, str):
        candidates: Sequence[Any] = (value,)
    elif isinstance(value, (list, tuple)):
        candidates = tuple(value)
    else:
        raise FilterError(
            f"filter {field_name!r} must be a string or a list of strings",
            error_type=FILTER_INVALID_VALUE,
            field=field_name,
        )

    values: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, str):
            raise FilterError(
                f"filter {field_name!r} must be a string or a list of strings",
                error_type=FILTER_INVALID_VALUE,
                field=field_name,
            )
        text = candidate.strip()
        if not text:
            raise FilterError(
                f"filter {field_name!r} carries an empty value",
                error_type=FILTER_INVALID_VALUE,
                field=field_name,
            )
        if text not in values:
            values.append(text)

    if not values:
        # An empty list of alternatives matches nothing. That is never what a
        # caller means, and served as an empty result it looks like a corpus gap.
        raise FilterError(
            f"filter {field_name!r} needs at least one value",
            error_type=FILTER_INVALID_VALUE,
            field=field_name,
        )
    return tuple(values)


def _as_text(value: Any) -> str:
    """Payload value as the string a keyword condition compares against."""
    return value if isinstance(value, str) else str(value)


def parse_filter_arguments(pairs: Sequence[str] | None) -> dict[str, list[str]]:
    """Turn repeated ``--filter field=value`` arguments into a filters object.

    Repeating a field accumulates alternatives (``--filter tags=bm25 --filter
    tags=rrf`` means "either tag"), which is why this returns lists rather than
    scalars. A repeatable flag is deliberately preferred over a JSON string
    argument: JSON on a PowerShell command line needs escaping that no runbook
    reader gets right the first time.

    Validation of the field names themselves is *not* done here — that is
    :meth:`FilterPolicy.build`'s job, and duplicating the allowlist in an argument
    parser is how the CLI and the API end up disagreeing about what is allowed.

    Raises:
        FilterError: An argument is not in ``field=value`` form.
    """
    filters: dict[str, list[str]] = {}
    for pair in pairs or ():
        name, separator, value = pair.partition("=")
        if not separator or not name.strip():
            raise FilterError(
                f"--filter expects field=value, got {pair!r}",
                error_type=FILTER_INVALID_VALUE,
            )
        filters.setdefault(name.strip(), []).append(value.strip())
    return filters


__all__ = [
    "FILTER_INVALID_VALUE",
    "FILTER_NOT_ALLOWED",
    "NO_FILTER_SUMMARY",
    "PUBLIC_TO_PAYLOAD",
    "FilterCondition",
    "FilterError",
    "FilterPolicy",
    "QueryFilter",
    "parse_filter_arguments",
]
