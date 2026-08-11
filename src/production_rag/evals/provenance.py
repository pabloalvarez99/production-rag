"""Fail-closed provenance checks for evaluation collections."""

from __future__ import annotations

from collections.abc import Mapping

from production_rag.retrieval.store import CollectionMismatchError, SearchableVectorStore


def collection_embedder_models(store: SearchableVectorStore) -> set[str]:
    """Read the embedder identities stamped on collection points."""
    client = getattr(store, "client", None)
    if client is not None:
        models: set[str] = set()
        offset: object | None = None
        while True:
            records, next_offset = client.scroll(
                collection_name=store.collection,
                limit=256,
                offset=offset,
                with_payload=["embedded_model"],
                with_vectors=False,
            )
            models.update(
                str(record.payload.get("embedded_model"))
                for record in records
                if record.payload and record.payload.get("embedded_model")
            )
            if next_offset is None:
                return models
            offset = next_offset
    payloads = getattr(store, "points", None)
    if payloads is None:
        payloads = getattr(store, "_payloads", None)
    if isinstance(payloads, Mapping):
        return {
            str(payload.get("embedded_model"))
            for payload in payloads.values()
            if isinstance(payload, Mapping) and payload.get("embedded_model")
        }
    raise CollectionMismatchError(
        f"collection {store.collection!r} cannot expose embedded_model provenance"
    )


def assert_collection_embedder(store: SearchableVectorStore, *, expected_model: str) -> None:
    """Refuse retrieval unless stored vectors and query vectors share an embedding space."""
    models = collection_embedder_models(store)
    if not models:
        raise CollectionMismatchError(
            f"collection {store.collection!r} has no embedded_model provenance; rebuild it"
        )
    if models != {expected_model}:
        raise CollectionMismatchError(
            f"collection {store.collection!r} was embedded with {sorted(models)!r}, "
            f"but queries use {expected_model!r}; rebuild or select the matching embedder"
        )


__all__ = ["assert_collection_embedder", "collection_embedder_models"]
