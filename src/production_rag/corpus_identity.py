"""Collection / corpus identity for readiness, cache keys, and mismatch checks.

Identity is the set of facts that make two indexes non-interchangeable:

* embedder id — dense space
* chunker version — how text was split
* document count — scale of the source walk
* corpus hash — content-addressed digest of source bytes under the corpus root

Without these, a cache hit or a query against the wrong collection can look
plausible and still be wrong. The free path keeps Qdrant local; identity is the
guardrail that replaces multi-tenant hosted collection URLs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from production_rag.config_loader import ChunkingConfig, load_yaml_config
from production_rag.ingest.hashing import normalise_source_path, sha256_text
from production_rag.ingest.loaders import iter_documents

# Bump when chunk_document semantics change in a way that invalidates labels.
CHUNKER_VERSION = "structural-markdown-v1"


@dataclass(frozen=True, slots=True)
class CorpusIdentity:
    """Snapshot of what an index was built from."""

    embedder_id: str
    chunker_version: str
    doc_count: int
    corpus_hash: str
    corpus_root: str = ""
    collection: str = ""

    def as_public_dict(self) -> dict[str, Any]:
        """JSON-ready mapping for /ready and debug surfaces."""
        return {
            "embedder_id": self.embedder_id,
            "chunker_version": self.chunker_version,
            "doc_count": self.doc_count,
            "corpus_hash": self.corpus_hash,
            "corpus_root": self.corpus_root,
            "collection": self.collection,
        }

    def cache_material(self) -> str:
        """Stable string included in query-cache keys."""
        return json.dumps(
            {
                "collection": self.collection,
                "embedder_id": self.embedder_id,
                "chunker_version": self.chunker_version,
                "doc_count": self.doc_count,
                "corpus_hash": self.corpus_hash,
            },
            separators=(",", ":"),
            sort_keys=True,
        )


def chunker_version_for(config: ChunkingConfig | None = None) -> str:
    """Return the chunker version id, including size/overlap from config."""
    if config is None:
        return CHUNKER_VERSION
    return (
        f"{CHUNKER_VERSION}:size={config.chunk_size}:overlap={config.chunk_overlap}"
    )


def hash_corpus_root(
    root: Path,
    *,
    include_extensions: Iterable[str] | None = None,
    exclude_globs: Iterable[str] | None = None,
) -> tuple[str, int]:
    """Content-address corpus files; return (hex digest, document count)."""
    config = load_yaml_config()
    extensions = (
        tuple(include_extensions)
        if include_extensions is not None
        else config.ingest.include_extensions
    )
    excludes = (
        tuple(exclude_globs)
        if exclude_globs is not None
        else config.ingest.exclude_globs
    )
    documents = sorted(
        iter_documents(root, include_extensions=extensions, exclude_globs=excludes),
        key=lambda document: normalise_source_path(document.source_path),
    )
    digest = hashlib.sha256()
    for document in documents:
        path = normalise_source_path(document.source_path)
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_text(document.text).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest(), len(documents)


def build_corpus_identity(
    *,
    corpus_root: Path | str,
    embedder_id: str,
    collection: str = "",
    chunking: ChunkingConfig | None = None,
) -> CorpusIdentity:
    """Compute identity for a corpus root and embedder."""
    root = Path(corpus_root)
    if chunking is None:
        chunking = load_yaml_config().ingest.chunking
    corpus_hash, doc_count = hash_corpus_root(root)
    return CorpusIdentity(
        embedder_id=embedder_id,
        chunker_version=chunker_version_for(chunking),
        doc_count=doc_count,
        corpus_hash=corpus_hash,
        corpus_root=normalise_source_path(str(root)),
        collection=collection,
    )


def identities_compatible(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """True when two public identity dicts describe the same index material."""
    keys = ("embedder_id", "chunker_version", "doc_count", "corpus_hash")
    return all(left.get(key) == right.get(key) for key in keys)


def load_identity_sidecar(path: Path) -> dict[str, Any] | None:
    """Load a previously written identity JSON, or None if missing."""
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def write_identity_sidecar(path: Path, identity: CorpusIdentity) -> None:
    """Persist identity next to an ingest run for /ready and cache keys."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(identity.as_public_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def default_identity_path(collection: str) -> Path:
    """Sidecar path under data/eval/reports for local free-path demos."""
    safe = collection.replace("/", "_").replace("\\", "_")
    return Path("data") / "eval" / "reports" / f"collection-identity-{safe}.json"


class WrongCollectionError(LookupError):
    """Caller asked for a collection that does not match the live identity."""

    error_type = "wrong_collection"

    def __init__(self, message: str, *, collection: str) -> None:
        super().__init__(message)
        self.collection = collection


__all__ = [
    "CHUNKER_VERSION",
    "CorpusIdentity",
    "WrongCollectionError",
    "build_corpus_identity",
    "chunker_version_for",
    "default_identity_path",
    "hash_corpus_root",
    "identities_compatible",
    "load_identity_sidecar",
    "write_identity_sidecar",
]
