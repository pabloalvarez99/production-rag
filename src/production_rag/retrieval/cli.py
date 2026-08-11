"""Command-line entrypoint for retrieval.

    python -m production_rag.retrieve --query "how are sparse vectors stored?"

Same two conventions as the ingest CLI, for the same reason — these commands are
meant to be called by scripts, not only read by humans:

* **Logs go to stderr, the result to stdout.** The last line of stdout is always a
  single JSON object, so ``$(... | tail -1)`` is a contract, not a guess.
* **Exit codes are graded.** ``0`` success, ``2`` the invocation or the
  configuration is wrong (nothing was attempted), ``1`` the run started and
  failed. A wrapper can retry ``1`` and must never retry ``2``.

**Where this lives.** The implementation is here, next to the retriever it drives;
``production_rag.retrieve`` is a thin package whose ``__main__`` delegates to
:func:`main`. One implementation, and the invocation a reader would guess
(``python -m production_rag.retrieve``, mirroring ``python -m
production_rag.ingest``) is the one that works.

``--embedder`` defaults to ``fake``, which is honest rather than convenient: fake
vectors carry no semantics, so dense results will look arbitrary while BM25 still
works. It keeps the command runnable with no API key; ``--embedder openai`` is
what a real measurement uses, and it must be the same model that built the index.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from typing import Any

import structlog

from production_rag.config import Settings, get_settings
from production_rag.config_loader import ConfigFileError, YamlConfig, load_yaml_config
from production_rag.ingest.cli import (
    EXIT_OK,
    EXIT_RUNTIME,
    EXIT_USAGE,
    configure_cli_logging,
    resolve_embedder,
)
from production_rag.retrieval.embeddings import (
    EMBEDDER_KINDS,
    FAKE_EMBEDDER,
    EmbeddingError,
)
from production_rag.retrieval.hybrid import (
    RETRIEVAL_MODES,
    RetrievalError,
    Retriever,
)
from production_rag.retrieval.rerank import (
    RERANK_KINDS,
    RERANK_OFF,
    RerankError,
    build_reranker,
)
from production_rag.retrieval.sparse import SparseError
from production_rag.retrieval.store import (
    CollectionMismatchError,
    QdrantStore,
    SearchableVectorStore,
    VectorStoreError,
)

_log = structlog.get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Define the CLI surface.

    Kept in its own function so ``--help`` is testable without running anything.
    """
    parser = argparse.ArgumentParser(
        prog="python -m production_rag.retrieve",
        description=(
            "Retrieve chunks for a query using dense, sparse or hybrid search, "
            "fused by reciprocal rank fusion."
        ),
        epilog=(
            "The last line of stdout is a JSON object with the hits and the "
            "parameters that produced them. Exit codes: 0 ok, 1 the run failed, "
            "2 the invocation or configuration is wrong."
        ),
    )
    parser.add_argument(
        "--query",
        required=True,
        help=(
            "The question to retrieve for. A query starting with '-' needs the "
            "equals form, --query=--recreate-collection, or argparse reads it as a flag."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=RETRIEVAL_MODES,
        help=(
            "Retrieval mode. Default from retrieval.mode ('hybrid'). "
            "'dense' and 'sparse' exist so a single-branch baseline is measurable."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        help="Hits to return after fusion. Defaults to retrieval.top_k.",
    )
    parser.add_argument(
        "--embedder",
        choices=EMBEDDER_KINDS,
        default=FAKE_EMBEDDER,
        help=(
            # ASCII only: --help is written to a console that may be cp1252, where
            # a stray em dash raises UnicodeEncodeError instead of printing help.
            "Embedding provider. Default 'fake': deterministic hash vectors, no API key, "
            "no network - dense results are arbitrary, BM25 still works. Must match "
            "the model that built the index."
        ),
    )
    parser.add_argument(
        "--rerank",
        choices=RERANK_KINDS,
        default=RERANK_OFF,
        help=(
            "Second-stage reranker. Default 'off', which is the fusion-only pipeline. "
            "'fake' is a deterministic offline term-overlap double, useful for wiring "
            "and demos, not a quality claim. 'local' runs a cross-encoder through "
            "sentence-transformers. 'cohere' needs COHERE_API_KEY in the environment. "
            "'auto' follows rerank.enabled and rerank.provider in the YAML profile."
        ),
    )
    parser.add_argument(
        "--collection",
        help="Collection to query. Defaults to QDRANT_COLLECTION, else qdrant.collection.",
    )
    parser.add_argument(
        "--config",
        help="YAML profile to load. Defaults to CONFIG_PATH, else configs/default.yaml.",
    )
    parser.add_argument(
        "--qdrant-url",
        help="Qdrant base URL. Defaults to QDRANT_URL.",
    )
    parser.add_argument(
        "--log-level",
        help="Log level for this run. Defaults to LOG_LEVEL.",
    )
    return parser


def resolve_searchable_store(
    *,
    config: YamlConfig,
    settings: Settings,
    collection: str,
    url: str,
) -> SearchableVectorStore:
    """Build the Qdrant store for querying. No connection is opened here.

    Separate from the ingest CLI's builder because the return type differs: this
    one promises search, and the write path must not be able to depend on that
    promise by accident.
    """
    qdrant = config.qdrant
    return QdrantStore(
        url=url,
        collection=collection,
        api_key=settings.qdrant_api_key or os.environ.get(qdrant.api_key_env),
        timeout_seconds=qdrant.timeout_seconds,
        hnsw=qdrant.hnsw,
        on_disk_payload=qdrant.on_disk_payload,
        wait_for_writes=qdrant.wait_for_writes,
        search_hnsw_ef=qdrant.search.hnsw_ef,
    )


def _emit(payload: dict[str, Any]) -> None:
    """Write one JSON object as the final line of stdout."""
    json.dump(payload, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    sys.stdout.flush()


def _fail(message: str, kind: str, code: int) -> int:
    """Report a failure on both channels and return *code*."""
    _log.error("retrieve_failed", error=message, error_type=kind)
    _emit({"ok": False, "error": message, "error_type": kind})
    return code


def main(argv: Sequence[str] | None = None) -> int:
    """Run one retrieval and return the process exit code."""
    args = build_parser().parse_args(argv)
    settings = get_settings()
    configure_cli_logging(args.log_level or settings.log_level)

    try:
        config = load_yaml_config(args.config or settings.config_path)
    except ConfigFileError as exc:
        return _fail(str(exc), type(exc).__name__, EXIT_USAGE)

    collection = args.collection or (
        settings.qdrant_collection
        if "qdrant_collection" in settings.model_fields_set
        else config.qdrant.collection
    )
    url = args.qdrant_url or settings.qdrant_url

    try:
        embedder = resolve_embedder(args.embedder, config=config, settings=settings)
    except EmbeddingError as exc:
        # Missing credential or unknown provider: nothing was attempted.
        return _fail(str(exc), type(exc).__name__, EXIT_USAGE)

    try:
        # The credential is read from the environment, never from a flag: a key on
        # a command line ends up in shell history, in `ps` output and in
        # `docker inspect`.
        reranker = build_reranker(
            args.rerank,
            config=config.rerank,
            api_key=os.environ.get(config.rerank.api_key_env),
        )
    except RerankError as exc:
        # Unknown provider or a missing key: nothing was attempted.
        return _fail(str(exc), type(exc).__name__, EXIT_USAGE)

    store = resolve_searchable_store(
        config=config, settings=settings, collection=collection, url=url
    )

    try:
        retriever = Retriever.from_config(
            store=store, embedder=embedder, config=config, reranker=reranker
        )
        result = retriever.retrieve(args.query, mode=args.mode, top_k=args.top_k)
    except RerankError as exc:
        # Only reachable with rerank.fail_open false; the run started, and the
        # failure may be transient.
        return _fail(str(exc), type(exc).__name__, EXIT_RUNTIME)
    except (RetrievalError, SparseError, CollectionMismatchError) as exc:
        # Bad query, bad mode, or a collection that cannot answer this shape of
        # search. Retrying changes nothing.
        return _fail(str(exc), type(exc).__name__, EXIT_USAGE)
    except (EmbeddingError, VectorStoreError) as exc:
        # The run started; the failure is environmental and may be retryable.
        return _fail(str(exc), type(exc).__name__, EXIT_RUNTIME)

    _emit(result.to_summary())
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover - exercised via __main__.py
    raise SystemExit(main())
