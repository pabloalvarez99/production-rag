"""Command-line entrypoint for the ingest job.

    python -m production_rag.ingest --source data/raw/sample --embedder fake

Two conventions make this usable from a script rather than only from a terminal:

* **Logs go to stderr, the summary to stdout.** The last line of stdout is always
  a single JSON object, so ``$(... | tail -1)`` is a contract, not a guess.
* **Exit codes are graded.** ``0`` success, ``2`` the invocation or the
  configuration is wrong (nothing was attempted), ``1`` the run started and
  failed. A wrapper can retry ``1`` and must never retry ``2``.

Credentials are read from the environment only. There is no ``--api-key`` flag,
because a key passed on a command line ends up in shell history, in ``ps`` output
and in ``docker inspect``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Sequence
from typing import Any

import structlog

from production_rag.config import Settings, get_settings
from production_rag.config_loader import ConfigFileError, YamlConfig, load_yaml_config
from production_rag.ingest.loaders import CorpusError
from production_rag.ingest.pipeline import IngestError, run_ingest
from production_rag.retrieval.embeddings import (
    EMBEDDER_KINDS,
    FAKE_EMBEDDER,
    EmbeddingError,
    EmbeddingProvider,
    build_embedder,
)
from production_rag.retrieval.store import (
    CollectionMismatchError,
    QdrantStore,
    VectorStore,
    VectorStoreError,
)

EXIT_OK = 0
EXIT_RUNTIME = 1
EXIT_USAGE = 2

_log = structlog.get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Define the CLI surface.

    Kept in its own function so ``--help`` is testable without running anything.
    """
    parser = argparse.ArgumentParser(
        prog="python -m production_rag.ingest",
        description=(
            "Walk a corpus, chunk it, embed the chunks and upsert them into Qdrant. "
            "Idempotent: re-running over unchanged files rewrites the same points."
        ),
        epilog=(
            "The last line of stdout is a JSON summary. Exit codes: 0 ok, "
            "1 the run failed, 2 the invocation or configuration is wrong."
        ),
    )
    parser.add_argument(
        "--source",
        help="Corpus root to walk. Defaults to ingest.source_dir from the config file.",
    )
    parser.add_argument(
        "--config",
        help="YAML profile to load. Defaults to CONFIG_PATH, else configs/default.yaml.",
    )
    parser.add_argument(
        "--embedder",
        choices=EMBEDDER_KINDS,
        default=FAKE_EMBEDDER,
        help=(
            # ASCII only: --help is written to a console that may be cp1252, where
            # a stray em dash raises UnicodeEncodeError instead of printing help.
            "Embedding provider. Default 'fake': deterministic hash vectors, no API key, "
            "no network, no spend - useful for plumbing, worthless for retrieval quality."
        ),
    )
    parser.add_argument(
        "--collection",
        help="Target collection. Defaults to QDRANT_COLLECTION, else qdrant.collection.",
    )
    parser.add_argument(
        "--qdrant-url",
        help="Qdrant base URL. Defaults to QDRANT_URL.",
    )
    parser.add_argument(
        # Both spellings are accepted: --recreate is what scripts/ingest.ps1 and
        # the Makefile pass, --recreate-collection is what the docs call it.
        "--recreate",
        "--recreate-collection",
        dest="recreate",
        action="store_true",
        help="Delete the collection before writing. DESTRUCTIVE; required to change vector size.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Walk and chunk only. No embedding call, no upsert, no connection to Qdrant.",
    )
    parser.add_argument(
        "--no-incremental",
        dest="incremental",
        action="store_false",
        default=None,
        help="Re-embed every chunk even if its point id is already stored.",
    )
    parser.add_argument(
        "--log-level",
        help="Log level for this run. Defaults to LOG_LEVEL.",
    )
    return parser


def _stderr_logger(*args: str) -> structlog.WriteLogger:
    """Build a logger bound to whatever ``sys.stderr`` is *right now*.

    Not ``WriteLoggerFactory(file=sys.stderr)``: that captures the stream at
    configuration time, and a caller that later replaces or closes ``sys.stderr``
    — pytest's capture does exactly this — would make every subsequent log call
    raise on a closed file.
    """
    del args
    return structlog.WriteLogger(sys.stderr)


def configure_cli_logging(level: str) -> None:
    """Send structlog output to **stderr** at *level*.

    Deliberately not reusing the API's logging setup: that one writes to stdout,
    which would interleave log lines with the JSON summary this command's callers
    parse.
    """
    numeric = logging.getLevelNamesMapping().get(level.upper(), logging.INFO)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric),
        logger_factory=_stderr_logger,
        cache_logger_on_first_use=False,
    )


def _from_env_or_yaml(settings: Settings, field: str, yaml_value: str) -> str:
    """Resolve one setting: an explicit environment value wins over the YAML file.

    ``model_fields_set`` is what makes this precise — it holds only the fields
    actually supplied by the environment or ``.env``, so a field left at its
    default never silently overrides the config file.
    """
    if field in settings.model_fields_set:
        value = getattr(settings, field)
        if isinstance(value, str) and value:
            return value
    return yaml_value


def resolve_embedder(kind: str, *, config: YamlConfig, settings: Settings) -> EmbeddingProvider:
    """Build the embedding provider, reading any credential from the environment.

    Raises:
        EmbeddingError: The provider needs a key and none is set.
    """
    embedding = config.ingest.embedding
    api_key = settings.openai_api_key or os.environ.get(embedding.api_key_env)
    return build_embedder(kind, config=embedding, api_key=api_key)


def resolve_store(
    *,
    config: YamlConfig,
    settings: Settings,
    collection: str,
    url: str,
) -> VectorStore:
    """Build the Qdrant store. No connection is opened here."""
    qdrant = config.qdrant
    return QdrantStore(
        url=url,
        collection=collection,
        api_key=settings.qdrant_api_key or os.environ.get(qdrant.api_key_env),
        timeout_seconds=qdrant.timeout_seconds,
        hnsw=qdrant.hnsw,
        payload_indexes=qdrant.payload_indexes,
        on_disk_payload=qdrant.on_disk_payload,
        wait_for_writes=qdrant.wait_for_writes,
        upsert_batch_size=config.ingest.embedding.batch_size,
    )


def _emit(payload: dict[str, Any]) -> None:
    """Write one JSON object as the final line of stdout."""
    json.dump(payload, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    sys.stdout.flush()


def _fail(message: str, kind: str, code: int) -> int:
    """Report a failure on both channels and return *code*."""
    _log.error("ingest_failed", error=message, error_type=kind)
    _emit({"ok": False, "error": message, "error_type": kind})
    return code


def main(argv: Sequence[str] | None = None) -> int:
    """Run one ingest and return the process exit code."""
    args = build_parser().parse_args(argv)
    settings = get_settings()
    configure_cli_logging(args.log_level or settings.log_level)

    try:
        config = load_yaml_config(args.config or settings.config_path)
    except ConfigFileError as exc:
        return _fail(str(exc), type(exc).__name__, EXIT_USAGE)

    source = args.source or config.ingest.source_dir
    collection = args.collection or _from_env_or_yaml(
        settings, "qdrant_collection", config.qdrant.collection
    )
    url = args.qdrant_url or settings.qdrant_url

    try:
        embedder = resolve_embedder(args.embedder, config=config, settings=settings)
    except EmbeddingError as exc:
        # Missing credential or unknown provider: nothing was attempted.
        return _fail(str(exc), type(exc).__name__, EXIT_USAGE)

    store = (
        None
        if args.dry_run
        else resolve_store(config=config, settings=settings, collection=collection, url=url)
    )

    try:
        result = run_ingest(
            source_dir=source,
            config=config,
            embedder=embedder,
            store=store,
            collection=collection,
            recreate=args.recreate,
            incremental=args.incremental,
            dry_run=args.dry_run,
        )
    except (CorpusError, IngestError, CollectionMismatchError) as exc:
        # A vector-size mismatch is a configuration decision, not a transient
        # fault: retrying cannot fix it, so it exits 2 like a bad invocation.
        return _fail(str(exc), type(exc).__name__, EXIT_USAGE)
    except (EmbeddingError, VectorStoreError) as exc:
        # The run started; the failure is environmental and may be retryable.
        return _fail(str(exc), type(exc).__name__, EXIT_RUNTIME)

    _emit(result.to_summary())
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover - exercised via __main__.py
    raise SystemExit(main())
