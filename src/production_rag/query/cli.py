"""Command-line adapter over the public query pipeline.

Logs go to stderr and the last stdout line is always one JSON object. The CLI
uses the same executor as the HTTP route, so there is one generation path and
one response projection.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any
from uuid import uuid4

import structlog
from pydantic import ValidationError

from production_rag.api.routes.query import QueryPipelineUnavailableError, execute_query
from production_rag.api.schemas import QueryRequest
from production_rag.config import get_settings
from production_rag.ingest.cli import (
    EXIT_OK,
    EXIT_RUNTIME,
    EXIT_USAGE,
    configure_cli_logging,
)
from production_rag.retrieval.embeddings import EMBEDDER_KINDS, FAKE_EMBEDDER
from production_rag.retrieval.filters import FilterError, parse_filter_arguments
from production_rag.retrieval.hybrid import RETRIEVAL_MODES
from production_rag.retrieval.rerank import RERANK_KINDS

_log = structlog.get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Define the query CLI without executing the pipeline."""
    parser = argparse.ArgumentParser(
        prog="python -m production_rag.query",
        description="Answer one question from indexed evidence with resolved citations.",
        epilog=(
            "The last line of stdout is one JSON object. Exit codes: 0 ok, "
            "1 runtime failure, 2 invalid invocation."
        ),
    )
    parser.add_argument("--question", required=True, help="Question to answer from the corpus.")
    parser.add_argument(
        "--mode",
        choices=RETRIEVAL_MODES,
        help="Retrieval mode override. Omit to use the configured default.",
    )
    parser.add_argument(
        "--rerank",
        choices=RERANK_KINDS,
        help="Reranker override. Omit to use the query pipeline default.",
    )
    parser.add_argument(
        "--llm",
        choices=("fake", "openai"),
        default="fake",
        help="Generation provider. 'fake' is deterministic offline plumbing only.",
    )
    parser.add_argument(
        "--embedder",
        choices=EMBEDDER_KINDS,
        default=FAKE_EMBEDDER,
        help=(
            "Query embedder. Default 'fake' is offline but must match the embedder "
            "that built the collection."
        ),
    )
    parser.add_argument(
        "--filter",
        dest="filters",
        action="append",
        metavar="FIELD=VALUE",
        help=(
            "Metadata filter, repeatable. Only fields in retrieval.filters.allowed_fields "
            "are accepted. Repeating a field means 'any of'; different fields are AND."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Include safe pipeline diagnostics in the JSON response.",
    )
    parser.add_argument(
        "--log-level",
        help="Log level for this run. Defaults to LOG_LEVEL.",
    )
    return parser


def _emit(payload: dict[str, Any]) -> None:
    """Write a single machine-readable result line to stdout."""
    json.dump(payload, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    sys.stdout.flush()


def _fail(message: str, kind: str, code: int) -> int:
    """Emit a safe error payload and return its process status."""
    _log.error("query_failed", error=message, error_type=kind)
    _emit({"ok": False, "error": message, "error_type": kind})
    return code


def main(argv: Sequence[str] | None = None) -> int:
    """Run one grounded query and return a graded process exit code."""
    args = build_parser().parse_args(argv)
    settings = get_settings()
    configure_cli_logging(args.log_level or settings.log_level)

    try:
        # Widened on purpose: the request schema also accepts a bare string per
        # field, and `dict[str, list[str]]` is not a subtype of that dict.
        filters: dict[str, str | list[str]] = dict(parse_filter_arguments(args.filters))
        payload = QueryRequest(
            question=args.question,
            mode=args.mode,
            rerank=args.rerank,
            llm=args.llm,
            filters=filters or None,
            debug=args.debug,
        )
    except FilterError as exc:
        return _fail(str(exc), exc.error_type, EXIT_USAGE)
    except ValidationError as exc:
        return _fail("invalid query arguments", type(exc).__name__, EXIT_USAGE)

    try:
        result = execute_query(
            payload,
            settings=settings,
            request_id=str(uuid4()),
            embedder_kind=args.embedder,
        )
    except FilterError as exc:
        # A field outside the allowlist. Same slug as the HTTP 422 body, and the
        # same graded meaning as any other bad invocation: nothing was retrieved.
        return _fail(str(exc), exc.error_type, EXIT_USAGE)
    except QueryPipelineUnavailableError as exc:
        return _fail(str(exc), type(exc).__name__, EXIT_RUNTIME)
    except Exception as exc:  # noqa: BLE001 - CLI must preserve the last-line JSON contract
        # Provider and storage exceptions are owned by the pipeline. Do not echo
        # their messages here: an upstream SDK may include sensitive request
        # context. The exception type is sufficient for automation and logs.
        return _fail("query failed", type(exc).__name__, EXIT_RUNTIME)

    # exclude_none so optional debug fields (e.g. cache) stay off the wire when
    # the cache did not run — same allowlist shape as the HTTP response.
    response_payload = result.model_dump(mode="json", exclude_none=True)
    if result.debug is None:
        response_payload.pop("debug", None)
    _emit({"ok": True, **response_payload})
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover - exercised via __main__.py
    raise SystemExit(main())
