"""Retrieval ablations over the canonical source hit@k scorer.

The harness holds the corpus, golden cases, embedder and cutoff constant while it
changes one retrieval stage at a time:

* dense only
* sparse only
* dense + sparse fused with RRF
* the same hybrid candidates reordered by the deterministic fake reranker

The fake embedder and fake reranker make this command an offline plumbing
experiment, not a semantic-quality claim.  Golden parsing and hit attribution
remain owned by :mod:`production_rag.evals.source_hit`; this module only
orchestrates repeated runs and computes the one requested delta.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import structlog

from production_rag.config import get_settings
from production_rag.config_loader import ConfigFileError, load_yaml_config
from production_rag.evals.source_hit import (
    DEFAULT_GOLDEN_PATH,
    DEFAULT_K,
    EvalError,
    GoldenCase,
    SourceHitReport,
    evaluate_source_hit,
    load_golden,
)
from production_rag.ingest.cli import (
    EXIT_OK,
    EXIT_RUNTIME,
    EXIT_USAGE,
    configure_cli_logging,
    resolve_embedder,
)
from production_rag.retrieval.cli import resolve_searchable_store
from production_rag.retrieval.embeddings import EmbeddingError
from production_rag.retrieval.hybrid import (
    MODE_DENSE,
    MODE_HYBRID,
    MODE_SPARSE,
    RetrievalError,
    Retriever,
)
from production_rag.retrieval.rerank import FakeReranker, RerankError
from production_rag.retrieval.sparse import SparseError
from production_rag.retrieval.store import CollectionMismatchError, VectorStoreError

MODE_HYBRID_RERANK = "hybrid+rerank(fake)"
ABLATION_MODES = (MODE_DENSE, MODE_SPARSE, MODE_HYBRID, MODE_HYBRID_RERANK)

_log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class AblationReport:
    """The four comparable source hit@k measurements."""

    k: int
    golden_path: str
    reports: dict[str, SourceHitReport]

    @property
    def delta_hybrid_vs_hybrid_rerank(self) -> float:
        """Absolute score change introduced by fake reranking."""
        return self.reports[MODE_HYBRID_RERANK].score - self.reports[MODE_HYBRID].score

    def to_summary(self) -> dict[str, Any]:
        """Return the stable JSON payload printed by the CLI."""
        modes = {
            mode: {
                "hit_at_k": round(self.reports[mode].score, 4),
                "hits": self.reports[mode].hits,
                "scored_cases": len(self.reports[mode].scored_cases),
                "unscored_cases": self.reports[mode].unscored,
            }
            for mode in ABLATION_MODES
        }
        hybrid = self.reports[MODE_HYBRID]
        return {
            "ok": True,
            "metric": "source_hit_at_k",
            "k": self.k,
            "collection": hybrid.collection,
            "embedded_model": hybrid.embedded_model,
            "golden_path": self.golden_path,
            "modes": modes,
            "delta_hybrid_vs_hybrid_rerank": round(self.delta_hybrid_vs_hybrid_rerank, 4),
        }


def evaluate_ablation(
    *,
    retriever: Retriever,
    reranked_retriever: Retriever,
    cases: Sequence[GoldenCase],
    k: int = DEFAULT_K,
    golden_path: str = "",
) -> AblationReport:
    """Run all modes through :func:`evaluate_source_hit` without forking scoring."""
    reports = {
        mode: evaluate_source_hit(retriever=retriever, cases=cases, k=k, mode=mode)
        for mode in (MODE_DENSE, MODE_SPARSE, MODE_HYBRID)
    }
    reports[MODE_HYBRID_RERANK] = evaluate_source_hit(
        retriever=reranked_retriever,
        cases=cases,
        k=k,
        mode=MODE_HYBRID,
    )
    return AblationReport(k=k, golden_path=golden_path, reports=reports)


def build_parser() -> argparse.ArgumentParser:
    """Define the offline-friendly CLI surface."""
    parser = argparse.ArgumentParser(
        prog="python -m production_rag.evals.ablation",
        description=("Compare dense, sparse, hybrid and hybrid+fake-rerank with source hit@k."),
        epilog=(
            "The last stdout line is one JSON report. The fake path needs no key "
            "or model download and is a plumbing experiment, not a quality claim."
        ),
    )
    parser.add_argument(
        "--golden",
        default=str(DEFAULT_GOLDEN_PATH),
        help=f"JSONL golden set. Default {DEFAULT_GOLDEN_PATH}.",
    )
    parser.add_argument("--k", type=int, default=DEFAULT_K, help=f"Cutoff. Default {DEFAULT_K}.")
    parser.add_argument(
        "--embedder",
        choices=("fake", "openai"),
        default="fake",
        help="Embedding provider. Default 'fake' is offline and carries no semantic signal.",
    )
    parser.add_argument(
        "--collection",
        help="Collection to query. Defaults to QDRANT_COLLECTION, else qdrant.collection.",
    )
    parser.add_argument(
        "--config",
        help="YAML profile to load. Defaults to CONFIG_PATH, else configs/default.yaml.",
    )
    parser.add_argument("--qdrant-url", help="Qdrant base URL. Defaults to QDRANT_URL.")
    parser.add_argument("--log-level", help="Log level. Defaults to LOG_LEVEL.")
    return parser


def _emit(payload: dict[str, Any]) -> None:
    """Write one JSON object as the final stdout line."""
    json.dump(payload, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    sys.stdout.flush()


def _fail(message: str, kind: str, code: int) -> int:
    """Emit the machine-readable failure contract and return its exit code."""
    _log.error("ablation_failed", error=message, error_type=kind)
    _emit({"ok": False, "error": message, "error_type": kind})
    return code


def main(argv: Sequence[str] | None = None) -> int:
    """Run the four-way retrieval ablation."""
    args = build_parser().parse_args(argv)
    settings = get_settings()
    configure_cli_logging(args.log_level or settings.log_level)

    try:
        config = load_yaml_config(args.config or settings.config_path)
        cases = load_golden(args.golden)
    except (ConfigFileError, EvalError) as exc:
        return _fail(str(exc), type(exc).__name__, EXIT_USAGE)

    collection = args.collection or (
        settings.qdrant_collection
        if "qdrant_collection" in settings.model_fields_set
        else config.qdrant.collection
    )
    try:
        embedder = resolve_embedder(args.embedder, config=config, settings=settings)
    except EmbeddingError as exc:
        return _fail(str(exc), type(exc).__name__, EXIT_USAGE)

    store = resolve_searchable_store(
        config=config,
        settings=settings,
        collection=collection,
        url=args.qdrant_url or settings.qdrant_url,
    )
    try:
        retriever = Retriever.from_config(store=store, embedder=embedder, config=config)
        reranked_retriever = Retriever.from_config(
            store=store,
            embedder=embedder,
            config=config,
            reranker=FakeReranker(),
        )
        report = evaluate_ablation(
            retriever=retriever,
            reranked_retriever=reranked_retriever,
            cases=cases,
            k=args.k,
            golden_path=args.golden,
        )
    except (EvalError, RetrievalError, RerankError, SparseError, CollectionMismatchError) as exc:
        return _fail(str(exc), type(exc).__name__, EXIT_USAGE)
    except (EmbeddingError, VectorStoreError) as exc:
        return _fail(str(exc), type(exc).__name__, EXIT_RUNTIME)

    _emit(report.to_summary())
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
