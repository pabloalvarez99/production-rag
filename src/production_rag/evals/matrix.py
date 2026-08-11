"""Four-configuration, paired, resumable evaluation matrix.

Run with ``--collection prag_matrix``. The checkpoint retains every per-item
binary hit outcome and is updated after each configuration.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from production_rag.config import get_settings
from production_rag.config_loader import load_yaml_config
from production_rag.evals.costs import (
    MODEL_RATES_USD_PER_MILLION,
    estimate_cost,
    require_spending_consent,
)
from production_rag.evals.judges import build_judge
from production_rag.evals.scorecard import CONFIG_NAMES, validate_scorecard
from production_rag.evals.source_hit import GoldenCase, load_golden
from production_rag.evals.stats import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    exact_mcnemar,
    minimum_detectable_effect,
    paired_delta,
    paired_percentile_bootstrap,
    reportability,
)
from production_rag.evals.tier1_retrieval import evaluate_tier1
from production_rag.evals.tier2_answer import evaluate_tier2
from production_rag.generation.llm import build_llm
from production_rag.ingest.cli import configure_cli_logging, resolve_embedder, resolve_store
from production_rag.ingest.pipeline import run_ingest
from production_rag.retrieval.cli import resolve_searchable_store
from production_rag.retrieval.hybrid import Retriever
from production_rag.retrieval.rerank import build_reranker

CONFIGURATIONS: Mapping[str, tuple[str, str]] = {
    "sparse": ("sparse", "off"),
    "dense": ("dense", "off"),
    "hybrid": ("hybrid", "off"),
    "hybrid_rerank": ("hybrid", "fake"),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--golden", default="data/eval/golden-corpus.jsonl")
    parser.add_argument("--corpus", default="data/corpus")
    parser.add_argument("--config")
    parser.add_argument("--qdrant-url")
    parser.add_argument("--embedder", choices=("fake", "openai"), default="fake")
    parser.add_argument("--llm", choices=("fake", "openai"), default="fake")
    parser.add_argument("--judge", choices=("fake", "openai"), default="fake")
    parser.add_argument("--ingest", action="store_true", help="Ingest once before the sweep.")
    parser.add_argument("--yes-spend", action="store_true")
    parser.add_argument("--checkpoint", default="data/eval/reports/matrix-checkpoint.json")
    parser.add_argument("--report", default="data/eval/reports/scorecard.json")
    parser.add_argument("--log-level")
    return parser


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _dry_counts(corpus: str, config: Any, embedder: Any, collection: str) -> tuple[int, int]:
    result = run_ingest(
        source_dir=corpus,
        config=config,
        embedder=embedder,
        collection=collection,
        dry_run=True,
    )
    return result.documents_scanned, result.chunks_created


def _print_estimate(estimate: Any) -> None:
    print("Cost estimate (USD; rate assumptions per 1M tokens):")
    for model, rate in MODEL_RATES_USD_PER_MILLION.items():
        print(f"  {model}: ${rate:.4f}")
    print(
        f"  chunks to embed={estimate.chunks_to_embed}; queries={estimate.queries}; "
        f"generations={estimate.generations}; judge calls={estimate.judge_calls}; "
        f"estimated total=${estimate.estimated_usd:.4f}"
    )


def _metrics(tier1: Any, tier2: Any) -> dict[str, float]:
    return {
        "hit_at_5": round(tier1.hit_at_k, 4),
        "recall_at_5": round(tier1.recall_at_k, 4),
        "mrr": round(tier1.mrr, 4),
        "ndcg_at_5": round(tier1.ndcg_at_k, 4),
        "citation_precision": round(tier2.citation_precision, 4),
        "invalid_marker_rate": round(tier2.invalid_marker_rate, 4),
        "refusal_accuracy": round(tier2.refusal_accuracy, 4),
    }


def build_scorecard(
    *,
    checkpoint: Mapping[str, Any],
    cases: Sequence[GoldenCase],
    commit: str,
    documents: int,
    chunks: int,
    golden_path: str,
    corpus_path: str,
    cost_usd: float,
    billed: bool,
) -> dict[str, Any]:
    """Build schema v1 from retained per-item outcomes, never aggregates alone."""
    completed = checkpoint["configs"]
    configs = {name: completed[name]["metrics"] for name in CONFIG_NAMES}
    categories: dict[str, list[int]] = defaultdict(list)
    for index, case in enumerate(cases):
        categories[case.category or "uncategorized"].append(index)
    slices: dict[str, Any] = {}
    for category, indices in sorted(categories.items()):
        baseline = [bool(completed["sparse"]["hit_vector"][index]) for index in indices]
        base_rate = sum(baseline) / len(baseline)
        slices[category] = {
            "n": len(indices),
            "powered": len(indices) >= 30,
            "mde_at_80_power": minimum_detectable_effect(len(indices), base_rate),
            "configs": {
                name: {
                    "hit_at_5": round(
                        sum(bool(completed[name]["hit_vector"][index]) for index in indices)
                        / len(indices),
                        4,
                    )
                }
                for name in CONFIG_NAMES
            },
        }
    scopes = {"overall": list(range(len(cases))), **categories}
    comparisons: list[dict[str, Any]] = []
    for name in ("dense", "hybrid", "hybrid_rerank"):
        for scope, indices in scopes.items():
            a = [bool(completed[name]["hit_vector"][index]) for index in indices]
            b = [bool(completed["sparse"]["hit_vector"][index]) for index in indices]
            ci = paired_percentile_bootstrap(a, b)
            mcnemar = exact_mcnemar(a, b)
            decision = reportability(len(indices), ci)
            comparisons.append(
                {
                    "a": name,
                    "b": "sparse",
                    "metric": "hit_at_5",
                    "scope": scope,
                    "n": len(indices),
                    "delta": round(paired_delta(a, b), 4),
                    "ci95": [round(ci[0], 4), round(ci[1], 4)],
                    "mcnemar": {
                        "a_only": mcnemar.a_only,
                        "b_only": mcnemar.b_only,
                        "p_exact": round(mcnemar.p_exact, 6),
                    },
                    "reportable": decision.reportable,
                    "reason": decision.reason,
                }
            )
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "provenance": {
            "commit": commit,
            "corpus": {"path": corpus_path, "documents": documents, "chunks": chunks},
            "golden": {"path": golden_path, "items": len(cases)},
            "embedder": checkpoint["embedder"],
            "llm": checkpoint["llm"],
            "judge": checkpoint["judge"],
            "cost_usd": cost_usd,
            "billed": billed,
        },
        "configs": configs,
        "slices": slices,
        "comparisons": comparisons,
    }
    validate_scorecard(payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    """Run or resume the paired matrix and emit the scorecard contract."""
    args = _parser().parse_args(argv)
    settings = get_settings()
    configure_cli_logging(args.log_level or settings.log_level)
    config = load_yaml_config(args.config or settings.config_path)
    cases = load_golden(args.golden)
    # Chunking is provider-independent. Keep the preflight free and usable
    # before a hosted-provider credential is present.
    counting_embedder = resolve_embedder("fake", config=config, settings=settings)
    documents, chunks = _dry_counts(args.corpus, config, counting_embedder, args.collection)
    billed = any(provider != "fake" for provider in (args.embedder, args.llm, args.judge))
    estimate = estimate_cost(
        chunks_to_embed=chunks if args.ingest else 0,
        queries=len(cases) * len(CONFIG_NAMES) * 2,
        generations=len(cases) * len(CONFIG_NAMES),
        judge_calls=len(cases) * len(CONFIG_NAMES),
        billed=billed,
    )
    _print_estimate(estimate)
    require_spending_consent(billed=billed, yes_spend=args.yes_spend)
    embedder = resolve_embedder(args.embedder, config=config, settings=settings)
    url = args.qdrant_url or settings.qdrant_url
    if args.ingest:
        ingest_store = resolve_store(
            config=config, settings=settings, collection=args.collection, url=url
        )
        run_ingest(
            source_dir=args.corpus,
            config=config,
            embedder=embedder,
            store=ingest_store,
            collection=args.collection,
        )
    checkpoint_path = Path(args.checkpoint)
    checkpoint: dict[str, Any]
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("case_ids") != [case.id for case in cases]:
            raise ValueError("checkpoint item set/order differs from this golden set")
    else:
        checkpoint = {
            "case_ids": [case.id for case in cases],
            "bootstrap_seed": DEFAULT_BOOTSTRAP_SEED,
            "bootstrap_resamples": DEFAULT_BOOTSTRAP_RESAMPLES,
            "embedder": args.embedder,
            "llm": args.llm,
            "judge": args.judge,
            "configs": {},
        }
    store = resolve_searchable_store(
        config=config, settings=settings, collection=args.collection, url=url
    )
    llm = build_llm(
        args.llm,
        config=config.generation,
        api_key=settings.openai_api_key or os.environ.get(config.generation.api_key_env),
    )
    judge = build_judge(args.judge, config=config.generation)
    for name, (mode, rerank_kind) in CONFIGURATIONS.items():
        if name in checkpoint["configs"]:
            continue
        reranker = build_reranker(rerank_kind, config=config.rerank)
        retriever = Retriever.from_config(
            store=store, embedder=embedder, config=config, reranker=reranker
        )
        tier1 = evaluate_tier1(retriever=retriever, cases=cases, k=5, mode=mode)
        tier2 = evaluate_tier2(
            retriever=retriever,
            llm=llm,
            cases=cases,
            judge=judge,
            config=config,
            mode=mode,
            k=5,
            reranker=reranker,
        )
        checkpoint["configs"][name] = {
            "metrics": _metrics(tier1, tier2),
            "hit_vector": [case.hit for case in tier1.cases],
            "case_ids": [case.id for case in tier1.cases],
        }
        _write_json(checkpoint_path, checkpoint)
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git executable not found")
    commit = subprocess.run(  # noqa: S603 - fixed executable and literal arguments
        [git, "rev-parse", "HEAD"], capture_output=True, check=True, text=True
    ).stdout.strip()
    scorecard = build_scorecard(
        checkpoint=checkpoint,
        cases=cases,
        commit=commit,
        documents=documents,
        chunks=chunks,
        golden_path=args.golden.replace("\\", "/"),
        corpus_path=args.corpus.replace("\\", "/"),
        cost_usd=estimate.estimated_usd,
        billed=billed,
    )
    _write_json(Path(args.report), scorecard)
    print(json.dumps(scorecard, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
