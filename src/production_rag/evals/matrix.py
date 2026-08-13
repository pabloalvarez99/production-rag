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
    UsageLedger,
    derive_call_counts,
    estimate_cost,
    exact_token_count,
    format_preflight,
    format_spend_summary,
    require_spending_consent,
)
from production_rag.evals.provenance import assert_collection_embedder
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
from production_rag.ingest.chunking import chunk_document
from production_rag.ingest.cli import configure_cli_logging, resolve_embedder, resolve_store
from production_rag.ingest.loaders import iter_documents
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
    parser.add_argument("--judge", choices=("none",), default="none")
    parser.add_argument("--ingest", action="store_true", help="Ingest once before the sweep.")
    parser.add_argument("--yes-spend", action="store_true")
    parser.add_argument("--checkpoint", default="data/eval/reports/matrix-checkpoint.json")
    parser.add_argument("--report", default="data/eval/reports/scorecard.json")
    parser.add_argument("--log-level")
    return parser


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _corpus_profile(corpus: str, config: Any) -> tuple[int, list[str]]:
    documents = 0
    embed_texts: list[str] = []
    for document in iter_documents(
        corpus,
        include_extensions=config.ingest.include_extensions,
        exclude_globs=config.ingest.exclude_globs,
    ):
        chunks, _ = chunk_document(document, config.ingest.chunking)
        if chunks:
            documents += 1
            embed_texts.extend(chunk.embed_text for chunk in chunks)
    return documents, embed_texts


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
    cost_estimate_usd: float,
    cost_expected_usd: float,
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
            ci = paired_percentile_bootstrap(
                a,
                b,
                seed=int(checkpoint["bootstrap_seed"]),
                resamples=int(checkpoint["bootstrap_resamples"]),
            )
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
            "cost_estimate_usd": cost_estimate_usd,
            "cost_expected_usd": cost_expected_usd,
            "billed": billed,
            "bootstrap_seed": checkpoint["bootstrap_seed"],
            "bootstrap_resamples": checkpoint["bootstrap_resamples"],
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
    documents, embed_texts = _corpus_profile(args.corpus, config)
    chunks = len(embed_texts)
    billed = any(provider != "fake" for provider in (args.embedder, args.llm))
    calls = derive_call_counts(
        items=len(cases),
        configurations=CONFIGURATIONS,
        chunks_to_embed=chunks,
        ingest=args.ingest,
        llm_provider=args.llm,
        judge_provider="none",
    )
    embedding_tokens = (
        exact_token_count(embed_texts, model=config.ingest.embedding.model)
        if billed and args.ingest and args.embedder == "openai"
        else 0
    )
    query_tokens = (
        exact_token_count(
            (case.question for case in cases for _ in range(6)),
            model=config.ingest.embedding.model,
        )
        if billed and args.embedder == "openai"
        else 0
    )
    estimate = estimate_cost(
        calls=calls,
        embedding_tokens=embedding_tokens,
        query_tokens=query_tokens,
        max_context_tokens=config.generation.max_context_tokens,
        max_output_tokens=config.generation.max_output_tokens,
        billed=billed,
    )
    for line in format_preflight(estimate):
        print(line)
    require_spending_consent(billed=billed, yes_spend=args.yes_spend)
    ledger = UsageLedger(billed=billed, upper_bound_usd=estimate.upper_bound_usd)
    embedder = resolve_embedder(
        args.embedder,
        config=config,
        settings=settings,
        usage_recorder=ledger.record_embedding,
    )
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
            "judge": "none",
            "configs": {},
        }
    store = resolve_searchable_store(
        config=config, settings=settings, collection=args.collection, url=url
    )
    llm = build_llm(
        args.llm,
        config=config.generation,
        api_key=settings.openai_api_key or os.environ.get(config.generation.api_key_env),
        usage_recorder=ledger.record_generation,
    )
    assert_collection_embedder(store, expected_model=embedder.model)
    judge = None
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
        cost_usd=ledger.cost_usd,
        cost_estimate_usd=estimate.upper_bound_usd,
        cost_expected_usd=estimate.expected_usd,
        billed=billed,
    )
    _write_json(Path(args.report), scorecard)
    print(format_spend_summary(estimate=estimate, actual_usd=ledger.cost_usd))
    print(json.dumps(scorecard, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
