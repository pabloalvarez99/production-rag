"""The unified eval runner: one command, two tiers, one JSON report.

    python -m production_rag.evals.run --tier all

Both tiers over one golden set, in one process, producing one versioned report,
because ADR-0003's stated cost of the two-tier split is "two commands and two
sets of results to reconcile" — and that reconciliation is what nobody does.
Retrieval and generation numbers computed from the same sample, in the same run,
against the same collection, are comparable by construction.

Defaults are the offline ones: fake embedder, fake model, fake judge. That makes
the default invocation free, deterministic and runnable in CI, and it makes the
numbers it produces *plumbing* numbers rather than quality numbers. The report
says which, in the field ``offline_defaults``, so nobody has to reconstruct the
invocation from the score.

Anything that costs money is opt-in twice: a flag, and — for the judge — the
``RUN_LLM_EVALS=1`` environment variable plus a credential. A judge that runs
because someone typed the wrong flag is a bill and a surprise.

Exit codes: ``0`` ok, ``1`` the run failed or a gate was not met, ``2`` the
invocation or configuration is wrong.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import structlog

from production_rag.config import Settings
from production_rag.config import get_settings as get_settings
from production_rag.config_loader import ConfigFileError, YamlConfig, load_yaml_config
from production_rag.evals.judges import (
    JUDGE_FAKE,
    JUDGE_KINDS,
    LLM_EVALS_ENV,
    AnswerJudge,
    JudgeError,
    build_judge,
)
from production_rag.evals.provenance import assert_collection_embedder
from production_rag.evals.source_hit import (
    DEFAULT_GOLDEN_PATH,
    DEFAULT_K,
    EvalError,
    GoldenCase,
    load_golden,
)
from production_rag.evals.tier1_retrieval import Tier1Report, evaluate_tier1
from production_rag.evals.tier2_answer import Tier2Report, evaluate_tier2
from production_rag.generation.llm import LLM_KINDS, LLMError, build_llm
from production_rag.ingest.cli import (
    EXIT_OK,
    EXIT_RUNTIME,
    EXIT_USAGE,
    configure_cli_logging,
    resolve_embedder,
)
from production_rag.retrieval.cli import resolve_searchable_store
from production_rag.retrieval.embeddings import EmbeddingError
from production_rag.retrieval.hybrid import RETRIEVAL_MODES, RetrievalError
from production_rag.retrieval.hybrid import Retriever as Retriever
from production_rag.retrieval.rerank import RERANK_KINDS, RERANK_OFF, RerankError, build_reranker
from production_rag.retrieval.sparse import SparseError
from production_rag.retrieval.store import CollectionMismatchError, VectorStoreError

_log = structlog.get_logger(__name__)

REPORT_VERSION = 1
"""Schema version of the emitted report.

Versioned because these files get committed, compared across weeks and read by
scripts. A consumer that cannot tell an old report from a new one silently
compares two different metrics.
"""

TIER_1 = "1"
TIER_2 = "2"
TIER_ALL = "all"
TIERS = (TIER_1, TIER_2, TIER_ALL)

DEFAULT_SEED = 42
"""ADR-0003's seed. Fixed so two runs over a sample are comparable."""


def sample_cases(cases: Sequence[GoldenCase], *, sample: int | None, seed: int) -> list[GoldenCase]:
    """Take a reproducible subset of the golden set.

    Args:
        cases: Every loaded case.
        sample: How many to keep, or ``None`` for all of them.
        seed: PRNG seed.

    Returns:
        The selected cases **in golden-set order**, not in draw order. Sampling
        decides membership; order is the file's, so two runs with different
        sample sizes produce reports whose shared cases line up.
    """
    if sample is None or sample >= len(cases):
        return list(cases)
    if sample <= 0:
        raise EvalError(f"--sample must be positive, got {sample}")
    # Seeded and reproducible by design; nothing here is a security decision.
    chosen = set(random.Random(seed).sample(range(len(cases)), sample))  # noqa: S311
    return [case for index, case in enumerate(cases) if index in chosen]


def resolve_judge(
    kind: str,
    *,
    config: YamlConfig,
    settings: Settings,
    allow_llm_evals: bool | None = None,
) -> AnswerJudge:
    """Build the judge, refusing to spend money that was not asked for.

    Args:
        kind: One of :data:`~production_rag.evals.judges.JUDGE_KINDS`.
        config: The YAML profile, for the judge model's settings.
        settings: Environment settings, for the credential.
        allow_llm_evals: Override for the ``RUN_LLM_EVALS`` gate, for tests.

    Returns:
        The judge.

    Raises:
        JudgeError: A hosted judge was asked for without the environment opt-in,
            or without a credential. Both are refusals to start rather than
            silent downgrades to the fake judge: a run that quietly swapped its
            judge would report offline numbers under a hosted judge's name.
    """
    if kind == JUDGE_FAKE:
        return build_judge(JUDGE_FAKE)

    enabled = (
        os.environ.get(LLM_EVALS_ENV, "").strip() == "1"
        if allow_llm_evals is None
        else allow_llm_evals
    )
    if not enabled:
        raise JudgeError(
            f"judge {kind!r} costs money and varies between runs; set {LLM_EVALS_ENV}=1 to allow it"
        )
    api_key = settings.openai_api_key or os.environ.get(config.generation.api_key_env)
    if not api_key:
        raise JudgeError(f"judge {kind!r} needs a credential in {config.generation.api_key_env}")
    return build_judge(kind, config=config.generation, api_key=api_key)


def build_parser() -> argparse.ArgumentParser:
    """Define the CLI surface."""
    parser = argparse.ArgumentParser(
        prog="python -m production_rag.evals.run",
        description=(
            "Run the two-tier evaluation over the golden set: tier 1 scores "
            "retrieval, tier 2 scores the answers the pipeline produces."
        ),
        epilog=(
            "The last line of stdout is a versioned JSON report with aggregates "
            "and per-case results. Defaults are offline (fake embedder, fake "
            "model, fake judge), which makes them free and deterministic and "
            "makes the scores a plumbing check rather than a quality claim. "
            "Exit codes: 0 ok, 1 the run failed or a gate was not met, 2 the "
            "invocation or configuration is wrong."
        ),
    )
    parser.add_argument(
        "--tier",
        choices=TIERS,
        default=TIER_ALL,
        help="Which tier to run. Default 'all'.",
    )
    parser.add_argument(
        "--golden",
        default=str(DEFAULT_GOLDEN_PATH),
        help=f"JSONL golden set. Default {DEFAULT_GOLDEN_PATH}.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"Retrieval cutoff, and top_k for tier 2. Default {DEFAULT_K}.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        help="Score a random subset of this many cases. Default: every case.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Sampling seed, so a sampled run is reproducible. Default {DEFAULT_SEED}.",
    )
    parser.add_argument(
        "--mode",
        choices=RETRIEVAL_MODES,
        help="Retrieval mode. Default from retrieval.mode ('hybrid').",
    )
    parser.add_argument(
        "--embedder",
        choices=("fake", "openai"),
        default="fake",
        help=(
            # ASCII only, as in the other CLIs: cp1252 consoles choke on an em dash.
            "Embedding provider. Default 'fake': runs with no API key, but dense "
            "results are arbitrary, so a fake-embedder score measures BM25 plus "
            "plumbing, not semantic retrieval quality."
        ),
    )
    parser.add_argument(
        "--llm",
        choices=LLM_KINDS,
        default="fake",
        help=(
            "Generator for tier 2. Default 'fake': extractive, offline and "
            "deterministic, so the citation and refusal metrics are real while "
            "the answers are not written by a model."
        ),
    )
    parser.add_argument(
        "--judge",
        choices=JUDGE_KINDS,
        default=JUDGE_FAKE,
        help=(
            "Judge for the tier 2 faithfulness and relevance columns. Default "
            "'fake': lexical overlap, offline, and not a semantic judgement. "
            f"A hosted judge also needs {LLM_EVALS_ENV}=1 and a credential."
        ),
    )
    parser.add_argument(
        "--rerank",
        choices=RERANK_KINDS,
        default=RERANK_OFF,
        help="Rerank the fused candidates before scoring. Default 'off'.",
    )
    parser.add_argument(
        "--fail-under-hit",
        type=float,
        default=0.0,
        help=(
            "Exit 1 when the tier 1 source hit@k falls below this. Default 0.0, "
            "i.e. reporting only. ADR-0003 sets thresholds from a baseline run, "
            "not from ambition."
        ),
    )
    parser.add_argument(
        "--report",
        help="Also write the JSON report to this path.",
    )
    parser.add_argument(
        "--no-answers",
        action="store_true",
        help=(
            "Omit generated answers from the report. The answers quote corpus "
            "text, and a report file gets pasted around."
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
    parser.add_argument("--qdrant-url", help="Qdrant base URL. Defaults to QDRANT_URL.")
    parser.add_argument("--log-level", help="Log level for this run. Defaults to LOG_LEVEL.")
    return parser


def _emit(payload: dict[str, Any]) -> None:
    """Write one JSON object as the final line of stdout."""
    json.dump(payload, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    sys.stdout.flush()


def _fail(message: str, kind: str, code: int) -> int:
    """Report a failure on both channels and return *code*."""
    _log.error("eval_run_failed", error=message, error_type=kind)
    _emit({"ok": False, "report_version": REPORT_VERSION, "error": message, "error_type": kind})
    return code


def _write_report(path: str, payload: dict[str, Any]) -> None:
    """Write the report to disk, creating the parent directory."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _log.info("eval_report_written", path=str(destination))


def build_report(
    *,
    args: argparse.Namespace,
    cases: Sequence[GoldenCase],
    total_cases: int,
    tier1: Tier1Report | None,
    tier2: Tier2Report | None,
) -> dict[str, Any]:
    """Assemble the versioned report from whichever tiers ran."""
    offline = args.embedder == "fake" and (tier2 is None or args.llm == "fake")
    payload: dict[str, Any] = {
        "ok": True,
        "report_version": REPORT_VERSION,
        "tier": args.tier,
        "golden_path": args.golden,
        "golden_cases": total_cases,
        "scored_cases": len(cases),
        "sample": args.sample,
        "seed": args.seed,
        "k": args.k,
        "embedder": args.embedder,
        "rerank": args.rerank,
        # Stated rather than implied: an offline run's numbers are a plumbing
        # check, and the reader of a committed report was not at the terminal.
        "offline_defaults": offline,
        "case_ids": [case.id for case in cases],
    }
    if tier1 is not None:
        payload["tier1"] = tier1.to_summary()
    if tier2 is not None:
        payload["tier2"] = tier2.to_summary(include_answers=not args.no_answers)
        payload["llm"] = args.llm
        payload["judge"] = tier2.judge
    return payload


def main(argv: Sequence[str] | None = None) -> int:  # noqa: C901 - a CLI's error paths
    """Run the requested tiers and return the process exit code."""
    effective_argv = list(argv) if argv is not None else sys.argv[1:]
    if "--matrix" in effective_argv:
        from production_rag.evals.matrix import main as matrix_main

        effective_argv.remove("--matrix")
        return matrix_main(effective_argv)
    args = build_parser().parse_args(effective_argv)
    settings = get_settings()
    configure_cli_logging(args.log_level or settings.log_level)

    try:
        config = load_yaml_config(args.config or settings.config_path)
        all_cases = load_golden(args.golden)
        cases = sample_cases(all_cases, sample=args.sample, seed=args.seed)
    except (ConfigFileError, EvalError) as exc:
        return _fail(str(exc), type(exc).__name__, EXIT_USAGE)

    wants_tier1 = args.tier in (TIER_1, TIER_ALL)
    wants_tier2 = args.tier in (TIER_2, TIER_ALL)

    try:
        embedder = resolve_embedder(args.embedder, config=config, settings=settings)
        judge = resolve_judge(args.judge, config=config, settings=settings) if wants_tier2 else None
        llm = (
            build_llm(
                args.llm,
                config=config.generation,
                api_key=settings.openai_api_key or os.environ.get(config.generation.api_key_env),
            )
            if wants_tier2
            else None
        )
        # Credential from the environment only, as everywhere else in this repo.
        reranker = build_reranker(
            args.rerank,
            config=config.rerank,
            api_key=os.environ.get(config.rerank.api_key_env),
        )
    except (EmbeddingError, JudgeError, LLMError, RerankError) as exc:
        return _fail(str(exc), type(exc).__name__, EXIT_USAGE)

    collection = args.collection or (
        settings.qdrant_collection
        if "qdrant_collection" in settings.model_fields_set
        else config.qdrant.collection
    )
    store = resolve_searchable_store(
        config=config,
        settings=settings,
        collection=collection,
        url=args.qdrant_url or settings.qdrant_url,
    )
    try:
        assert_collection_embedder(store, expected_model=embedder.model)
    except CollectionMismatchError as exc:
        return _fail(str(exc), type(exc).__name__, EXIT_USAGE)

    tier1: Tier1Report | None = None
    tier2: Tier2Report | None = None
    try:
        retriever = Retriever.from_config(
            store=store, embedder=embedder, config=config, reranker=reranker
        )
        if wants_tier1:
            tier1 = evaluate_tier1(retriever=retriever, cases=cases, k=args.k, mode=args.mode)
        if wants_tier2 and llm is not None:
            tier2 = evaluate_tier2(
                retriever=retriever,
                llm=llm,
                cases=cases,
                judge=judge,
                config=config,
                mode=args.mode,
                k=args.k,
            )
    except (EvalError, RetrievalError, SparseError, CollectionMismatchError) as exc:
        return _fail(str(exc), type(exc).__name__, EXIT_USAGE)
    except (EmbeddingError, LLMError, RerankError, VectorStoreError) as exc:
        return _fail(str(exc), type(exc).__name__, EXIT_RUNTIME)

    payload = build_report(
        args=args, cases=cases, total_cases=len(all_cases), tier1=tier1, tier2=tier2
    )

    gate_ok = True
    if tier1 is not None and args.fail_under_hit > 0.0:
        gate_ok = tier1.hit_at_k >= args.fail_under_hit
        payload["gate"] = {
            "metric": "source_hit_at_k",
            "threshold": args.fail_under_hit,
            "value": round(tier1.hit_at_k, 4),
            "passed": gate_ok,
        }
        payload["ok"] = gate_ok

    if args.report:
        _write_report(args.report, payload)
    _emit(payload)
    if not gate_ok:
        _log.error(
            "eval_gate_failed",
            metric="source_hit_at_k",
            threshold=args.fail_under_hit,
            value=round(tier1.hit_at_k, 4) if tier1 else None,
        )
        return EXIT_RUNTIME
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
