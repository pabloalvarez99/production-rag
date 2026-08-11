#!/usr/bin/env python
"""Source-level ``hit@k`` over the golden set, using the M2 retriever.

    python scripts/eval_hit.py
    python scripts/eval_hit.py --embedder openai --per-branch
    python scripts/eval_hit.py --json | jq .hit_at_k

This is a **script, not the evaluation harness**. It computes one coarse metric,
reads no thresholds, and never fails a build. The Ragas harness with chunk-level
labels and a merge gate is M6 (``docs/evaluation.md``); pre-building its
abstractions around a metric that may not survive chunk-level labelling would be
the wrong kind of early.

Standard library only, like ``scripts/smoke_health.py``: it has to run inside the
API container, where the dev extras are not installed.

What it measures
----------------
For each answerable golden item it runs one retrieval and asks whether any
returned chunk came from a labelled document — matching ``expected_source_paths``
against each hit's ``source_path``, as exact strings. Both sides are relative to
the corpus root, which makes the ingest ``SOURCE`` argument part of this
contract: ingesting ``data/raw/sample`` instead of ``data/raw`` strips the
``sample/`` prefix from every stored path, every label misses, and the score
reads ``0.00`` — shaped exactly like a total retrieval failure.

Unresolvable labels (a path with no file under the corpus root) are reported
separately from misses, because that is a dataset bug and not a result.

What the number is worth
------------------------
Everything depends on which embedder built the collection. On ``fake`` the dense
branch is hash noise, so the score is a plumbing assertion — with one real
exception: BM25 weights are computed from the text in pure Python, so the sparse
branch is genuinely lexical even there. On ``openai`` it is a real measurement
over 14 items, which is a smoke test with error bars a whole document wide.

Never quote a number from this script without the embedder that produced it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = REPO_ROOT / "data" / "eval" / "golden.jsonl"
DEFAULT_CORPUS_ROOT = REPO_ROOT / "data" / "raw"
DEFAULT_K_VALUES = (1, 3, 5, 10)

EXIT_OK = 0
EXIT_RUNTIME = 1
EXIT_USAGE = 2


class EvalError(RuntimeError):
    """The evaluation cannot be run as invoked."""


def build_parser() -> argparse.ArgumentParser:
    """Define the CLI surface, so ``--help`` is testable without running."""
    parser = argparse.ArgumentParser(
        prog="python scripts/eval_hit.py",
        description=(
            "Source-level hit@k over data/eval/golden.jsonl using the M2 retriever. "
            "Reports only - no thresholds, no gate."
        ),
        epilog=(
            "The last line of stdout is a JSON summary. Exit codes: 0 ok, "
            "1 a retrieval failed, 2 the invocation is wrong."
        ),
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Golden JSONL path.")
    parser.add_argument(
        "--corpus-root",
        default=str(DEFAULT_CORPUS_ROOT),
        help="Root the labels are relative to. Used only to flag unresolvable labels.",
    )
    parser.add_argument(
        "--embedder",
        choices=("fake", "openai"),
        default="fake",
        help=(
            # ASCII only: --help may be written to a cp1252 console.
            "Query embedder. MUST match the one that built the collection - "
            "nothing detects a mismatch, both produce 1536 dimensions."
        ),
    )
    parser.add_argument("--collection", help="Target collection. Defaults to the job's config.")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML profile to load.")
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=list(DEFAULT_K_VALUES),
        help="k values to report. Retrieval requests max(k) hits.",
    )
    parser.add_argument(
        "--per-branch",
        action="store_true",
        help=(
            "Also score dense-only and sparse-only runs. Triples the number of "
            "retrievals, and on --embedder openai triples the embedding spend."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Emit only the JSON summary.")
    return parser


def load_dataset(path: Path) -> list[dict[str, object]]:
    """Read the golden JSONL, failing on the line that is malformed.

    A line number in the error is the whole point of JSONL over one big array.
    """
    if not path.is_file():
        raise EvalError(f"dataset not found: {path}")
    items: list[dict[str, object]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            item = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise EvalError(f"{path}:{lineno} is not valid JSON: {exc}") from exc
        if not isinstance(item, dict):
            raise EvalError(f"{path}:{lineno} is not a JSON object")
        items.append(item)
    if not items:
        raise EvalError(f"{path} holds no items")
    return items


def unresolvable_labels(items: list[dict[str, object]], corpus_root: Path) -> list[str]:
    """Return labelled paths with no file under *corpus_root*.

    Reported apart from misses: a label pointing at a deleted document scores as
    a permanent miss and reads exactly like a retrieval regression.
    """
    missing: list[str] = []
    for item in items:
        for raw in item.get("expected_source_paths") or []:
            path = str(raw)
            if not (corpus_root / path).is_file() and path not in missing:
                missing.append(path)
    return sorted(missing)


def retrieve(question: str, *, mode: str, top_k: int, args: argparse.Namespace) -> list[dict]:
    """Run one retrieval and return its hits.

    Shelling out rather than importing keeps this script dependency-free and
    exercises exactly the surface an operator uses. The retrieve command's
    contract is that the last line of stdout is a JSON object; logs go to stderr.
    """
    command = [
        sys.executable,
        "-m",
        "production_rag.retrieval",
        "--config",
        args.config,
        "--query",
        question,
        "--mode",
        mode,
        "--embedder",
        args.embedder,
        "--top-k",
        str(top_k),
        "--json",
    ]
    if args.collection:
        command += ["--collection", args.collection]

    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        command, capture_output=True, text=True, cwd=REPO_ROOT, check=False
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip().splitlines()
        tail = detail[-1] if detail else "no output"
        raise EvalError(
            f"retrieve exited {completed.returncode} for {question!r}: {tail}\n"
            "if the collection predates M2 it has no `sparse` vector - rebuild it "
            "with `make reingest-fake`."
        )

    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise EvalError(f"retrieve produced no output for {question!r}")
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise EvalError(f"retrieve did not end with a JSON object: {exc}") from exc
    hits = payload.get("hits", [])
    if not isinstance(hits, list):
        raise EvalError("retrieve returned a `hits` field that is not a list")
    return hits


def first_hit_rank(hits: list[dict], expected: set[str]) -> int | None:
    """1-based rank of the first hit from a labelled document, or ``None``.

    Rank is taken from list position, not from the hit's own ``rank`` field: the
    metric is about what the caller was handed, in the order it was handed over.
    """
    for position, hit in enumerate(hits, start=1):
        if str(hit.get("source_path", "")) in expected:
            return position
    return None


def branch_of(hits: list[dict], rank: int) -> str:
    """Which branch put the hit at *rank* into the fused list.

    Reads the hit's ``ranks`` mapping (branch name to that branch's rank), which
    fusion carries for exactly this question. Contribution, not a per-branch
    score: a hit both branches returned says nothing about whether either would
    have found it alone. Use ``--per-branch`` for that.
    """
    ranks = hits[rank - 1].get("ranks") or {}
    if not isinstance(ranks, dict):
        return "unknown"
    dense = "dense" in ranks
    sparse = "sparse" in ranks
    if dense and sparse:
        return "both"
    if sparse:
        return "sparse_only"
    if dense:
        return "dense_only"
    return "unknown"


def score(items: list[dict[str, object]], *, mode: str, args: argparse.Namespace) -> dict:
    """Run every answerable item and aggregate hit@k for one retrieval mode."""
    k_values = sorted(set(args.k))
    top_k = max(k_values)

    scored = 0
    hits_at: dict[int, int] = dict.fromkeys(k_values, 0)
    by_category: dict[str, dict[str, int]] = {}
    contribution: dict[str, int] = {"both": 0, "dense_only": 0, "sparse_only": 0, "unknown": 0}
    misses: list[str] = []

    for item in items:
        expected = {str(path) for path in (item.get("expected_source_paths") or [])}
        if not expected:
            # Unanswerable items have nothing to hit. They measure the refusal
            # path, which needs generation (M4), so they are counted and skipped.
            continue

        scored += 1
        category = str(item.get("category", "uncategorised"))
        bucket = by_category.setdefault(category, {"scored": 0, "hits": 0})
        bucket["scored"] += 1

        hits = retrieve(str(item["question"]), mode=mode, top_k=top_k, args=args)
        rank = first_hit_rank(hits, expected)

        if rank is None:
            misses.append(str(item.get("id", "?")))
            continue
        for k in k_values:
            if rank <= k:
                hits_at[k] += 1
        # Category and contribution are reported at the largest k: at smaller k
        # they measure ordering, which is what the hit@k table already shows.
        bucket["hits"] += 1
        contribution[branch_of(hits, rank)] += 1

    return {
        "mode": mode,
        "scored": scored,
        "hit_at_k": {str(k): round(hits_at[k] / scored, 4) if scored else 0.0 for k in k_values},
        "hits_at_k": {str(k): hits_at[k] for k in k_values},
        "by_category": by_category,
        "branch_contribution": contribution,
        "misses": misses,
    }


def render(result: dict, *, embedder: str, top_k: int) -> None:
    """Print the human-readable report to stdout."""
    print(f"\nhit@k  ({result['mode']} retrieval, --embedder {embedder}, top {top_k})")
    print(f"  scored {result['scored']} answerable items\n")
    print("    k   hit@k   hits")
    for k, value in result["hit_at_k"].items():
        print(f"  {k:>3}   {value:5.2f}   {result['hits_at_k'][k]}")

    print("\n  by category (at the largest k)")
    for category, counts in sorted(result["by_category"].items()):
        rate = counts["hits"] / counts["scored"] if counts["scored"] else 0.0
        print(f"    {category:<14} {rate:5.2f}   {counts['hits']}/{counts['scored']}")

    print("\n  which branch found the first correct hit")
    for branch, count in result["branch_contribution"].items():
        if count:
            print(f"    {branch:<14} {count}")

    if result["misses"]:
        print(f"\n  missed entirely: {', '.join(result['misses'])}")


def main(argv: list[str] | None = None) -> int:
    """Run the evaluation and return the process exit code."""
    args = build_parser().parse_args(argv)

    try:
        items = load_dataset(Path(args.dataset))
    except EvalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    orphans = unresolvable_labels(items, Path(args.corpus_root))
    modes = ["hybrid"] + (["dense", "sparse"] if args.per_branch else [])

    try:
        results = [score(items, mode=mode, args=args) for mode in modes]
    except EvalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_RUNTIME

    unanswerable = sum(1 for item in items if not (item.get("expected_source_paths") or []))
    summary = {
        "ok": True,
        "embedder": args.embedder,
        "dataset": str(args.dataset),
        "items": len(items),
        "unanswerable_skipped": unanswerable,
        "unresolvable_labels": orphans,
        "results": results,
        # Convenience for `jq .hit_at_k`: the fused run is the headline.
        "hit_at_k": results[0]["hit_at_k"],
    }

    if not args.json:
        for result in results:
            render(result, embedder=args.embedder, top_k=max(args.k))
        if orphans:
            print("\n  labels with no file under the corpus root (dataset bug, not a miss):")
            for path in orphans:
                print(f"    {path}")
        if unanswerable:
            print(f"\n  {unanswerable} unanswerable item(s) skipped: no document to hit.")
        reading = "plumbing, not quality" if args.embedder == "fake" else "a smoke test"
        print(f"\n  read this as: {reading}.\n  never quote it without the embedder.\n")

    json.dump(summary, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
