"""Source hit@k: did retrieval surface the document that holds the answer?

    hit@k(query) = 1 if any of the top-k hits comes from an expected source path

Averaged over the golden set. This is the metric M2 can defend:

* **No judge, no key, no labels beyond the golden set.** It compares source paths,
  so it runs offline, in CI, on every change, for free.
* **It measures the thing this milestone changed.** Whether hybrid beats dense on
  literal-token questions is exactly a source hit@k question, and running the same
  golden set in ``--mode dense`` gives the baseline that makes the hybrid number
  mean something.

**What it does not measure**, stated because a metric that oversells itself is
worse than no metric: not whether the retrieved chunk contains the answer, only
whether it came from the right file; not ranking quality beyond the cutoff (MRR
and nDCG are the next step); and nothing at all about answer quality, which needs
generation and is a later milestone's problem.

The golden set's ``unanswerable`` cases carry no expected source, because the
correct behaviour there is a refusal — a generation-time property. Those cases are
excluded from the aggregate and counted separately in the report: scoring them as
misses would depress a number retrieval cannot improve, and scoring them as hits
would be a lie.

Per-query results are returned alongside the aggregate. An average of 0.7 without
the list of the three questions that missed is not actionable, and those three are
where the next retrieval improvement comes from.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from production_rag.config import get_settings
from production_rag.config_loader import ConfigFileError, load_yaml_config
from production_rag.ingest.cli import (
    EXIT_OK,
    EXIT_RUNTIME,
    EXIT_USAGE,
    configure_cli_logging,
    resolve_embedder,
)
from production_rag.ingest.hashing import normalise_source_path
from production_rag.retrieval.cli import resolve_searchable_store
from production_rag.retrieval.embeddings import EmbeddingError
from production_rag.retrieval.hybrid import RETRIEVAL_MODES, RetrievalError, Retriever
from production_rag.retrieval.sparse import SparseError
from production_rag.retrieval.store import CollectionMismatchError, VectorStoreError

DEFAULT_GOLDEN_PATH = Path("data/eval/golden.jsonl")
DEFAULT_K = 5

_log = structlog.get_logger(__name__)


class EvalError(RuntimeError):
    """The evaluation cannot run as specified."""


@dataclass(frozen=True, slots=True)
class GoldenCase:
    """One labelled question from the golden set.

    Only the fields the metric needs are typed; ``category`` and ``notes`` are
    carried through because a per-category breakdown is the first thing anyone
    asks for once the aggregate moves.
    """

    id: str
    question: str
    expected_source_paths: tuple[str, ...]
    category: str | None = None

    @property
    def is_scorable(self) -> bool:
        """Whether source hit@k can say anything about this case.

        The golden set deliberately contains ``unanswerable`` cases with no
        expected source: the correct behaviour there is a refusal at generation
        time, which is a later milestone's metric. Scoring them as misses would
        depress this number by something retrieval cannot fix; scoring them as
        hits would be worse. They are excluded and counted separately.
        """
        return bool(self.expected_source_paths)

    @classmethod
    def from_json(cls, raw: dict[str, Any], *, line_number: int) -> GoldenCase:
        """Parse one JSONL record.

        Raises:
            EvalError: A required field is missing or empty. A silently skipped
                case would inflate the average over the cases that remain.
        """
        case_id = raw.get("id")
        question = raw.get("question")
        expected = raw.get("expected_source_paths")
        if not isinstance(case_id, str) or not case_id:
            raise EvalError(f"golden line {line_number}: missing 'id'")
        if not isinstance(question, str) or not question.strip():
            raise EvalError(f"golden line {line_number}: missing 'question'")
        if not isinstance(expected, list):
            # An empty list is valid — that is how an unanswerable case is written.
            # A missing or non-list field is a malformed record.
            raise EvalError(f"golden line {line_number}: 'expected_source_paths' must be a list")
        category = raw.get("category")
        return cls(
            id=case_id,
            question=question,
            # Normalised with the ingest's own function: the golden set is written
            # by hand, and a stray backslash or "./" prefix must not read as a miss.
            expected_source_paths=tuple(normalise_source_path(str(path)) for path in expected),
            category=category if isinstance(category, str) else None,
        )


@dataclass(frozen=True, slots=True)
class CaseOutcome:
    """What retrieval did for one golden case."""

    id: str
    question: str
    hit: bool
    expected_source_paths: tuple[str, ...]
    retrieved_source_paths: tuple[str, ...]
    hit_rank: int | None = None
    category: str | None = None
    scored: bool = True
    """Whether this case contributes to the aggregate. ``False`` for the golden
    set's unanswerable cases, which have no expected source to hit."""

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form for the report."""
        return {
            "id": self.id,
            "question": self.question,
            "hit": self.hit,
            "hit_rank": self.hit_rank,
            "scored": self.scored,
            "category": self.category,
            "expected_source_paths": list(self.expected_source_paths),
            "retrieved_source_paths": list(self.retrieved_source_paths),
        }


@dataclass(frozen=True, slots=True)
class SourceHitReport:
    """The aggregate plus every per-query outcome."""

    mode: str
    k: int
    collection: str
    embedded_model: str
    golden_path: str
    cases: tuple[CaseOutcome, ...] = ()

    @property
    def scored_cases(self) -> tuple[CaseOutcome, ...]:
        """The cases the metric can speak about: those with an expected source."""
        return tuple(case for case in self.cases if case.scored)

    @property
    def unscored(self) -> int:
        """Cases excluded from the aggregate, i.e. the unanswerable ones."""
        return len(self.cases) - len(self.scored_cases)

    @property
    def hits(self) -> int:
        """Number of scored cases whose expected source appeared in the top *k*."""
        return sum(1 for case in self.scored_cases if case.hit)

    @property
    def score(self) -> float:
        """Source hit@k over the scored cases, in ``[0, 1]``.

        Zero when there is nothing to score, rather than a division by zero: an
        eval harness that crashes on an empty input is an eval harness nobody
        wires into CI.
        """
        scored = self.scored_cases
        return 0.0 if not scored else self.hits / len(scored)

    @property
    def by_category(self) -> dict[str, float]:
        """Hit rate per golden-set category; uncategorised and unscored excluded."""
        totals: dict[str, list[bool]] = {}
        for case in self.scored_cases:
            if case.category:
                totals.setdefault(case.category, []).append(case.hit)
        return {
            category: sum(results) / len(results) for category, results in sorted(totals.items())
        }

    def to_summary(self) -> dict[str, Any]:
        """Machine-readable summary, the CLI's last line of stdout."""
        return {
            "ok": True,
            "metric": "source_hit_at_k",
            "mode": self.mode,
            "k": self.k,
            "collection": self.collection,
            "embedded_model": self.embedded_model,
            "golden_path": self.golden_path,
            "cases": len(self.cases),
            "scored_cases": len(self.scored_cases),
            # Named, not hidden: the aggregate is over scored cases only, and a
            # reader must be able to see how many were left out and why.
            "unscored_cases": self.unscored,
            "hits": self.hits,
            "score": round(self.score, 4),
            "by_category": {
                category: round(value, 4) for category, value in self.by_category.items()
            },
            # The misses are the actionable part: they are where the next
            # retrieval improvement comes from.
            "misses": [case.to_dict() for case in self.scored_cases if not case.hit],
            "results": [case.to_dict() for case in self.cases],
        }


def load_golden(path: str | Path) -> list[GoldenCase]:
    """Read a JSONL golden set.

    Raises:
        EvalError: The file is missing, a line is not valid JSON, or a record is
            missing a required field.
    """
    candidate = Path(path)
    if not candidate.is_file():
        raise EvalError(f"golden set not found: {candidate}")
    cases = [
        GoldenCase.from_json(raw, line_number=number) for number, raw in _iter_records(candidate)
    ]
    if not cases:
        raise EvalError(f"golden set {candidate} contains no cases")
    return cases


def _iter_records(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(line number, record)`` for each non-empty line of a JSONL file."""
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            try:
                raw = json.loads(text)
            except json.JSONDecodeError as exc:
                raise EvalError(f"golden line {number} is not valid JSON: {exc}") from exc
            if not isinstance(raw, dict):
                raise EvalError(f"golden line {number} is not a JSON object")
            yield number, raw


def evaluate_source_hit(
    *,
    retriever: Retriever,
    cases: Sequence[GoldenCase],
    k: int = DEFAULT_K,
    mode: str | None = None,
) -> SourceHitReport:
    """Run every golden case and score source hit@k.

    Args:
        retriever: A configured :class:`Retriever`. Its store and embedder decide
            what "the index" means here.
        cases: The golden set.
        k: Cutoff. Passed as ``top_k`` so the retriever returns exactly the hits
            being scored; a larger candidate depth per branch is unaffected.
        mode: ``dense``, ``sparse`` or ``hybrid``. The point of exposing it is
            that a hybrid number without the two single-branch numbers is not
            evidence of anything.

    Returns:
        The report, aggregate and per-case.

    Raises:
        EvalError: *k* is not positive.
        RetrievalError: A query could not be run.
    """
    if k <= 0:
        raise EvalError(f"k must be positive, got {k}")

    outcomes: list[CaseOutcome] = []
    resolved_mode = mode or retriever.config.mode
    for case in cases:
        result = retriever.retrieve(case.question, mode=resolved_mode, top_k=k)
        expected = set(case.expected_source_paths)
        retrieved = tuple(normalise_source_path(hit.source_path) for hit in result.hits)
        hit_rank = next(
            (index for index, path in enumerate(retrieved, start=1) if path in expected),
            None,
        )
        outcomes.append(
            CaseOutcome(
                id=case.id,
                question=case.question,
                hit=hit_rank is not None,
                expected_source_paths=case.expected_source_paths,
                retrieved_source_paths=retrieved,
                hit_rank=hit_rank,
                category=case.category,
                scored=case.is_scorable,
            )
        )
        _log.debug(
            "golden_case_evaluated",
            case_id=case.id,
            hit=hit_rank is not None,
            hit_rank=hit_rank,
        )

    report = SourceHitReport(
        mode=resolved_mode,
        k=k,
        collection=retriever.store.collection,
        embedded_model=retriever.embedder.model,
        golden_path="",
        cases=tuple(outcomes),
    )
    _log.info(
        "source_hit_evaluated",
        mode=resolved_mode,
        k=k,
        cases=len(outcomes),
        scored_cases=len(report.scored_cases),
        unscored_cases=report.unscored,
        hits=report.hits,
        score=round(report.score, 4),
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    """Define the CLI surface."""
    parser = argparse.ArgumentParser(
        prog="python -m production_rag.evals.source_hit",
        description=(
            "Score source hit@k over the golden set: did retrieval surface the "
            "document that holds the answer?"
        ),
        epilog=(
            "The last line of stdout is a JSON report with the aggregate score, "
            "the per-category breakdown and every miss. Exit codes: 0 ok, "
            "1 the run failed, 2 the invocation or configuration is wrong."
        ),
    )
    parser.add_argument(
        "--golden",
        default=str(DEFAULT_GOLDEN_PATH),
        help=f"JSONL golden set. Default {DEFAULT_GOLDEN_PATH}.",
    )
    parser.add_argument(
        "--mode",
        choices=RETRIEVAL_MODES,
        help="Retrieval mode to score. Default from retrieval.mode ('hybrid').",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"Cutoff for hit@k. Default {DEFAULT_K}.",
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
    _log.error("source_hit_failed", error=message, error_type=kind)
    _emit({"ok": False, "error": message, "error_type": kind})
    return code


def main(argv: Sequence[str] | None = None) -> int:
    """Score the golden set and return the process exit code."""
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
        report = evaluate_source_hit(retriever=retriever, cases=cases, k=args.k, mode=args.mode)
    except (EvalError, RetrievalError, SparseError, CollectionMismatchError) as exc:
        return _fail(str(exc), type(exc).__name__, EXIT_USAGE)
    except (EmbeddingError, VectorStoreError) as exc:
        return _fail(str(exc), type(exc).__name__, EXIT_RUNTIME)

    summary = report.to_summary()
    summary["golden_path"] = args.golden
    _emit(summary)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
