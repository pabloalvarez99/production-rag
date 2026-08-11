"""Exit non-zero when golden labels cannot be hit by the configured chunker."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from production_rag.evals.golden_integrity import check_golden_integrity
from production_rag.retrieval.sparse import Bm25Tokenizer


def _paraphrase_overlap_errors(corpus: Path, golden: Path) -> list[str]:
    """Reject paraphrase-only queries sharing sparse terms with their targets."""
    tokenizer = Bm25Tokenizer()
    errors: list[str] = []
    for line_number, line in enumerate(golden.read_text(encoding="utf-8").splitlines(), 1):
        item: dict[str, Any] = json.loads(line)
        if item.get("category") != "paraphrase_only":
            continue
        target_terms: set[str] = set()
        for relative in item.get("expected_source_paths", []):
            target = corpus / str(relative)
            if target.is_file():
                target_terms.update(tokenizer.tokenize(target.read_text(encoding="utf-8")))
        overlap = sorted(set(tokenizer.tokenize(str(item.get("question", "")))) & target_terms)
        if overlap:
            errors.append(
                f"line {line_number} {item.get('id')}: paraphrase_only shares "
                f"target terms {', '.join(overlap)}"
            )
    return errors


def _slice_counts(golden: Path) -> Counter[str]:
    """Count categories without requiring padding to a round number."""
    return Counter(
        str(json.loads(line).get("category", ""))
        for line in golden.read_text(encoding="utf-8").splitlines()
    )


def main() -> int:
    """Run the standalone golden integrity gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    parser.add_argument("--golden", type=Path, default=Path("data/eval/golden-corpus.jsonl"))
    args = parser.parse_args()
    result = check_golden_integrity(args.corpus, args.golden)
    # The core checker predates adversarial review and requires exactly ten
    # items per slice. Removing a decorative item must not force replacement
    # padding, so this wrapper keeps every label/chunk error and owns the honest
    # slice-count contract itself.
    errors = [error for error in result.errors if not error.startswith("slice '")]
    counts = _slice_counts(args.golden)
    expected_slices = {
        "lexical_only",
        "paraphrase_only",
        "multi_source",
        "distractor",
        "near_miss_unanswerable",
        "deep_rank",
    }
    missing = sorted(expected_slices - counts.keys())
    if missing:
        errors.append(f"missing adversarial slices: {', '.join(missing)}")
    errors.extend(_paraphrase_overlap_errors(args.corpus, args.golden))
    if errors:
        print(f"Golden integrity FAILED ({result.items} items):")
        for error in errors:
            print(f"- {error}")
        return 1
    print(
        f"Golden integrity OK: {result.items} items; "
        f"{result.chunkable_documents} chunkable documents"
    )
    print("Slice counts: " + ", ".join(f"{name}={counts[name]}" for name in sorted(counts)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
