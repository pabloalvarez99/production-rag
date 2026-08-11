"""Exit non-zero when golden labels cannot be hit by the configured chunker."""

from __future__ import annotations

import argparse
from pathlib import Path

from production_rag.evals.golden_integrity import check_golden_integrity


def main() -> int:
    """Run the standalone golden integrity gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    parser.add_argument("--golden", type=Path, default=Path("data/eval/golden-corpus.jsonl"))
    args = parser.parse_args()
    result = check_golden_integrity(args.corpus, args.golden)
    if result.errors:
        print(f"Golden integrity FAILED ({result.items} items):")
        for error in result.errors:
            print(f"- {error}")
        return 1
    print(
        f"Golden integrity OK: {result.items} items; "
        f"{result.chunkable_documents} chunkable documents"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
