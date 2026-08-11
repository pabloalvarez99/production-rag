from pathlib import Path

from production_rag.evals.golden_integrity import check_golden_integrity


def test_corpus_golden_is_integral_and_chunkable() -> None:
    result = check_golden_integrity(Path("data/corpus"), Path("data/eval/golden-corpus.jsonl"))
    assert result.items == 60
    assert result.errors == ()
