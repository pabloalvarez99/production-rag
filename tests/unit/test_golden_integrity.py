import os
import subprocess
import sys
from pathlib import Path


def test_corpus_golden_is_integral_and_chunkable() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "src"
    result = subprocess.run(  # noqa: S603 - fixed interpreter and repository script
        [sys.executable, str(Path("scripts/check_golden_integrity.py"))],
        capture_output=True,
        check=False,
        text=True,
        env=environment,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Golden integrity OK: 60 items" in result.stdout
