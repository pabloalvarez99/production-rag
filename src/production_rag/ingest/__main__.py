"""Makes ``python -m production_rag.ingest`` runnable.

Kept to one line of logic so the CLI itself stays importable and testable
without a subprocess.
"""

from __future__ import annotations

from production_rag.ingest.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
