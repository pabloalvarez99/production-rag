"""Offline ingest: corpus on disk to vectors in Qdrant.

A batch job, not part of the request path. Run it with

    python -m production_rag.ingest --source data/raw/sample --embedder fake

Deliberately empty of imports. ``retrieval.store`` needs ``ingest.models`` and
``ingest.pipeline`` needs ``retrieval.store``; re-exporting either package's
contents from its ``__init__`` turns that legal dependency into an import cycle.
Import the module you want:

    from production_rag.ingest.pipeline import run_ingest
"""

from __future__ import annotations
