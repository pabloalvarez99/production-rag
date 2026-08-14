"""production-rag — a production-grade Retrieval-Augmented Generation service.

The package is deliberately thin at milestone M0: only configuration and the
HTTP surface (liveness/readiness) exist. Retrieval, reranking, generation and
evaluation land in later milestones and each get their own subpackage so that
every stage stays testable in isolation.

Public surface::

    from production_rag import Settings, get_settings, __version__

The application factory lives in :mod:`production_rag.main` and is imported
lazily (``uvicorn production_rag.main:app``) so that importing this package
does not pull FastAPI into processes that only need configuration — for
example a future ingest CLI or an evaluation runner.
"""

from production_rag.config import Settings, get_settings

__version__ = "1.0.0"

__all__ = [
    "Settings",
    "__version__",
    "get_settings",
]
