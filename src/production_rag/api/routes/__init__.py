"""Routers for the operational endpoints, one module per resource.

M0 ships the operational surface only. ``/v1/query``, ``/v1/ingest`` and
``/v1/eval`` join it in later milestones as sibling modules.
"""

from production_rag.api.routes import health, ready

__all__ = ["health", "ready"]
