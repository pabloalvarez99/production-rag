"""Observability seams: request context and an optional trace export.

Three modules, in dependency order:

* :mod:`~production_rag.observability.context` — the request id, and the
  contextvar binding that puts it on every log line a request emits.
* :mod:`~production_rag.observability.tracer` — a ``Tracer`` Protocol, the
  :class:`~production_rag.observability.tracer.NullTracer` that is the default,
  a recording double for tests, and an OpenTelemetry adapter.
* :mod:`~production_rag.observability.langfuse_client` — the Langfuse adapter,
  lazily imported and never required.

The shape follows ADR-0006: **structured logs and per-node timings are the
baseline signal and carry no vendor; tracing is an export, never a dependency.**
Nothing in :mod:`production_rag.retrieval` or :mod:`production_rag.generation`
imports this package — tracing attaches at the pipeline boundary and at the graph
nodes, which are already adapters rather than logic.

No re-exports, matching the other packages here. Import the module:

    from production_rag.observability.tracer import build_tracer
"""

from __future__ import annotations
