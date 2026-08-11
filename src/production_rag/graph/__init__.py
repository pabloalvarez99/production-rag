"""LangGraph orchestration for the query path (ADR-0002).

Three modules, in dependency order:

* :mod:`~production_rag.graph.state` — ``QueryState``, a project-owned Pydantic
  model rather than a framework dict.
* :mod:`~production_rag.graph.nodes` — thin adapters. Every node is a call into
  :mod:`production_rag.retrieval` or :mod:`production_rag.generation` plus a
  stopwatch; none of them contains business logic.
* :mod:`~production_rag.graph.build` — the wiring, and the only file a LangGraph
  version bump should touch.

Callers want :func:`production_rag.query_pipeline.run_query`, not this package.

No re-exports here: importing this ``__init__`` would drag LangGraph into any
process that touches the package, and the point of the ADR's constraints is that
retrieval and generation stay runnable without it.
"""

from __future__ import annotations
