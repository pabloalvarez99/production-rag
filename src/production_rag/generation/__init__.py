"""Generation side: prompt assembly, the LLM seam, citations and guardrails.

Everything here is framework-free and synchronous. The LangGraph nodes in
:mod:`production_rag.graph` are thin adapters over these functions (ADR-0002),
which is what keeps every stage callable — and unit-testable — without
importing a graph library.

No re-exports, matching :mod:`production_rag.retrieval`. Import the module:

    from production_rag.generation.generator import generate_answer
"""

from __future__ import annotations
