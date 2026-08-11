"""Retrieval side: embedding providers and the vector store.

Written in M1 for the ingest path (embed and upsert). The query path — hybrid
search, fusion, rerank — lands in M2 and M3 against these same seams.

No re-exports here, for the same reason as :mod:`production_rag.ingest`: this
package and that one depend on each other at module level, and a re-exporting
``__init__`` would make importing either one first a cycle. Import the module:

    from production_rag.retrieval.store import QdrantStore
"""

from __future__ import annotations
