"""Runnable alias for the retrieval CLI.

Exists so ``python -m production_rag.retrieve`` mirrors ``python -m
production_rag.ingest``. The implementation lives in
:mod:`production_rag.retrieval.cli`, next to the retriever it drives; this package
only makes it reachable under the name a reader would guess.

No re-exports: :mod:`production_rag.retrieval.store` imports
:mod:`production_rag.ingest.models`, and a package ``__init__`` that pulls modules
in eagerly is how a legal two-way module dependency turns into an import cycle.
"""
