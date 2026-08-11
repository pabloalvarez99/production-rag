"""Evaluation harness: measurable retrieval quality, offline where possible.

M2 ships one metric, source hit@k — see :mod:`production_rag.evals.source_hit`.
It is deliberately the cheapest useful one: it needs no judge model, no API key
and no human labels, only the golden set's ``expected_source_paths``, so it can
run on every change and give a number that is comparable across runs.

Answer-quality metrics (faithfulness, answer relevance, the Ragas gate) belong to
the milestones that add generation. Claiming them here would be dishonest: there
is nothing generating an answer yet to evaluate.

No re-exports, for the same import-cycle reason as the other packages.
"""
