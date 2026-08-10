# data/eval

Versioned evaluation datasets: question / gold-answer pairs used by the eval
harness (see `docs/evaluation.md` and ADR 0003).

Rules:

- Committed to the repo; changes are reviewed like code.
- Never written by the runtime or the ingest job.
- Keep the dataset small and stable — it is a regression detector, not a
  benchmark.
