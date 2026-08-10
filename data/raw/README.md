# data/raw

Source corpus. Committed, hand-curated, and the **source of truth** for
everything downstream — the vector index is always rebuildable from here.

- Only extensions in `configs/default.yaml → ingest.include_extensions`
  (`.md`, `.markdown`, `.txt`) are ingested; anything else is skipped loudly.
- `sample/` holds two tiny RAG-domain docs so a fresh clone can run a real
  end-to-end ingest without staging a corpus.
- Large or private corpora do not belong in git; mount or symlink them and
  keep them out of commits.
