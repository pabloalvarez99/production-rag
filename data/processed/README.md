# `data/processed/` — derived artifacts

Everything here is **generated and disposable**. Deleting the whole directory is
always safe; the ingest and eval jobs rebuild what they need from
[`data/raw/`](../raw/README.md) and [`data/eval/`](../eval/README.md).

## Contents

| Path | Written by | Purpose |
|---|---|---|
| `chunks/` | ingest | materialised chunks with their content hashes, used to skip re-embedding unchanged text |
| `embeddings-cache/` | ingest | cached vectors keyed by content hash — this is what makes a re-run free instead of expensive |
| `eval-runs/<timestamp>/` | evals | `summary.json` plus per-query `details.jsonl` |
| `manifests/` | ingest | per-run record: file count, chunk count, config snapshot |

## Not committed

`data/processed` is listed in `.gitignore` and in `.dockerignore`. Two separate
reasons, both load-bearing:

- **git** — these files are large, binary-ish, and change on every run. They
  would dominate the history while carrying no reviewable information.
- **docker** — derived state baked into an image goes stale the moment the
  corpus changes, and produces an image that behaves differently from a
  freshly-built one. The ingest job regenerates it at runtime.

`.gitkeep` exists so the directory is present in a fresh clone and the ingest
job does not have to create it.

## Caution

The embedding cache is keyed by content hash **only** — it does not include the
embedding model name. Changing `ingest.embedding.model` without clearing this
directory will silently mix vectors from two models in one collection, which
degrades retrieval in a way that looks like a chunking bug. Clear it on any
embedding model change:

```bash
rm -rf data/processed/embeddings-cache
```
