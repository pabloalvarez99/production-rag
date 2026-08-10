# `data/raw/` — source corpus

Input to the ingest job and the single source of truth for the vector index.
Everything downstream (`data/processed/`, the Qdrant collection) is derived and
can be rebuilt from here.

## Layout

```
data/raw/
  sample/                 # tiny committed corpus, enough to smoke-test ingest
    00-intro.md
    01-hybrid-search.md
  <your-corpus>/          # add your own folders here
```

The **first path segment under `data/raw/` becomes the `source` payload field**,
which is a filterable, keyword-indexed field. Group documents into folders that
match how you would want to filter them later.

## What gets ingested

Only extensions listed in `configs/default.yaml → ingest.include_extensions`
(`.md`, `.markdown`, `.txt` by default). Anything else is skipped and logged —
a silently ignored PDF looks exactly like an empty corpus, so the skip is loud
on purpose.

Dotfiles and `node_modules` are excluded via `ingest.exclude_globs`.

## Conventions

- **Headings carry weight.** Chunking splits on `##` / `###` before falling back
  to paragraphs, and each chunk is prefixed with its heading path before
  embedding. Well-structured Markdown retrieves measurably better than a wall of
  text.
- **Optional YAML front matter** is read for `title` and `tags`:

  ```markdown
  ---
  title: Hybrid search
  tags: [rag, retrieval]
  ---
  ```

  Without front matter, the first H1 becomes the title.
- **Stable filenames.** `doc_id` is derived from the path relative to
  `data/raw/`. Renaming a file orphans its old chunks until the next full
  re-ingest.

## Runtime access

`data/` is bind-mounted **read-only** into the API container at `/app/data`, so
new documents are visible without an image rebuild. The ingest job never writes
here.

## Committing corpora

Only `sample/` is committed. Real corpora are usually large, often licensed, and
sometimes sensitive — keep them out of git and mount or copy them locally. If
you add a corpus that must not be committed, add it to `.gitignore` in the same
commit, not later.
