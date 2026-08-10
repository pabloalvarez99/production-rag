# Data Model

How data is shaped across the corpus, the chunk artifacts, and the Qdrant
collection. Scope: M0.

## Corpus documents (`data/raw/`)

Plain Markdown or text files. The file itself is the unit of provenance:

- `source_path` — path relative to `data/raw/`, e.g. `sample/00-intro.md`.
- The first Markdown heading, when present, is used as the document title.

Only extensions listed in `configs/default.yaml → ingest.include_extensions`
are ingested; anything else is skipped loudly.

## Chunks

Documents are split with a recursive strategy: structural boundaries first
(headings, then paragraphs, then sentences), falling back to a hard character
cut. Chunk size and overlap are set in `configs/default.yaml → ingest.chunking`.

Each chunk carries:

| Field | Type | Notes |
|-------|------|-------|
| `chunk_id` | string (UUID) | Stable per (document, chunk index, text hash). |
| `text` | string | The chunk body as embedded. |
| `source_path` | string | Provenance, see above. |
| `title` | string \| null | Document title when known. |
| `heading` | string \| null | Nearest enclosing heading. |
| `chunk_index` | int | Position within the document. |

## Qdrant collection

One collection (default name `production_rag`, configurable via
`QDRANT_COLLECTION`) holds everything:

- **Dense vectors** — named vector for semantic similarity; dimension and
  distance metric follow the embedding model configured in
  `configs/default.yaml → qdrant`.
- **Sparse vectors** — named sparse vector for BM25-style lexical matching,
  enabling hybrid queries through Qdrant's query API (see
  [ADR 0001](adr/0001-hybrid-qdrant.md)).
- **Payload** — the chunk fields above, stored alongside the vectors so
  retrieval returns citable provenance without a second store.

There is deliberately no separate metadata database in M0: Qdrant payloads
cover every field the query path reads.

## Derived artifacts (`data/processed/`)

Chunking/embedding caches the ingest job may write to avoid recomputation.
Gitignored, excluded from the Docker build context, and safe to delete at any
time — the ingest job rebuilds them from `data/raw/`.

## Eval datasets (`data/eval/`)

Question/answer pairs used by the evaluation harness (see
[evaluation.md](evaluation.md)). Committed, versioned with the repo, and
never written by the runtime.
