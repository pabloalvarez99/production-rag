# `data/corpus/` — measurement corpus

This directory contains the real corpus used for retrieval measurement. It is
a vendored snapshot of the Qdrant documentation: 3,067 Markdown source
documents, 416,102 whitespace-delimited words, and 4,010,466 bytes of source
text. See [`NOTICE.md`](NOTICE.md) for provenance and [`LICENSE`](LICENSE) for
the Apache-2.0 license.

## Why Qdrant documentation

The corpus was selected against five criteria, in priority order:

1. **At least 100 source documents:** 3,067 documents leave substantial
   headroom for source-level `hit@5`; five random sources cover about 0.16% of
   this corpus, rather than about 56% of the nine-document seed fixture.
2. **Genuine internal redundancy:** collections, payloads, filtering,
   quantization, deployment, and search behavior recur across concepts,
   reference material, tutorials, cloud guides, and troubleshooting pages.
   This supplies natural distractors and near misses instead of invented ones.
3. **Permissive license:** Apache License 2.0, with attribution and the license
   text committed alongside the snapshot.
4. **Structured text:** the sources are Markdown with real heading trees. They
   exercise heading-path prefixes while retaining front matter, tables, code
   blocks, and imperfect links.
5. **Small clone footprint:** the source text is about 4.0 MB, below the 10 MB
   target. No Markdown file exceeds 200 KB.

## Why `data/raw/sample/` remains

The two corpora have different jobs. `data/raw/sample/` remains the tiny,
deterministic offline fixture used by CI and fast plumbing tests. It is not a
quality benchmark: with only nine source documents, random retrieval of five
distinct sources hits a labelled source about 56% of the time. `data/corpus/`
is the larger measurement corpus used with `data/eval/golden-corpus.jsonl` to
compare retrieval strategies without that saturation.

Ingest the measurement corpus with an explicit collection so it cannot replace
another run's collection:

```powershell
python -m production_rag.ingest --source data/corpus --embedder fake --collection prag_corpus_v2 --dry-run
python -m production_rag.ingest --source data/corpus --embedder fake --collection prag_corpus_v2 --recreate-collection
```

## Ingestion validation

Validated on 2026-08-11 with the commands above. The dry run discovered 3,048
non-empty ingestible documents and produced 8,003 chunks. Chunk text length in
characters was p50 544, p95 795, p99 964, and maximum 1,424 (minimum 120,
mean 522.8). The real fake-embedder ingest wrote all 8,003 points to
`prag_corpus_v2` with BM25 sparse vectors enabled.

The real corpus exposed two chunker behaviors that are recorded, not repaired
here:

- 20 Markdown files become empty after front matter removal, and 239 additional
  non-empty files produce zero chunks because all fragments are shorter than
  the 120-character minimum. These are predominantly generated API snippets.
- Although configured `chunk_size` is 800 characters, 5% of chunks approach
  that ceiling and some exceed it; the maximum is 1,424. This indicates that
  overlap/merge behavior can violate the configured maximum on real content.

No single source file produced one enormous unsplit chunk: the largest source,
`qdrant/private-cloud/api-reference.md`, was divided into 121 chunks.
