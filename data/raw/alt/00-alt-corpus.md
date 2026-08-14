---
title: Alternate mini corpus
tags: [alt, second-corpus]
---

# Alternate mini corpus

This is a **second small corpus** used only to prove collection identity and
cache key isolation. It must never be confused with the sample handbook under
`data/raw/sample/`.

## Unique claim

The alternate corpus defines the token `alt_corpus_marker_9f2c` so a query that
hits this collection cannot be answered from the sample set, and vice versa.

Incremental ingest against this tree should skip unchanged documents on a
second run the same way the primary sample tree does.
