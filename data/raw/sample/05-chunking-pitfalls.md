---
title: Chunking pitfalls — the retrieval decision that hides as preprocessing
tags: [rag, chunking, ingest, preprocessing]
---

# Chunking pitfalls — the retrieval decision that hides as preprocessing

Chunking looks like a preprocessing detail and behaves like the most consequential
retrieval parameter in the system. Every embedding, every BM25 statistic, and
every citation is computed over chunks, so a chunking change invalidates all
three at once.

## Size trades two failure modes

Chunks that are too small lose the context that makes them interpretable. A
three-sentence chunk that says "this is not supported in the free tier" is
useless without the heading that says what *this* is.

Chunks that are too large dilute the embedding. A single vector has to represent
the whole chunk, so a passage covering three topics ends up in the average of
three regions of the space and matches none of them strongly. Sparse retrieval
degrades differently but just as reliably: BM25 length normalisation penalises
the long chunk, so it loses to a short chunk that mentions the term once.

There is no universal correct size. There is a size that is correct for a corpus
and a query distribution, and the only way to find it is to move it and re-run
the retrieval metrics.

## Split on structure before length

Splitting on a fixed character count cuts sentences in half and separates a
heading from the paragraph it introduces. Recursive splitting tries structural
boundaries in order — headings, then paragraphs, then sentences — and falls back
to a hard character cut only when a single paragraph exceeds the budget.

The result is chunks that are semantically coherent rather than merely uniform
in length, and uniformity was never the goal.

## Overlap exists for exactly one reason

A sentence that straddles a chunk boundary is retrievable from neither side: the
first chunk has its opening, the second has its close, and neither carries the
whole claim. Overlap of roughly 10–20% of the chunk size covers the straddle.

Larger overlap is not safer. It inflates the index, duplicates near-identical
text across chunks, and produces result lists where the top three hits are three
views of the same paragraph — which shrinks the effective diversity of the
context window without anyone noticing.

## Heading context is nearly free

Prefixing each chunk with its document title and heading path before embedding
costs a few tokens and measurably helps short queries, because it puts the words
the user is likely to type into the text that gets embedded. The chunk body
stored in the payload can stay clean; only the embedded text carries the prefix.

## Tables, code, and lists break naive splitters

A Markdown table split across two chunks yields a header with no rows and rows
with no header — both unreadable and both indexed. Code blocks split mid-function
lose the signature that made them findable. Numbered lists split mid-item lose
the enumeration.

Treat these as atomic where possible: keep a fenced code block or a table in one
chunk even when it exceeds the target size, or drop it into its own chunk with
the enclosing heading attached.

## Chunk ids are coupled to chunk config

`chunk_id` is derived from the document path, the chunk index, and a hash of the
chunk text. Changing `chunk_size` or `chunk_overlap` therefore changes every id
in the corpus.

Two things break at that moment, and both break silently. Hand-labelled eval
data that names relevant chunk ids stops matching anything, so recall reads as
zero for reasons that have nothing to do with retrieval. Stored citations in
previously produced answers stop resolving.

Re-chunking is a re-labelling event. Plan it as one.

## Minimum chunk size

Fragments below roughly 100 characters — a stray heading, a table caption, a
single-line list item — carry no retrievable signal and add noise to both
indexes. Drop them at ingest, and log how many were dropped: a sudden jump in
that count usually means an upstream document format changed.
