# ADR 0009 — Hard chunk bounds and preservation of short documents

- **Status:** Accepted
- **Date:** 2026-08-11
- **Deciders:** production-rag maintainers
- **Supersedes:** —
- **Relates to:** [ADR 0003](0003-eval-strategy.md) (the fixture whose
  retrieval baseline must remain attributable to a chunking change)

## Context

The first real-corpus measurement exposed two independent defects in the
recursive chunker. With `chunk_size: 800`, one chunk reached 1,424 characters:
a 77% overshoot. Separately, the 120-character minimum removed every chunk from
239 non-empty source documents. Together with 20 documents empty after front
matter, the reported corpus had 259 of 3,067 sources absent from the index.

Independent reproduction against the corpus snapshot available to this change
found the same material behaviour: 8,003 chunks with character lengths
min/p50/p95/max 120/544/795/1,424, 239 non-empty documents producing no chunks,
and 20 empty documents. That snapshot contains 3,068 ingestible documents
(3,048 non-empty plus 20 empty), one more than the reported 3,067; the snapshot
difference is recorded rather than normalised away.

The overshoot was not structure preservation. `_split_recursive` already
hard-cut every indivisible piece at `chunk_size`. `_merge_pieces` then carried
up to 120 characters of overlap into the next chunk and appended an
already-bounded piece without checking whether the combination still fit. The
implementation and its docstrings already promised a maximum, so this was a
counting bug rather than a badly named target.

The minimum had a different scope error. It is useful for fragments created by
splitting a longer document, but it was applied after heading segmentation to
every piece. A short document was therefore treated as one disposable fragment
instead of one complete source.

## Decision

`chunk_size` remains the public name and is a hard character ceiling for chunk
text, including overlap. When the next source piece leaves insufficient room,
the merger trims repeated overlap first. Source text is never trimmed for the
sake of overlap, and an indivisible run is still hard-cut. Tests pin the ceiling
for ordinary prose, a full-budget piece after overlap, and a code fence longer
than the ceiling.

`min_chunk_chars` applies only to fragments produced by splitting a document.
A non-empty document whose complete post-front-matter text is shorter than the
floor emits exactly one chunk containing that complete text. A genuinely empty
document emits none. Heading-path metadata and embedding prefixes remain
present for normally sectioned documents; a preserved whole short document has
no single honest heading path, so its title is the only synthetic prefix and
its Markdown headings remain in the chunk text.

After the change, the measured real-corpus snapshot produces 8,243 chunks with
lengths 24/531/794/800 and zero non-empty documents without chunks. The same 20
empty documents remain absent. Thus 239 policy-dropped documents are recovered;
the remaining 20 have no retrievable body.

## Fixture and evaluation impact

The nine-document `data/raw/sample/` fixture does not change. Before and after,
it produces 66 chunks with lengths min/p50/p95/max 184/449/714/781. No fixture
chunk exercised either defect, so chunk ids and indexed content remain stable.

Tier 1 is unchanged: source hit@5 0.9231, source recall@5 0.8846, MRR 0.5833,
and nDCG@5 0.6383 over 13 scored cases (17 total, four unanswerable). No
threshold was changed to obtain those numbers.

## Consequences

**Positive**

- Chunk text now has a provider-facing hard bound instead of an accidental
  `chunk_size + overlap` envelope.
- Short but potentially answer-bearing sources are retrievable.
- Empty and policy-dropped documents are counted separately in the corpus
  measurement, so the measurable ceiling is visible.
- The offline fixture and its tier-1 baseline remain attributable to the same
  indexed content.

**Negative**

- Trimming overlap before a full-budget piece can cut repeated context in the
  middle of a word. Overlap is redundant context, so preserving source text and
  the hard bound takes priority.
- A preserved whole short Markdown document cannot claim one heading path when
  it contains multiple sections; its headings remain in the text instead.
- Chunk identities change across the real corpus wherever the old merger
  overshot, so a real collection requires re-ingestion before comparing
  retrieval results.

**Unresolved**

- The ceiling is character-based, not tokenizer-specific. Provider token
  limits still require validation when a production embedding model is chosen.
- Synthetic title and heading prefixes are outside the stored chunk-text
  ceiling. A future provider-specific embedding-input budget should bound the
  complete prefixed input without weakening the stable payload-text contract.
- The available corpus snapshot has one more ingestible document than the
  original 3,067-document report; corpus provenance should be versioned before
  its metrics become a release gate.
