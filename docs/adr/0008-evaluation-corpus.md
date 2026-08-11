# ADR 0008 — Separate the CI fixture from the retrieval measurement corpus

- **Status:** Accepted
- **Date:** 2026-08-11
- **Deciders:** production-rag maintainers
- **Supersedes:** —
- **Relates to:** [ADR 0003](0003-eval-strategy.md) (the source-level metrics
  whose discriminating power depends on corpus size), [ADR 0001](0001-hybrid-qdrant.md)
  (the dense/sparse claim the adversarial slices are designed to compare)

## Context

`data/raw/sample/` contains nine documents. It is useful as a deterministic
offline fixture, but it cannot support a discriminating source-level retrieval
score. A retriever returning five distinct random sources from nine has a
<!-- provenance-allow: historical-measurement: random-baseline calculation recorded when ADR-0008 was accepted -->
`5 / 9`, or approximately **56%**, chance of including the labelled source.
`hit@5` therefore starts more than halfway to perfect before the retriever has
used the query at all. Hybrid, sparse, and dense runs have too little headroom
to separate meaningfully.

Fake embeddings are a second reason historical numbers over that fixture make
no quality claim. Replacing them with paid embeddings would remove only one of
the two defects: it would purchase a more realistic representation while the
nine-document random baseline still saturated the source metric. Corpus must
come before spend.

## Decision

Vendor the Qdrant documentation at upstream `qdrant/landing_page` commit
`cc9f98286dd98eca3c5bc57110b50887ca0da446` into `data/corpus/qdrant/`, and
pair it with `data/eval/golden-corpus.jsonl`.

The corpus scores against the selection criteria as follows:

1. **Document count:** 3,067 Markdown documents, far above the binding minimum
<!-- provenance-allow: historical-measurement: corpus-scale random baseline recorded when ADR-0008 was accepted -->
of 100. A random five-source result covers about 0.16% of this corpus.
2. **Internal redundancy:** concepts such as payloads, filters, collections,
   quantization, hybrid search, and deployment recur naturally in reference,
   tutorial, cloud, operations, and troubleshooting pages.
3. **License:** Apache License 2.0, with a verbatim license and provenance notice
   committed beside the corpus.
4. **Structure:** upstream Markdown headings are retained, along with front
   matter, code blocks, tables, short generated snippets, and imperfect links.
5. **Size:** 4,010,466 bytes and 416,102 whitespace-delimited words, below the
   approximate 10 MB target.

Both corpora remain intentionally. `data/raw/sample/` is the fast offline CI
fixture for deterministic plumbing and regression tests. `data/corpus/` is the
measurement corpus for retrieval experiments. They are two corpora with two
purposes, not competing versions of the same dataset.

The new golden set contains ten cases in each of six adversarial slices:

- `lexical_only` isolates rare exact tokens where sparse retrieval should win.
- `paraphrase_only` removes target vocabulary where dense retrieval should win.
- `multi_source` requires evidence from two or more documents.
- `distractor` supplies a surface-close document that does not contain the answer.
- `near_miss_unanswerable` asks for a specific fact absent from an otherwise
  relevant corpus and measures abstention under plausible evidence.
- `deep_rank` targets narrow passages expected below rank three before reranking
  and measures whether the cross-encoder earns its latency.

## Consequences

**Positive**

- Source-level `hit@5` now has enough headroom to distinguish retrieval modes.
- Natural redundancy makes distractor and near-miss cases properties of the
  corpus rather than inventions of the evaluator.
- Per-slice results can reveal which retrieval stage helped instead of hiding
  all behavior in one aggregate.
- The messy upstream Markdown exercises the actual chunker assumptions.

**Negative**

- The measurement ingest is materially slower than the nine-document fixture.
- Upstream generated snippets expose many below-minimum documents, and some
  produced chunks exceed the configured 800-character size. These are recorded
  corpus findings; this decision does not change the chunker.
- Source-level labels still cannot prove passage-level relevance.

**Neutral / follow-ups**

- No real-provider embedding or hosted judge run happened in this wave. The
  fake-embedder ingest proves walking, chunking, sparse indexing, and storage
  only; **no retrieval-quality claim follows from this ADR**.
- The first billed baseline remains gated on explicit approval and should use
  this corpus and golden set without tuning labels to its output.
