---
title: BM25 versus dense embeddings — the mechanics behind the trade-off
tags: [bm25, sparse, dense, embeddings, tokenisation]
---

# BM25 versus dense embeddings — the mechanics behind the trade-off

"Dense wins on paraphrase, sparse wins on rare tokens" is the correct summary
and it explains nothing. The behaviour falls out of how each method computes a
score, and knowing the mechanism is what lets you predict which one a given
query will break.

## What BM25 actually computes

BM25 scores a document by summing a contribution per query term:

```
score(D, Q) = Σ  IDF(q) · ( f(q,D) · (k1 + 1) )
              q               ─────────────────────────────────
                              f(q,D) + k1 · (1 - b + b · |D|/avgdl)
```

Three ideas, each doing one job:

- **`IDF(q)`** — a term appearing in few documents is worth more. This is the
  whole reason an error code outranks the word "error".
- **`k1`** — term frequency saturation, typically `1.2`. The tenth occurrence of
  a term adds much less than the second, so a document cannot win by repetition.
- **`b`** — length normalisation, typically `0.75`. At `b = 1` a document is
  fully penalised for being long; at `b = 0` length is ignored entirely.

There is no training and no model. The scores are computed from corpus counts,
which means BM25 works on day one over a corpus it has never seen — and also
that its statistics shift as the corpus grows.

## What a dense embedding computes

An embedding model maps text to a fixed-length vector, and similarity is cosine
distance between vectors. The mapping is learned, so the notion of "similar" is
whatever the training data taught it. Two passages with no shared vocabulary can
sit close together, which is the entire value proposition.

The learned mapping is also the limitation. A token that was rare or absent in
training — an internal product code, a customer-specific acronym, a function
name from a private codebase — has no meaningful position in the space. Its
embedding is close to arbitrary, and so is any ranking that depends on it.

## Tokenisation decides more than people expect

BM25 matches surface forms after tokenisation, so the tokeniser is part of the
retrieval quality story. Lowercasing merges `Error` and `error` and is almost
always right. Stemming merges `retrieving` and `retrieval` and occasionally
destroys a distinction that mattered. Splitting on punctuation turns
`text-embedding-3-small` into four tokens, none of them selective, and a query
for the exact model name stops being an exact-token query at all.

Stopword lists are the quiet failure. A list built for English applied to a
non-English corpus removes the wrong words, and recall drops with no error
anywhere. If the corpus is not English, set the list explicitly rather than
accepting a default.

## Predicting which branch wins

| Query shape | Winner | Why |
|---|---|---|
| "how do I make search handle typos" | dense | no shared vocabulary with the passage |
| `QDRANT__SERVICE__GRPC_PORT` | sparse | high IDF, zero semantic content |
| "what does k1 control" | either | the term is both literal and contextual |
| a whole error message pasted in | sparse | several rare tokens co-occurring |
| a question rephrased from a heading | dense | paraphrase is exactly the trained case |

The two failure sets barely intersect, which is what makes running both cheaper
than choosing.

## Cost profiles differ, not just quality

Dense retrieval front-loads its cost: every chunk is embedded once at ingest,
paid in provider tokens, and query time is one embedding call plus an approximate
nearest-neighbour walk. BM25 has no ingest model cost at all, but its index
carries corpus statistics that shift with every document added.

That asymmetry matters operationally. Re-chunking a corpus means re-embedding
it, which costs money proportional to corpus size. Re-chunking the sparse index
costs only CPU.

## Where each degrades as the corpus grows

Dense retrieval degrades gradually: more points means more near-neighbours at
similar distance, so the top of the list gets more crowded and ordering matters
more. Sparse retrieval degrades differently — IDF values shift, so a term that
was selective in a 1,000-document corpus can become common in a 100,000-document
one and stop discriminating.

Both are arguments for measuring retrieval on the corpus you actually have,
rather than transferring numbers from a benchmark.
