---
title: Hybrid search — dense, sparse, and rank fusion
tags: [rag, retrieval, hybrid, bm25]
---

# Hybrid search — dense, sparse, and rank fusion

Hybrid search runs two different retrievers over the same corpus and merges
their results. It exists because dense and sparse retrieval fail on disjoint
inputs, and running both is cheaper than choosing which failure to accept.

## Dense retrieval

An embedding model maps text into a vector space where semantic similarity is
geometric proximity. A question and a passage that share no words at all can
still land close together, which is exactly what makes dense retrieval robust to
paraphrase.

Its blind spot is rare literal tokens. Error codes, product SKUs, function
names, and internal acronyms were never meaningfully represented during
training, so their embeddings carry little signal and they rank close to
arbitrary.

## Sparse retrieval

BM25 scores a document by how often the query's terms appear in it, damped by
document length and weighted by how rare each term is across the corpus. Rare
terms therefore dominate the score — which is precisely the case dense
retrieval handles worst.

Its blind spot is the mirror image: if the question and the passage share no
surface vocabulary, BM25 scores zero no matter how obviously related they are to
a human reader.

## Fusion

The two retrievers produce scores on incomparable scales. Cosine similarity sits
roughly in [-1, 1] and clusters tightly; BM25 is unbounded and corpus-dependent.
Blending them by weighted sum requires normalising both, and the normalisation
constants drift as the corpus grows.

Reciprocal rank fusion sidesteps the problem by discarding the scores entirely
and using only rank position:

```
score(d) = Σ  1 / (k + rank_i(d))
           i
```

where `i` ranges over the retrievers and `k` is a smoothing constant, commonly
60. A document ranked first by either retriever gets a large contribution; a
document ranked well by both gets more. Because only ranks are used, no
calibration is needed and none can drift.

The cost is that magnitude is lost. A chunk that is overwhelmingly the best
match contributes exactly as much as a merely-good one at the same rank.

## Over-retrieve before you fuse

Fusion can only reorder what it was given. Pulling 40 candidates from each
branch to emit a final 12 is deliberate: a chunk that ranks 30th in one branch
and 4th in the other is exactly the case hybrid search exists to catch, and a
top-10 pull from each branch would have thrown it away before fusion ever saw
it.

## Reranking

A cross-encoder reads the query and a candidate passage *together* and scores
their relevance directly. It is far more accurate than either retriever and far
too slow to run over a whole corpus — which is why it runs last, over a few
dozen candidates, and why it is capped by whatever recall the retrieval stage
achieved.
