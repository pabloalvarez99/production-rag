---
title: Reranking — why a second pass beats a better retriever
tags: [rag, rerank, cross-encoder, precision]
---

# Reranking — why a second pass beats a better retriever

Retrieval and ranking look like the same problem and are not. Retrieval has to
touch the whole corpus, so it must be cheap. Ranking only has to order a few
dozen candidates, so it can afford to be expensive. A reranker is the stage that
buys accuracy with compute the retrieval stage cannot spend.

## Bi-encoder versus cross-encoder

A bi-encoder — the model behind dense retrieval — embeds the query and each
passage *independently*. The passage vectors can therefore be computed once, at
ingest time, and reused for every query forever. That precomputation is what
makes vector search fast, and it is also the source of its ceiling: the model
never sees the query and the passage at the same time, so it cannot reason about
how they relate.

A cross-encoder concatenates the query and one candidate passage into a single
input and runs the full attention stack over the pair. It can notice that the
passage answers a different question that merely shares vocabulary, or that the
one sentence that matters sits in a subordinate clause. The price is that
nothing is precomputable: scoring N candidates costs N forward passes at query
time.

## The recall ceiling

A reranker can only reorder what retrieval handed it. If the supporting chunk
was never in the candidate set, the reranker cannot invent it, and no amount of
reranker quality recovers the answer.

This gives the ordering rule for tuning: fix recall first, precision second.
Measure `recall@40` on the fused candidate list, and only once that number is
high does reranker quality become the binding constraint. Teams that tune in the
opposite order spend weeks improving the ordering of a list that never contained
the right passage.

## Feed it more than you keep

The reranker's input window should be much wider than its output. Pulling 40
fused candidates down to 6 is deliberate: rank 30 in the fused list is exactly
where a chunk lands when it was strong in one branch and mediocre in the other,
and that is the chunk a cross-encoder is best at rescuing.

Narrowing the input to save latency defeats the purpose. If the input set is 8
and the output is 6, the reranker is doing almost nothing except adding a
network hop.

## Latency budget

Cross-encoder scoring is roughly linear in the number of candidates. A local
`bge-reranker-base` over 40 short passages costs tens of milliseconds on CPU and
single-digit milliseconds on GPU; a hosted reranking API costs a round trip plus
provider queueing, typically 100–300 ms.

That is a real fraction of a RAG request, but it sits in front of a generation
call measured in seconds, so it rarely dominates the total. Measure it as its
own stage rather than folding it into "retrieval", or a reranker regression
looks like a vector store problem.

## Fail open, not closed

A reranker is a quality improvement, not a correctness requirement. When the
reranker times out or errors, the right behaviour is to fall through to fusion
order and serve a slightly worse answer. Failing the request instead trades a
few points of nDCG for an outage, which is never the better deal.

Log the fallback. A reranker that has been quietly failing open for a week looks
exactly like a reranker that was never helping.
