---
title: What retrieval-augmented generation actually is
tags: [rag, intro]
---

# What retrieval-augmented generation actually is

Retrieval-augmented generation (RAG) answers a question by first *finding*
relevant text in a corpus and then *conditioning* a language model on that text.
The model is not asked to recall a fact from its weights; it is asked to read a
short set of passages and answer from them.

This distinction matters more than it first appears. A model answering from
weights has no way to tell you where an answer came from, no way to be corrected
without retraining, and no way to signal that it does not know. A model
answering from retrieved passages can cite, can be updated by editing a
document, and can refuse when the passages do not contain the answer.

## The two stages

**Ingest** runs offline. Documents are split into chunks, each chunk is turned
into one or more vectors, and the vectors plus the original text are written to
a vector store. This is the only stage that costs money per document.

**Query** runs per request. The question is turned into a query, the store
returns candidate chunks, and a language model reads them and produces an
answer with citations.

## Where RAG systems actually fail

Most quality problems are retrieval problems wearing a generation costume. If
the supporting passage never reaches the model, no amount of prompt engineering
recovers it — the information simply is not in the context window. This is why
retrieval is measured separately and first.

The second most common failure is the opposite: the passage was present and the
model produced a confident claim that the passage does not support. This is
worse than a wrong retrieval, because a fluent unsupported answer with a
citation attached survives review. Faithfulness — every claim traceable to a
cited chunk — is the metric that guards against it.

The third is subtler: the system answers a question it should have refused.
A corpus does not contain every answer, and a system that never says "I could
not find support for that" is not more capable, only less honest.

## Chunking is a retrieval decision

Chunk size trades two failure modes against each other. Chunks that are too
small lose the context that makes them interpretable; chunks that are too large
dilute the embedding, so a passage about three topics matches none of them
strongly. Splitting on structural boundaries — headings first, then paragraphs —
keeps chunks semantically coherent rather than merely uniform in length.

Overlap exists for one reason: a sentence that straddles a chunk boundary would
otherwise be retrievable from neither side.
