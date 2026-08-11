---
title: Qdrant payloads, named vectors, and filterable metadata
tags: [qdrant, payload, vectors, filtering]
---

# Qdrant payloads, named vectors, and filterable metadata

Qdrant stores a point as three things: an id, one or more vectors, and a payload
of arbitrary JSON. Using the payload as the system of record for chunk text and
provenance is what lets a RAG service run on one datastore instead of two.

## Named vectors put dense and sparse on the same point

A collection can declare several vectors per point, each with its own name,
dimensionality, and distance metric. A hybrid setup declares two: a `dense`
vector of the embedding model's dimensionality under cosine distance, and a
`sparse` vector holding term weights.

Both belong to the same point, so both branches of a hybrid query return the
same ids and the same payload. Fusion is then a matter of merging two ranked id
lists, with no join against a second store and no possibility of the two sides
disagreeing about what a chunk says.

Sparse vectors are stored as index/value pairs rather than as a fixed-width
array, so a 30,000-term vocabulary costs only as many entries as the chunk has
distinct terms. The `idf` modifier tells Qdrant to apply inverse document
frequency weighting at query time using corpus statistics it maintains itself.

## Payload as the source of citable provenance

The payload carries everything the answer needs to be checkable:

| Field | Why it is there |
|---|---|
| `doc_id` | groups chunks belonging to one document |
| `chunk_id` | the citation target; stable across re-ingest |
| `source_path` | path relative to the corpus root, shown to the user |
| `title` | document title, from front matter or the first H1 |
| `heading_path` | the heading chain enclosing the chunk |
| `chunk_index` | position within the document, for ordering |
| `text` | the chunk body, returned so no second fetch is needed |

Returning `text` in the payload is the decision that removes an entire tier from
the architecture. The alternative — store ids in Qdrant, fetch text from a
relational database — adds a round trip on the hot path and a consistency
problem on the ingest path.

Keep the returned field list tight. Every payload field handed to generation is
tokens in the prompt, and a field nobody reads is a field paid for on every
request.

## Payload indexes make filters cheap

By default a filtered search scans payloads. Declaring a keyword index on a
field — `doc_id`, `source`, `tags` — makes filtering on it use an index instead,
which matters as soon as filters become selective.

The pattern that catches people out is a highly selective filter combined with
HNSW search: the graph traversal is guided by the vector, not the filter, so a
filter that matches 0.1% of points can force a great deal of traversal before
enough matching neighbours are found. Qdrant handles this by falling back to a
brute-force scan below `full_scan_threshold`, which is faster than an HNSW walk
on a small candidate set.

## Filters as an allowlist

The query API should accept filters only on fields declared in configuration,
never on arbitrary payload keys supplied by the caller. An open filter surface
lets a caller probe payload structure and pay for expensive unindexed scans, and
it makes the indexed-field set impossible to reason about.

## Writes, consistency, and re-ingest

Upserts accept a `wait` flag. Setting it makes the call return only once the
operation is applied, which is what stops a smoke test immediately after ingest
from reading a half-built index. It costs latency on the ingest path, which is
offline and does not care.

Incremental ingest compares a content hash per chunk and skips embedding for
chunks whose text is unchanged. Embedding is the only step that costs money per
document, so incremental is the sensible default rather than an optimisation.

## On-disk payload

`on_disk_payload: true` keeps payload bodies out of RAM and reads them only for
points that survive to the result set. With chunk text stored inline this is
usually the difference between a collection that fits in memory and one that
does not, and the extra read affects only the handful of points actually
returned.
