# Hybrid Search: Dense and Sparse Retrieval

Dense retrieval embeds text into vectors and matches by semantic similarity.
It handles paraphrases well: a question about "fixing login failures" can
match a chunk about "resolving authentication errors" even with no words in
common. Its weakness is exact-token matching — identifiers, error codes, and
unusual product names may embed poorly and rank below vaguer matches.

Sparse retrieval is the lexical complement. Methods like BM25 score documents
by term frequency and rarity, so rare exact tokens dominate the ranking.
Sparse methods excel at keywords but cannot bridge vocabulary gaps.

Hybrid search fuses both rankings, commonly with reciprocal rank fusion, so a
chunk that scores well on either signal surfaces in the final list. In this
project both vector kinds live in one Qdrant collection and fusion runs
server-side, keeping the query path a single round trip. Hybrid retrieval is
the default in this repository because production corpora almost always mix
prose with identifiers.
