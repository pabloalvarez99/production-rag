"""One-shot generator for free-path goldens + difficulty ranks (season Month 1)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

docs = {
    "intro": "sample/00-intro.md",
    "hybrid": "sample/01-hybrid-search.md",
    "rerank": "sample/02-reranking.md",
    "cite": "sample/03-citations-and-grounding.md",
    "eval": "sample/04-evaluation-ragas.md",
    "chunk": "sample/05-chunking-pitfalls.md",
    "qdrant": "sample/06-qdrant-payloads.md",
    "guard": "sample/07-abstention-and-guardrails.md",
    "bm25": "sample/08-bm25-vs-dense.md",
}

items: list[dict[object, object]] = []
ranks: dict[str, object] = {
    "baseline": "declared_fixture_not_quality",
    "r_min": 3,
    "items": {},
}
rank_items: dict[str, dict[str, object]] = ranks["items"]  # type: ignore[assignment]

answerable_qs = [
    (
        "fp-ans-001",
        "Why does a system that answers from retrieved passages behave differently from one answering from model weights?",
        docs["intro"],
        2,
        "Baseline paraphrase case.",
    ),
    (
        "fp-ans-002",
        "What are the two stages of a RAG system and which one costs money per document?",
        docs["intro"],
        3,
        "Ingest vs query distinction.",
    ),
    (
        "fp-ans-003",
        "Why does reciprocal rank fusion avoid blending dense and sparse raw scores?",
        docs["hybrid"],
        4,
        "Fusion rationale, not formula only.",
    ),
    (
        "fp-ans-004",
        "What blind spot does dense retrieval have on rare literal tokens?",
        docs["hybrid"],
        2,
        "Dense failure mode.",
    ),
    (
        "fp-ans-005",
        "What blind spot does sparse BM25 have when vocabularies do not overlap?",
        docs["hybrid"],
        5,
        "Sparse failure mode.",
    ),
    (
        "fp-ans-006",
        "If the reranking service times out, should the request fail hard?",
        docs["rerank"],
        3,
        "Fail-open behaviour.",
    ),
    (
        "fp-ans-007",
        "What is the difference between a bi-encoder and a cross-encoder?",
        docs["rerank"],
        2,
        "Exact-token flavoured definition.",
    ),
    (
        "fp-ans-008",
        "Why should the model emit bracketed ordinals instead of chunk ids in citations?",
        docs["cite"],
        4,
        "Citation design.",
    ),
    (
        "fp-ans-009",
        "What failure does faithfulness guard against in grounded generation?",
        docs["intro"],
        6,
        "Faithfulness concept in intro.",
    ),
    (
        "fp-ans-010",
        "Why are most quality problems retrieval problems wearing a generation costume?",
        docs["intro"],
        3,
        "Retrieval-first measurement.",
    ),
    (
        "fp-ans-011",
        "What does overlap never cross in the chunker design, and why?",
        docs["chunk"],
        4,
        "Heading boundary rule.",
    ),
    (
        "fp-ans-012",
        "Why is splitting only on character count a citation risk?",
        docs["chunk"],
        5,
        "Chunking pitfall.",
    ),
    (
        "fp-ans-013",
        "What payload fields make a citation checkable without reopening the store?",
        docs["qdrant"],
        3,
        "Payload design.",
    ),
    (
        "fp-ans-014",
        "When should a RAG system refuse rather than answer fluently?",
        docs["guard"],
        2,
        "Abstention principle.",
    ),
    (
        "fp-ans-015",
        "Why is a fluent unsupported answer with a citation worse than a miss?",
        docs["intro"],
        7,
        "Review survival of bad answers.",
    ),
    (
        "fp-ans-016",
        "How does hybrid search combine two retrievers over one corpus?",
        docs["hybrid"],
        1,
        "One easy rank-1 allowed in slice.",
    ),
    (
        "fp-ans-017",
        "What metric family measures whether the right document was retrieved?",
        docs["eval"],
        4,
        "Eval framing.",
    ),
    (
        "fp-ans-018",
        "Why must unanswerable items exist in an evaluation set?",
        docs["guard"],
        5,
        "Refusal measurement needs labels.",
    ),
    (
        "fp-ans-019",
        "What does BM25 weight by term rarity across the corpus?",
        docs["bm25"],
        3,
        "IDF intuition.",
    ),
    (
        "fp-ans-020",
        "Why can dense and sparse fail on disjoint inputs?",
        docs["hybrid"],
        6,
        "Hybrid motivation.",
    ),
]
for id_, q, path, rank, notes in answerable_qs:
    items.append(
        {
            "id": id_,
            "question": q,
            "expected_source_paths": [path],
            "answerable": True,
            "category": "answerable",
            "notes": notes,
        }
    )
    rank_items[id_] = {"target_rank": rank, "mode": "sparse_or_fused"}

unans = [
    (
        "fp-un-001",
        "What is the concurrency limit of this production-rag deployment under peak load?",
        "No capacity figure in corpus.",
    ),
    (
        "fp-un-002",
        "How many dollars per month does running hybrid search cost in this repository?",
        "No currency figures.",
    ),
    (
        "fp-un-003",
        "Which exact Qdrant server version string does the sample corpus pin as production?",
        "No version pin in sample docs.",
    ),
    (
        "fp-un-004",
        "Which multilingual embedding model does the sample handbook recommend for Spanish?",
        "No multilingual recommendation.",
    ),
    ("fp-un-005", "Who is the on-call engineer for production-rag this week?", "No personnel data."),
    ("fp-un-006", "What is the SLA percentage for the free-path UI demo?", "No SLA numbers."),
    (
        "fp-un-007",
        "How many GPU hours does a billed baseline run require on this laptop?",
        "No GPU budget.",
    ),
    (
        "fp-un-008",
        "What customer name appears in the sample corpus as a case study?",
        "No customer PII.",
    ),
    (
        "fp-un-009",
        "What is the private API key used by the default compose stack?",
        "Secrets never in corpus.",
    ),
    (
        "fp-un-010",
        "Which Antarctic tournament did hybrid search win according to the docs?",
        "Nonsense; must refuse.",
    ),
    (
        "fp-un-011",
        "What is the stock ticker symbol of reciprocal rank fusion?",
        "Category error; refuse.",
    ),
    (
        "fp-un-012",
        "How many citations must a refused answer carry?",
        "Refusal carries none; not a numeric policy stated for this ask.",
    ),
]
for id_, q, notes in unans:
    items.append(
        {
            "id": id_,
            "question": q,
            "expected_source_paths": [],
            "answerable": False,
            "category": "unanswerable",
            "notes": notes,
        }
    )
    rank_items[id_] = {"target_rank": 99, "mode": "none", "unanswerable": True}

filter_items = [
    (
        "fp-fil-001",
        "Why does hybrid search use reciprocal rank fusion?",
        docs["hybrid"],
        {"tags": "hybrid"},
        3,
    ),
    (
        "fp-fil-002",
        "What fails when dense and sparse are fused by raw score sum?",
        docs["hybrid"],
        {"tags": "hybrid"},
        4,
    ),
    ("fp-fil-003", "How does BM25 weight rare terms?", docs["bm25"], {"tags": "retrieval"}, 2),
    (
        "fp-fil-004",
        "What is the bi-encoder versus cross-encoder distinction?",
        docs["rerank"],
        {"tags": "retrieval"},
        5,
    ),
    (
        "fp-fil-005",
        "Why refuse when passages do not support an answer?",
        docs["guard"],
        {"tags": "rag"},
        3,
    ),
    (
        "fp-fil-006",
        "What makes a citation marker resolvable?",
        docs["cite"],
        {"tags": "rag"},
        4,
    ),
    (
        "fp-fil-007",
        "Why measure retrieval separately from generation?",
        docs["eval"],
        {"tags": "rag"},
        6,
    ),
    (
        "fp-fil-008",
        "What does the intro say retrieval-augmented generation is?",
        docs["intro"],
        {"tags": "intro"},
        2,
    ),
    (
        "fp-fil-009",
        "How do payload indexes relate to filtered search?",
        docs["qdrant"],
        {"tags": "qdrant"},
        5,
    ),
    (
        "fp-fil-010",
        "Why must overlap not cross heading boundaries?",
        docs["chunk"],
        {"tags": "chunking"},
        3,
    ),
]
for id_, q, path, filters, rank in filter_items:
    items.append(
        {
            "id": id_,
            "question": q,
            "expected_source_paths": [path],
            "answerable": True,
            "category": "filter",
            "filters": filters,
            "notes": f"Filter-narrowed case; filters={filters}.",
        }
    )
    rank_items[id_] = {"target_rank": rank, "mode": "filtered_sparse", "filters": filters}

hybrid = [
    (
        "fp-hvd-001",
        "What do the k1 and b parameters control in the BM25 scoring formula?",
        docs["bm25"],
        "k1",
        4,
    ),
    (
        "fp-hvd-002",
        "Where is the full_scan_threshold setting discussed for payload indexes?",
        docs["qdrant"],
        "full_scan_threshold",
        5,
    ),
    (
        "fp-hvd-003",
        "Which document explains bge-reranker-base as a local cross-encoder option?",
        docs["rerank"],
        "bge-reranker-base",
        3,
    ),
    (
        "fp-hvd-004",
        "What token-level method is BM25 an example of versus dense vectors?",
        docs["bm25"],
        "BM25",
        2,
    ),
    (
        "fp-hvd-005",
        "How does reciprocal rank fusion write the rank term 1/(k+rank)?",
        docs["hybrid"],
        "1/(k",
        6,
    ),
    (
        "fp-hvd-006",
        "What heading-aware structure does the chunker preserve as heading_path?",
        docs["chunk"],
        "heading_path",
        4,
    ),
    (
        "fp-hvd-007",
        "Which metric name invalid_markers counts invented citation ordinals?",
        docs["cite"],
        "invalid_markers",
        5,
    ),
    (
        "fp-hvd-008",
        "What does score_threshold do before the model is called?",
        docs["guard"],
        "score_threshold",
        3,
    ),
    (
        "fp-hvd-009",
        "Why is ndcg@k mentioned when graded relevance labels exist?",
        docs["eval"],
        "ndcg",
        7,
    ),
    (
        "fp-hvd-010",
        "What named vector pair does hybrid Qdrant storage use: dense and sparse?",
        docs["qdrant"],
        "named vector",
        2,
    ),
]
for id_, q, path, token, rank in hybrid:
    items.append(
        {
            "id": id_,
            "question": q,
            "expected_source_paths": [path],
            "answerable": True,
            "category": "hybrid-vs-dense",
            "notes": f"Lexical/hybrid pressure; rare token cue: {token}.",
            "rare_token": token,
        }
    )
    rank_items[id_] = {"target_rank": rank, "mode": "sparse_preferred", "rare_token": token}

assert len(items) >= 50, len(items)
by: dict[str, list[int]] = defaultdict(list)
for it in items:
    by[str(it["category"])].append(int(rank_items[str(it["id"])]["target_rank"]))
for cat, rs in by.items():
    n1 = sum(1 for r in rs if r == 1)
    assert n1 < len(rs), (cat, rs)
    assert n1 / len(rs) < 0.5, (cat, n1, len(rs))

out = ROOT / "data" / "eval" / "golden-free-path.jsonl"
out.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in items) + "\n", encoding="utf-8")
(ROOT / "data" / "eval" / "difficulty-ranks.json").write_text(
    json.dumps(ranks, indent=2) + "\n", encoding="utf-8"
)
print("items", len(items), "by", {k: len(v) for k, v in by.items()})
print("rank1 fractions", {k: sum(1 for r in v if r == 1) / len(v) for k, v in by.items()})
