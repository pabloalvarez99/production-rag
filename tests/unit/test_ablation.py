"""Offline tests for the four-way retrieval ablation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from production_rag.config_loader import RerankConfig, RetrievalConfig
from production_rag.evals import ablation
from production_rag.evals.ablation import (
    ABLATION_MODES,
    MODE_HYBRID_RERANK,
    evaluate_ablation,
)
from production_rag.evals.source_hit import GoldenCase, SourceHitReport
from production_rag.ingest.models import Chunk, Document
from production_rag.retrieval.embeddings import FakeEmbeddingProvider
from production_rag.retrieval.hybrid import MODE_HYBRID, Retriever
from production_rag.retrieval.rerank import FakeReranker
from production_rag.retrieval.sparse import Bm25Encoder
from production_rag.retrieval.store import InMemoryVectorStore

CORPUS = {
    "sample/hybrid.md": "Reciprocal rank fusion combines dense and sparse result ranks.",
    "sample/rerank.md": "A cross-encoder reranks query and passage pairs together.",
    "sample/chunks.md": "Markdown headings preserve useful chunk boundaries.",
}


def _retrievers() -> tuple[Retriever, Retriever]:
    embedder = FakeEmbeddingProvider()
    chunks = []
    for index, (path, body) in enumerate(CORPUS.items()):
        document = Document(source_path=path, text=body, title=path, source="sample")
        chunks.append(Chunk.build(document=document, chunk_index=index, text=body, embed_text=body))

    texts = [chunk.embed_text for chunk in chunks]
    encoder = Bm25Encoder()
    encoder.fit(texts)
    store = InMemoryVectorStore(collection="ablation_collection")
    store.ensure_collection(vector_size=embedder.dimensions, with_sparse=True)
    store.upsert_chunks(
        chunks,
        embedder.embed_documents(texts),
        ingest_run_id="run-ablation",
        embedded_model=embedder.model,
        sparse_vectors=encoder.encode_documents(texts),
    )
    retrieval_config = RetrievalConfig(dense_top_k=3, sparse_top_k=3, top_k=2)
    base = Retriever(store=store, embedder=embedder, config=retrieval_config)
    reranked = Retriever(
        store=store,
        embedder=embedder,
        config=retrieval_config,
        reranker=FakeReranker(),
        rerank_config=RerankConfig(input_top_k=3, top_k=2, fail_open=False),
    )
    return base, reranked


def _cases() -> list[GoldenCase]:
    return [
        GoldenCase(
            id="q1",
            question="reciprocal rank fusion",
            expected_source_paths=("sample/hybrid.md",),
            category="conceptual",
        ),
        GoldenCase(
            id="q2",
            question="cross-encoder passage pairs",
            expected_source_paths=("sample/rerank.md",),
            category="exact-token",
        ),
        GoldenCase(
            id="q3",
            question="an answer unavailable in this corpus",
            expected_source_paths=(),
            category="unanswerable",
        ),
    ]


class TestAblation:
    def test_scores_every_mode_with_the_shared_source_hit_metric(self) -> None:
        base, reranked = _retrievers()
        report = evaluate_ablation(
            retriever=base,
            reranked_retriever=reranked,
            cases=_cases(),
            k=2,
            golden_path="golden.jsonl",
        )

        assert tuple(report.reports) == ABLATION_MODES
        assert all(result.k == 2 for result in report.reports.values())
        assert all(len(result.scored_cases) == 2 for result in report.reports.values())
        assert report.reports[MODE_HYBRID_RERANK].mode == MODE_HYBRID

    def test_delta_is_reranked_minus_hybrid_and_json_has_per_mode_hit_at_k(self) -> None:
        base, reranked = _retrievers()
        report = evaluate_ablation(
            retriever=base,
            reranked_retriever=reranked,
            cases=_cases(),
            k=1,
        )
        summary = report.to_summary()

        expected_delta = (
            report.reports[MODE_HYBRID_RERANK].score - report.reports[MODE_HYBRID].score
        )
        assert summary["delta_hybrid_vs_hybrid_rerank"] == pytest.approx(expected_delta)
        assert tuple(summary["modes"]) == ABLATION_MODES
        assert all("hit_at_k" in mode for mode in summary["modes"].values())

    def test_orchestration_calls_source_hit_instead_of_forking_the_metric(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        base, reranked = _retrievers()
        calls: list[tuple[Retriever, str | None]] = []

        def score(
            *,
            retriever: Retriever,
            cases: Any,
            k: int,
            mode: str | None,
        ) -> SourceHitReport:
            calls.append((retriever, mode))
            return SourceHitReport(
                mode=mode or retriever.config.mode,
                k=k,
                collection=retriever.store.collection,
                embedded_model=retriever.embedder.model,
                golden_path="",
            )

        monkeypatch.setattr(ablation, "evaluate_source_hit", score)
        evaluate_ablation(
            retriever=base,
            reranked_retriever=reranked,
            cases=[],
            k=3,
        )

        assert [mode for _, mode in calls] == ["dense", "sparse", "hybrid", "hybrid"]
        assert [retriever for retriever, _ in calls[:3]] == [base, base, base]
        assert calls[3][0] is reranked


class TestCli:
    @pytest.fixture
    def store(self, monkeypatch: pytest.MonkeyPatch) -> InMemoryVectorStore:
        base, _ = _retrievers()
        store = base.store
        assert isinstance(store, InMemoryVectorStore)
        monkeypatch.setattr(ablation, "resolve_searchable_store", lambda **_: store)
        return store

    @staticmethod
    def _golden(tmp_path: Path) -> Path:
        path = tmp_path / "golden.jsonl"
        path.write_text(
            json.dumps(
                {
                    "id": "q1",
                    "question": "reciprocal rank fusion",
                    "expected_source_paths": ["sample/hybrid.md"],
                    "category": "exact-token",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    @staticmethod
    def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict[str, Any]]:
        code = ablation.main(list(argv))
        output = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
        return code, json.loads(output[-1])

    def test_cli_emits_one_four_mode_json_report(
        self,
        store: InMemoryVectorStore,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        golden = self._golden(tmp_path)
        code, payload = self._run(capsys, "--golden", str(golden), "--k", "2")

        assert store.collection == "ablation_collection"
        assert code == 0
        assert payload["ok"] is True
        assert payload["golden_path"] == str(golden)
        assert set(payload["modes"]) == set(ABLATION_MODES)
        assert isinstance(payload["delta_hybrid_vs_hybrid_rerank"], float)

    def test_missing_golden_is_a_usage_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, payload = self._run(capsys, "--golden", str(tmp_path / "missing.jsonl"))
        assert code == 2
        assert payload["error_type"] == "EvalError"

    def test_defaults_and_help_are_offline_and_ascii(self) -> None:
        args = ablation.build_parser().parse_args([])
        assert args.embedder == "fake"
        assert args.k == 5
        assert ablation.build_parser().format_help().encode("ascii", errors="strict")
