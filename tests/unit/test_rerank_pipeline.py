"""The rerank stage as wired into the retriever and the CLI.

Everything here runs against ``InMemoryVectorStore`` and the offline
:class:`FakeReranker`: no Qdrant, no model weights, no network.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import pytest
import structlog

from production_rag.config_loader import RerankConfig, RetrievalConfig
from production_rag.ingest.models import Chunk, Document
from production_rag.retrieval import cli
from production_rag.retrieval.embeddings import FakeEmbeddingProvider
from production_rag.retrieval.hybrid import RetrievalHit, Retriever
from production_rag.retrieval.rerank import FakeReranker, RerankError
from production_rag.retrieval.sparse import Bm25Encoder
from production_rag.retrieval.store import InMemoryVectorStore

CORPUS = {
    "sample/00-intro.md": "Production RAG combines retrieval and generation.",
    "sample/01-qdrant.md": "Qdrant stores dense and sparse vectors in one collection.",
    "sample/02-flags.md": "Pass --recreate-collection to rebuild the index.",
    "sample/03-rerank.md": "A cross encoder reranker reads query and passage together.",
}


@pytest.fixture(autouse=True)
def _reset_structlog() -> Iterator[None]:
    yield
    structlog.reset_defaults()


def _store() -> InMemoryVectorStore:
    embedder = FakeEmbeddingProvider()
    chunks = []
    for index, (path, text) in enumerate(CORPUS.items()):
        document = Document(source_path=path, text=text, title=path, source="sample")
        chunks.append(Chunk.build(document=document, chunk_index=index, text=text, embed_text=text))
    texts = [chunk.embed_text for chunk in chunks]
    encoder = Bm25Encoder()
    encoder.fit(texts)
    store = InMemoryVectorStore(collection="test_collection")
    store.ensure_collection(vector_size=embedder.dimensions, with_sparse=True)
    store.upsert_chunks(
        chunks,
        embedder.embed_documents(texts),
        ingest_run_id="run-test",
        embedded_model=embedder.model,
        sparse_vectors=encoder.encode_documents(texts),
    )
    return store


def _retriever(**kwargs: Any) -> Retriever:
    # Hybrid, so the dense branch contributes every document: this corpus is four
    # sentences, and a sparse-only shortlist would be too shallow to show that the
    # reranker sees more candidates than the caller asked for.
    return Retriever(
        store=_store(),
        embedder=FakeEmbeddingProvider(),
        config=RetrievalConfig(mode="hybrid", top_k=2),
        **kwargs,
    )


class WideReranker:
    """Records how deep a shortlist it was handed."""

    def __init__(self) -> None:
        self.seen = 0

    @property
    def name(self) -> str:
        return "wide"

    def rerank(self, query: str, hits: Any, *, top_n: int) -> list[RetrievalHit]:
        shortlist = list(hits)
        self.seen = len(shortlist)
        return [
            replace(hit, rank=index, rerank_score=0.0, pre_rerank_rank=hit.rank)
            for index, hit in enumerate(shortlist[:top_n], start=1)
        ]


class BrokenReranker:
    @property
    def name(self) -> str:
        return "broken"

    def rerank(self, query: str, hits: Any, *, top_n: int) -> list[RetrievalHit]:
        raise RerankError("the reranker is down")


class TestRetrieverWithoutReranker:
    def test_the_m2_pipeline_is_unchanged(self) -> None:
        result = _retriever().retrieve("qdrant sparse vectors")
        assert len(result.hits) == 2
        assert result.rerank is None
        assert all(hit.rerank_score is None for hit in result.hits)

    def test_hits_do_not_carry_rerank_keys(self) -> None:
        hit = _retriever().retrieve("qdrant sparse vectors").hits[0]
        assert "rerank_score" not in hit.to_dict()
        assert "pre_rerank_rank" not in hit.to_dict()

    def test_the_summary_still_reports_the_stage(self) -> None:
        # Present rather than omitted: an eval row lacking the key would be
        # indistinguishable from one written before the key existed.
        summary = _retriever().retrieve("qdrant sparse vectors").to_summary()
        assert summary["rerank"] == {
            "applied": False,
            "reranker": None,
            "candidates": 0,
            "error": None,
        }

    def test_every_mode_still_works(self) -> None:
        retriever = _retriever()
        for mode in ("dense", "sparse", "hybrid"):
            assert retriever.retrieve("qdrant sparse vectors", mode=mode).mode == mode


class TestRetrieverWithReranker:
    def test_reports_that_reranking_happened(self) -> None:
        result = _retriever(reranker=FakeReranker()).retrieve("cross encoder reranker")
        assert result.rerank is not None
        assert result.rerank.applied is True
        assert result.rerank.reranker == "fake"

    def test_still_returns_top_k_hits(self) -> None:
        result = _retriever(reranker=FakeReranker()).retrieve("qdrant sparse vectors")
        assert len(result.hits) == 2
        assert [hit.rank for hit in result.hits] == [1, 2]

    def test_hits_keep_their_provenance(self) -> None:
        hit = _retriever(reranker=FakeReranker()).retrieve("cross encoder reranker").hits[0]
        assert hit.source_path.startswith("sample/")
        assert hit.chunk_id
        assert hit.point_id
        assert hit.branches
        assert hit.branch_ranks

    def test_the_shortlist_is_deeper_than_top_k(self) -> None:
        # This is the whole point of input_top_k: the reranker can only recover a
        # relevant chunk it was actually shown.
        reranker = WideReranker()
        _retriever(reranker=reranker, rerank_config=RerankConfig(input_top_k=4)).retrieve(
            "qdrant retrieval generation index"
        )
        assert reranker.seen == 4

    def test_a_larger_top_k_than_input_top_k_still_wins(self) -> None:
        reranker = WideReranker()
        _retriever(reranker=reranker, rerank_config=RerankConfig(input_top_k=1)).retrieve(
            "qdrant retrieval generation index", top_k=3
        )
        assert reranker.seen == 3

    def test_the_summary_serialises(self) -> None:
        summary = _retriever(reranker=FakeReranker()).retrieve("cross encoder").to_summary()
        assert json.loads(json.dumps(summary))["rerank"]["applied"] is True
        assert "rerank_score" in summary["hits"][0]

    def test_fail_open_returns_the_fusion_order(self) -> None:
        baseline = _retriever().retrieve("qdrant sparse vectors")
        degraded = _retriever(
            reranker=BrokenReranker(), rerank_config=RerankConfig(fail_open=True)
        ).retrieve("qdrant sparse vectors")
        assert [hit.chunk_id for hit in degraded.hits] == [hit.chunk_id for hit in baseline.hits]
        assert degraded.rerank is not None
        assert degraded.rerank.applied is False
        assert degraded.rerank.error is not None

    def test_fail_closed_propagates(self) -> None:
        retriever = _retriever(
            reranker=BrokenReranker(), rerank_config=RerankConfig(fail_open=False)
        )
        with pytest.raises(RerankError, match="the reranker is down"):
            retriever.retrieve("qdrant sparse vectors")


class TestRetrieveCliRerankFlag:
    @pytest.fixture(autouse=True)
    def _patch_store(self, monkeypatch: pytest.MonkeyPatch) -> None:
        store = _store()
        monkeypatch.setattr(cli, "resolve_searchable_store", lambda **_: store)

    def _run(self, capsys: pytest.CaptureFixture[str], *argv: str) -> dict[str, Any]:
        code = cli.main(list(argv))
        lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
        payload: dict[str, Any] = json.loads(lines[-1])
        payload["_exit_code"] = code
        return payload

    def test_defaults_to_off_so_m2_behaviour_is_preserved(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        payload = self._run(capsys, "--query", "qdrant sparse vectors", "--mode", "hybrid")
        assert payload["_exit_code"] == 0
        assert payload["rerank"]["applied"] is False
        assert payload["rerank"]["reranker"] is None

    def test_rerank_fake_runs_offline_and_is_reported(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        payload = self._run(
            capsys,
            "--query",
            "cross encoder reranker",
            "--mode",
            "hybrid",
            "--rerank",
            "fake",
            "--top-k",
            "2",
        )
        assert payload["_exit_code"] == 0
        assert payload["rerank"]["applied"] is True
        assert payload["rerank"]["reranker"] == "fake"
        assert payload["rerank"]["error"] is None
        # Fusion handed the reranker more than the two hits the caller asked for.
        assert payload["rerank"]["candidates"] > len(payload["hits"])
        assert payload["hits"][0]["source_path"] == "sample/03-rerank.md"
        assert payload["hits"][0]["pre_rerank_rank"] >= 1

    def test_an_unknown_reranker_is_a_usage_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as excinfo:
            cli.main(["--query", "q", "--rerank", "cross-encoder"])
        # argparse rejects it before anything runs; exit code 2 matches the CLI's
        # own "the invocation is wrong" contract.
        assert excinfo.value.code == 2

    def test_cohere_without_a_key_exits_two_without_querying(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        payload = self._run(capsys, "--query", "anything", "--rerank", "cohere")
        assert payload["_exit_code"] == 2
        assert payload["ok"] is False
        assert "API key" in payload["error"]

    def test_help_is_ascii_only(self) -> None:
        # A cp1252 console raises UnicodeEncodeError on an em dash in --help.
        text = cli.build_parser().format_help()
        assert text.isascii()
        assert "--rerank" in text
