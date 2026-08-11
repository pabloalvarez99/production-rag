"""The retrieve CLI: stream discipline, exit codes, and flag precedence.

The store is patched out, so nothing here touches a network. What is being tested
is the contract a wrapper script depends on: the last line of stdout is one JSON
object, logs never pollute it, and the exit code says whether retrying is worth it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
import structlog

from production_rag.ingest.models import Chunk, Document
from production_rag.retrieval import cli
from production_rag.retrieval.embeddings import EmbeddingError, FakeEmbeddingProvider
from production_rag.retrieval.sparse import Bm25Encoder
from production_rag.retrieval.store import InMemoryVectorStore, VectorStoreError

CORPUS = {
    "sample/00-intro.md": "Production RAG combines retrieval and generation.",
    "sample/01-qdrant.md": "Qdrant stores dense and sparse vectors in one collection.",
    "sample/02-flags.md": "Pass --recreate-collection to rebuild the index.",
}


@pytest.fixture(autouse=True)
def _reset_structlog() -> Iterator[None]:
    """Undo the CLI's global logging configuration between tests.

    ``main`` calls ``structlog.configure``; leaving that in place would let one
    test's log configuration decide another test's behaviour.
    """
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


@pytest.fixture
def patched_store(monkeypatch: pytest.MonkeyPatch) -> InMemoryVectorStore:
    """Replace the Qdrant store with the offline one, keeping the CLI intact."""
    store = _store()
    monkeypatch.setattr(cli, "resolve_searchable_store", lambda **_: store)
    return store


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict[str, Any], str]:
    """Run the CLI and return ``(exit code, parsed last stdout line, stderr)``."""
    code = cli.main(list(argv))
    captured = capsys.readouterr()
    lines = [line for line in captured.out.splitlines() if line.strip()]
    return code, json.loads(lines[-1]), captured.err


class TestSuccessPath:
    def test_returns_hits_and_exit_zero(
        self, patched_store: InMemoryVectorStore, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, payload, _ = _run(capsys, "--query", "qdrant vectors")
        assert code == 0
        assert payload["ok"] is True
        assert payload["hits"]
        assert payload["collection"] == "test_collection"

    def test_the_last_stdout_line_is_the_only_json(
        self, patched_store: InMemoryVectorStore, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cli.main(["--query", "qdrant", "--log-level", "DEBUG"])
        captured = capsys.readouterr()
        assert len(captured.out.strip().splitlines()) == 1
        # Logs exist, and they are on the other stream.
        assert captured.err

    def test_hits_carry_their_citation_fields(
        self, patched_store: InMemoryVectorStore, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _, payload, _ = _run(capsys, "--query", "qdrant vectors")
        hit = payload["hits"][0]
        assert hit["source_path"] in CORPUS
        assert hit["chunk_id"]
        assert hit["rank"] == 1
        assert set(hit["branches"]) <= {"dense", "sparse"}

    def test_mode_flag_reaches_the_retriever(
        self, patched_store: InMemoryVectorStore, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Equals form: a query starting with "-" would otherwise be parsed as a flag.
        _, payload, _ = _run(capsys, "--query=--recreate-collection", "--mode", "sparse")
        assert payload["mode"] == "sparse"
        assert payload["dense_candidates"] == 0
        assert payload["hits"][0]["source_path"] == "sample/02-flags.md"

    def test_top_k_flag_bounds_the_hits(
        self, patched_store: InMemoryVectorStore, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _, payload, _ = _run(capsys, "--query", "qdrant", "--top-k", "1")
        assert payload["returned"] == 1

    def test_a_query_matching_nothing_is_an_honest_empty_success(
        self, patched_store: InMemoryVectorStore, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # No hits is a result, not a failure: exit 0 with returned == 0.
        code, payload, _ = _run(capsys, "--query", "kubernetes", "--mode", "sparse")
        assert code == 0
        assert payload["returned"] == 0
        assert payload["hits"] == []

    def test_the_summary_records_what_produced_it(
        self, patched_store: InMemoryVectorStore, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _, payload, _ = _run(capsys, "--query", "qdrant")
        for key in ("mode", "fusion_k", "weights", "embedded_model", "score_threshold"):
            assert key in payload


class TestUsageErrors:
    def test_a_bad_mode_is_rejected_by_the_parser(self) -> None:
        # argparse exits 2 itself, which is the same code the CLI uses for a bad
        # invocation, so the contract holds either way.
        with pytest.raises(SystemExit) as exit_info:
            cli.main(["--query", "x", "--mode", "rerank"])
        assert exit_info.value.code == 2

    def test_a_missing_query_is_rejected(self) -> None:
        with pytest.raises(SystemExit) as exit_info:
            cli.main([])
        assert exit_info.value.code == 2

    def test_an_empty_query_exits_two_with_a_json_error(
        self, patched_store: InMemoryVectorStore, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, payload, _ = _run(capsys, "--query", "   ")
        assert code == 2
        assert payload["ok"] is False
        assert payload["error_type"] == "RetrievalError"

    def test_a_bad_config_file_exits_two(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Any
    ) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("ingest: [unclosed", encoding="utf-8")
        code, payload, _ = _run(capsys, "--query", "x", "--config", str(bad))
        assert code == 2
        assert payload["error_type"] == "ConfigFileError"

    def test_a_missing_api_key_exits_two_before_anything_runs(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def _refuse(*_: object, **__: object) -> None:
            raise EmbeddingError("OPENAI_API_KEY is not set; use --embedder fake")

        monkeypatch.setattr(cli, "resolve_embedder", _refuse)
        code, payload, _ = _run(capsys, "--query", "x", "--embedder", "openai")
        assert code == 2
        assert "--embedder fake" in payload["error"]


class TestRuntimeErrors:
    def test_an_unreachable_store_exits_one(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        class _Broken(InMemoryVectorStore):
            def search_dense(self, *_: object, **__: object) -> list[Any]:
                raise VectorStoreError("connection refused")

        broken = _Broken()
        broken.ensure_collection(vector_size=FakeEmbeddingProvider().dimensions, with_sparse=True)
        monkeypatch.setattr(cli, "resolve_searchable_store", lambda **_: broken)
        code, payload, _ = _run(capsys, "--query", "qdrant")
        # Environmental and possibly transient: a wrapper may retry this one.
        assert code == 1
        assert payload["error_type"] == "VectorStoreError"


class TestParser:
    def test_help_is_ascii_only(self) -> None:
        # --help is written to a console that may be cp1252, where a stray em dash
        # raises UnicodeEncodeError instead of printing help.
        text = cli.build_parser().format_help()
        assert text.encode("ascii", errors="strict")

    def test_help_states_the_exit_code_contract(self) -> None:
        text = cli.build_parser().format_help()
        assert "0 ok" in text
        assert "JSON" in text

    def test_the_default_embedder_is_fake(self) -> None:
        assert cli.build_parser().parse_args(["--query", "x"]).embedder == "fake"

    def test_there_is_no_api_key_flag(self) -> None:
        # A key on a command line lands in shell history and in `ps` output.
        assert "--api-key" not in cli.build_parser().format_help()
