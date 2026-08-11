"""Source hit@k: the metric, the golden-set parser, and the eval CLI."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import structlog

from production_rag.config_loader import RetrievalConfig
from production_rag.evals import source_hit
from production_rag.evals.source_hit import (
    CaseOutcome,
    EvalError,
    GoldenCase,
    SourceHitReport,
    evaluate_source_hit,
    load_golden,
)
from production_rag.ingest.models import Chunk, Document
from production_rag.retrieval.embeddings import FakeEmbeddingProvider
from production_rag.retrieval.hybrid import MODE_DENSE, MODE_SPARSE, Retriever
from production_rag.retrieval.sparse import Bm25Encoder
from production_rag.retrieval.store import InMemoryVectorStore

CORPUS = {
    "sample/00-intro.md": "Production RAG combines retrieval with generation over a corpus.",
    "sample/01-hybrid-search.md": "Reciprocal rank fusion merges ranked lists by position.",
    "sample/02-reranking.md": "A cross-encoder scores a query and passage together.",
    "sample/03-chunking.md": "The chunker splits markdown on heading boundaries.",
}


@pytest.fixture(autouse=True)
def _reset_structlog() -> Iterator[None]:
    """Undo the CLI's global logging configuration between tests."""
    yield
    structlog.reset_defaults()


def _retriever(**overrides: object) -> Retriever:
    embedder = FakeEmbeddingProvider()
    chunks = []
    for index, (path, text) in enumerate(CORPUS.items()):
        document = Document(source_path=path, text=text, title=path, source="sample")
        chunks.append(Chunk.build(document=document, chunk_index=index, text=text, embed_text=text))
    texts = [chunk.embed_text for chunk in chunks]
    encoder = Bm25Encoder()
    encoder.fit(texts)
    store = InMemoryVectorStore(collection="eval_collection")
    store.ensure_collection(vector_size=embedder.dimensions, with_sparse=True)
    store.upsert_chunks(
        chunks,
        embedder.embed_documents(texts),
        ingest_run_id="run-test",
        embedded_model=embedder.model,
        sparse_vectors=encoder.encode_documents(texts),
    )
    return Retriever(
        store=store,
        embedder=embedder,
        config=RetrievalConfig(**overrides),  # type: ignore[arg-type]
    )


def _case(case_id: str, question: str, *paths: str, category: str | None = None) -> GoldenCase:
    return GoldenCase(
        id=case_id,
        question=question,
        expected_source_paths=paths,
        category=category,
    )


class TestGoldenSet:
    def test_reads_the_repository_golden_set(self) -> None:
        cases = load_golden(Path("data/eval/golden.jsonl"))
        assert cases
        # The paths carry the sample/ prefix, so the corpus root is data/raw.
        assert all(
            path.startswith("sample/") for case in cases for path in case.expected_source_paths
        )

    def test_the_unanswerable_cases_are_kept_but_not_scorable(self) -> None:
        # The golden set deliberately contains questions the corpus cannot answer.
        # Retrieval cannot hit a source that does not exist, and refusing is a
        # generation-time property, so they are excluded from the aggregate.
        cases = load_golden(Path("data/eval/golden.jsonl"))
        unanswerable = [case for case in cases if not case.is_scorable]
        assert unanswerable
        assert all(case.category == "unanswerable" for case in unanswerable)

    def test_an_empty_expected_list_is_accepted_as_unanswerable(self, tmp_path: Path) -> None:
        golden = tmp_path / "golden.jsonl"
        golden.write_text(
            json.dumps({"id": "q", "question": "why", "expected_source_paths": []}) + "\n",
            encoding="utf-8",
        )
        assert load_golden(golden)[0].is_scorable is False

    def test_paths_are_normalised_like_the_ingest_does(self, tmp_path: Path) -> None:
        # A hand-written golden set must not read as a miss because of a stray
        # backslash or "./" prefix.
        golden = tmp_path / "golden.jsonl"
        golden.write_text(
            json.dumps({"id": "q", "question": "why", "expected_source_paths": ["./sample\\a.md"]})
            + "\n",
            encoding="utf-8",
        )
        assert load_golden(golden)[0].expected_source_paths == ("sample/a.md",)

    def test_blank_and_comment_lines_are_skipped(self, tmp_path: Path) -> None:
        golden = tmp_path / "golden.jsonl"
        golden.write_text(
            "# a comment\n\n"
            + json.dumps({"id": "q", "question": "why", "expected_source_paths": ["a.md"]})
            + "\n",
            encoding="utf-8",
        )
        assert len(load_golden(golden)) == 1

    def test_a_missing_file_is_reported_with_its_path(self, tmp_path: Path) -> None:
        with pytest.raises(EvalError, match="golden set not found"):
            load_golden(tmp_path / "absent.jsonl")

    def test_an_empty_golden_set_is_refused(self, tmp_path: Path) -> None:
        golden = tmp_path / "golden.jsonl"
        golden.write_text("\n", encoding="utf-8")
        with pytest.raises(EvalError, match="no cases"):
            load_golden(golden)

    def test_malformed_json_names_the_line(self, tmp_path: Path) -> None:
        golden = tmp_path / "golden.jsonl"
        golden.write_text("{not json}\n", encoding="utf-8")
        with pytest.raises(EvalError, match="line 1 is not valid JSON"):
            load_golden(golden)

    @pytest.mark.parametrize(
        "record",
        [
            {"question": "why", "expected_source_paths": ["a.md"]},
            {"id": "q", "expected_source_paths": ["a.md"]},
            {"id": "q", "question": "why"},
            {"id": "q", "question": "why", "expected_source_paths": "a.md"},
        ],
    )
    def test_an_incomplete_record_is_refused(self, tmp_path: Path, record: dict[str, Any]) -> None:
        # Silently skipping a case would inflate the average over the ones left.
        golden = tmp_path / "golden.jsonl"
        golden.write_text(json.dumps(record) + "\n", encoding="utf-8")
        with pytest.raises(EvalError):
            load_golden(golden)


class TestMetric:
    def test_a_hit_is_scored_with_its_rank(self) -> None:
        report = evaluate_source_hit(
            retriever=_retriever(),
            cases=[_case("q1", "reciprocal rank fusion", "sample/01-hybrid-search.md")],
            k=5,
            mode=MODE_SPARSE,
        )
        assert report.score == 1.0
        assert report.cases[0].hit_rank == 1

    def test_a_miss_is_scored_zero_and_keeps_what_was_retrieved(self) -> None:
        report = evaluate_source_hit(
            retriever=_retriever(),
            cases=[_case("q1", "cross-encoder", "sample/03-chunking.md")],
            k=1,
            mode=MODE_SPARSE,
        )
        assert report.score == 0.0
        assert report.cases[0].hit is False
        assert report.cases[0].hit_rank is None
        # The misses are the actionable part, so what came back is recorded.
        assert report.cases[0].retrieved_source_paths

    def test_the_score_is_the_mean_over_cases(self) -> None:
        report = evaluate_source_hit(
            retriever=_retriever(),
            cases=[
                _case("q1", "reciprocal rank fusion", "sample/01-hybrid-search.md"),
                _case("q2", "cross-encoder", "sample/03-chunking.md"),
            ],
            k=1,
            mode=MODE_SPARSE,
        )
        assert report.score == pytest.approx(0.5)
        assert report.hits == 1

    def test_k_bounds_what_counts_as_a_hit(self) -> None:
        cases = [_case("q1", "markdown heading boundaries", "sample/03-chunking.md")]
        deep = evaluate_source_hit(retriever=_retriever(), cases=cases, k=4, mode=MODE_DENSE)
        shallow = evaluate_source_hit(retriever=_retriever(), cases=cases, k=1, mode=MODE_DENSE)
        # Same retrieval, stricter cutoff: hit@k can only fall as k falls.
        assert shallow.score <= deep.score

    def test_any_expected_path_counts_as_a_hit(self) -> None:
        report = evaluate_source_hit(
            retriever=_retriever(),
            cases=[
                _case(
                    "q1",
                    "reciprocal rank fusion",
                    "sample/99-absent.md",
                    "sample/01-hybrid-search.md",
                )
            ],
            k=5,
            mode=MODE_SPARSE,
        )
        assert report.cases[0].hit is True

    def test_the_mode_is_recorded_and_switchable(self) -> None:
        cases = [_case("q1", "reciprocal rank fusion", "sample/01-hybrid-search.md")]
        # A hybrid number without the single-branch numbers is not evidence.
        for mode in (MODE_DENSE, MODE_SPARSE):
            assert (
                evaluate_source_hit(retriever=_retriever(), cases=cases, k=3, mode=mode).mode
                == mode
            )

    def test_the_report_records_collection_and_model(self) -> None:
        report = evaluate_source_hit(
            retriever=_retriever(),
            cases=[_case("q1", "fusion", "sample/01-hybrid-search.md")],
            k=3,
        )
        assert report.collection == "eval_collection"
        assert report.embedded_model == FakeEmbeddingProvider().model

    def test_an_unanswerable_case_is_excluded_from_the_aggregate(self) -> None:
        report = evaluate_source_hit(
            retriever=_retriever(),
            cases=[
                _case("q1", "reciprocal rank fusion", "sample/01-hybrid-search.md"),
                _case("q2", "how many concurrent requests", category="unanswerable"),
            ],
            k=3,
            mode=MODE_SPARSE,
        )
        # One scored case, one hit: excluding the unanswerable one keeps the score
        # at 1.0 instead of halving it for something retrieval cannot fix.
        assert report.score == 1.0
        assert report.unscored == 1
        assert len(report.scored_cases) == 1
        assert report.to_summary()["unscored_cases"] == 1

    def test_a_non_positive_k_is_refused(self) -> None:
        with pytest.raises(EvalError, match="k must be positive"):
            evaluate_source_hit(retriever=_retriever(), cases=[], k=0)

    def test_an_empty_case_list_scores_zero_rather_than_dividing_by_zero(self) -> None:
        report = evaluate_source_hit(retriever=_retriever(), cases=[], k=3)
        assert report.score == 0.0


class TestReport:
    @staticmethod
    def _report() -> SourceHitReport:
        return SourceHitReport(
            mode="hybrid",
            k=5,
            collection="c",
            embedded_model="m",
            golden_path="g",
            cases=(
                CaseOutcome(
                    id="q1",
                    question="a",
                    hit=True,
                    expected_source_paths=("a.md",),
                    retrieved_source_paths=("a.md",),
                    hit_rank=1,
                    category="conceptual",
                ),
                CaseOutcome(
                    id="q2",
                    question="b",
                    hit=False,
                    expected_source_paths=("b.md",),
                    retrieved_source_paths=("a.md",),
                    category="conceptual",
                ),
                CaseOutcome(
                    id="q3",
                    question="c",
                    hit=True,
                    expected_source_paths=("c.md",),
                    retrieved_source_paths=("c.md",),
                    hit_rank=2,
                    category="exact-token",
                ),
            ),
        )

    def test_per_category_breakdown(self) -> None:
        assert self._report().by_category == {
            "conceptual": pytest.approx(0.5),
            "exact-token": pytest.approx(1.0),
        }

    def test_summary_carries_the_aggregate_and_the_misses(self) -> None:
        summary = self._report().to_summary()
        assert summary["ok"] is True
        assert summary["metric"] == "source_hit_at_k"
        assert summary["cases"] == 3
        assert summary["hits"] == 2
        assert summary["score"] == pytest.approx(0.6667, abs=1e-4)
        assert [case["id"] for case in summary["misses"]] == ["q2"]
        assert len(summary["results"]) == 3

    def test_uncategorised_cases_do_not_appear_in_the_breakdown(self) -> None:
        report = SourceHitReport(
            mode="hybrid",
            k=1,
            collection="c",
            embedded_model="m",
            golden_path="g",
            cases=(
                CaseOutcome(
                    id="q",
                    question="a",
                    hit=True,
                    expected_source_paths=("a.md",),
                    retrieved_source_paths=("a.md",),
                ),
            ),
        )
        assert report.by_category == {}


class TestCli:
    @pytest.fixture
    def patched(self, monkeypatch: pytest.MonkeyPatch) -> Retriever:
        retriever = _retriever()
        monkeypatch.setattr(source_hit, "resolve_searchable_store", lambda **_: retriever.store)
        return retriever

    @staticmethod
    def _golden(tmp_path: Path) -> Path:
        golden = tmp_path / "golden.jsonl"
        golden.write_text(
            "\n".join(
                json.dumps(record)
                for record in (
                    {
                        "id": "q-0001",
                        "question": "reciprocal rank fusion",
                        "expected_source_paths": ["sample/01-hybrid-search.md"],
                        "category": "conceptual",
                    },
                    {
                        "id": "q-0002",
                        "question": "cross-encoder",
                        "expected_source_paths": ["sample/02-reranking.md"],
                        "category": "exact-token",
                    },
                )
            )
            + "\n",
            encoding="utf-8",
        )
        return golden

    def _run(self, capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict[str, Any]]:
        code = source_hit.main(list(argv))
        lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
        return code, json.loads(lines[-1])

    def test_scores_the_golden_set_and_exits_zero(
        self, patched: Retriever, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, payload = self._run(
            capsys, "--golden", str(self._golden(tmp_path)), "--mode", "sparse", "--k", "3"
        )
        assert code == 0
        assert payload["score"] == pytest.approx(1.0)
        assert payload["k"] == 3
        assert payload["mode"] == "sparse"
        assert payload["golden_path"] == str(self._golden(tmp_path))

    def test_the_last_stdout_line_is_the_only_json(
        self, patched: Retriever, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        source_hit.main(["--golden", str(self._golden(tmp_path)), "--log-level", "DEBUG"])
        captured = capsys.readouterr()
        assert len(captured.out.strip().splitlines()) == 1
        assert captured.err

    def test_a_missing_golden_file_exits_two(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        code, payload = self._run(capsys, "--golden", str(tmp_path / "absent.jsonl"))
        assert code == 2
        assert payload["error_type"] == "EvalError"

    def test_defaults_are_the_documented_ones(self) -> None:
        args = source_hit.build_parser().parse_args([])
        assert args.k == 5
        assert args.embedder == "fake"
        assert args.golden.endswith("golden.jsonl")
        assert args.mode is None

    def test_help_is_ascii_only(self) -> None:
        assert source_hit.build_parser().format_help().encode("ascii", errors="strict")
