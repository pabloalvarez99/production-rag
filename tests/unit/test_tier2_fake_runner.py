"""The unified eval runner: sampling, judge gating, the report, and the CLI.

The CLI tests substitute the vector store for an in-memory one, exactly as the
ablation tests do, so the whole runner is exercised end to end without Qdrant,
without a key and without a bill.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import structlog

from production_rag.config_loader import RetrievalConfig, YamlConfig
from production_rag.evals import run as runner
from production_rag.evals.judges import FakeJudge, JudgeError, OpenAIJudge
from production_rag.evals.run import (
    DEFAULT_SEED,
    REPORT_VERSION,
    build_parser,
    resolve_judge,
    sample_cases,
)
from production_rag.evals.source_hit import EvalError, GoldenCase
from production_rag.ingest.models import Chunk, Document
from production_rag.retrieval.embeddings import FakeEmbeddingProvider
from production_rag.retrieval.hybrid import Retriever
from production_rag.retrieval.sparse import Bm25Encoder
from production_rag.retrieval.store import InMemoryVectorStore

CORPUS = {
    "sample/01-hybrid-search.md": "Reciprocal rank fusion merges ranked lists by position.",
    "sample/02-reranking.md": "A cross-encoder scores a query and passage together.",
}


@pytest.fixture(autouse=True)
def _reset_structlog() -> Iterator[None]:
    """Undo the CLI's global logging configuration between tests."""
    yield
    structlog.reset_defaults()


def _cases(count: int) -> list[GoldenCase]:
    return [
        GoldenCase(id=f"q-{index}", question=f"question {index}", expected_source_paths=("a.md",))
        for index in range(count)
    ]


def _store() -> InMemoryVectorStore:
    embedder = FakeEmbeddingProvider()
    store = InMemoryVectorStore(collection="eval_collection")
    store.ensure_collection(vector_size=embedder.dimensions, with_sparse=True)
    chunks = [
        Chunk.build(
            document=Document(source_path=path, text=text, title=path, source="sample"),
            chunk_index=index,
            text=text,
            embed_text=text,
        )
        for index, (path, text) in enumerate(CORPUS.items())
    ]
    texts = [chunk.embed_text for chunk in chunks]
    encoder = Bm25Encoder()
    encoder.fit(texts)
    store.upsert_chunks(
        chunks,
        embedder.embed_documents(texts),
        ingest_run_id="run-test",
        embedded_model=embedder.model,
        sparse_vectors=encoder.encode_documents(texts),
    )
    return store


class TestSampling:
    def test_no_sample_keeps_every_case(self) -> None:
        cases = _cases(5)
        assert sample_cases(cases, sample=None, seed=DEFAULT_SEED) == cases

    def test_a_sample_larger_than_the_set_keeps_every_case(self) -> None:
        cases = _cases(3)
        assert sample_cases(cases, sample=99, seed=DEFAULT_SEED) == cases

    def test_a_sample_is_reproducible_for_a_seed(self) -> None:
        cases = _cases(20)
        first = sample_cases(cases, sample=5, seed=42)
        second = sample_cases(cases, sample=5, seed=42)
        assert [case.id for case in first] == [case.id for case in second]

    def test_a_different_seed_draws_differently(self) -> None:
        cases = _cases(50)
        first = {case.id for case in sample_cases(cases, sample=10, seed=1)}
        second = {case.id for case in sample_cases(cases, sample=10, seed=2)}
        assert first != second

    def test_the_sample_keeps_golden_order(self) -> None:
        # Membership is sampled; order is the file's, so two runs at different
        # sample sizes produce reports whose shared cases line up.
        cases = _cases(20)
        drawn = sample_cases(cases, sample=6, seed=42)
        assert [case.id for case in drawn] == [
            case.id for case in cases if case.id in {drawn_case.id for drawn_case in drawn}
        ]

    def test_a_non_positive_sample_is_rejected(self) -> None:
        with pytest.raises(EvalError, match="--sample must be positive"):
            sample_cases(_cases(3), sample=0, seed=DEFAULT_SEED)


class TestJudgeGating:
    def test_the_fake_judge_needs_no_permission(self) -> None:
        judge = resolve_judge(
            "fake", config=YamlConfig(), settings=runner.get_settings(), allow_llm_evals=False
        )
        assert isinstance(judge, FakeJudge)

    def test_a_hosted_judge_without_the_opt_in_refuses_to_start(self) -> None:
        # A refusal, not a silent downgrade: a run that swapped its judge would
        # report offline numbers under a hosted judge's name.
        with pytest.raises(JudgeError, match="RUN_LLM_EVALS=1"):
            resolve_judge(
                "openai",
                config=YamlConfig(),
                settings=runner.get_settings(),
                allow_llm_evals=False,
            )

    def test_a_hosted_judge_without_a_credential_refuses_to_start(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        settings = runner.get_settings().model_copy(update={"openai_api_key": None})
        with pytest.raises(JudgeError, match="needs a credential"):
            resolve_judge("openai", config=YamlConfig(), settings=settings, allow_llm_evals=True)

    def test_the_env_var_is_the_gate_when_no_override_is_given(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RUN_LLM_EVALS", "1")
        monkeypatch.setenv("OPENAI_API_KEY", "test-credential-must-not-leak")
        settings = runner.get_settings().model_copy(
            update={"openai_api_key": "test-credential-must-not-leak"}
        )
        judge = resolve_judge("openai", config=YamlConfig(), settings=settings)
        assert isinstance(judge, OpenAIJudge)


class TestParser:
    def test_the_defaults_are_the_offline_ones(self) -> None:
        args = build_parser().parse_args([])
        assert (args.tier, args.embedder, args.llm, args.judge) == ("all", "fake", "fake", "fake")
        assert (args.k, args.seed, args.fail_under_hit) == (5, DEFAULT_SEED, 0.0)

    @pytest.mark.parametrize("tier", ["1", "2", "all"])
    def test_every_locked_tier_is_accepted(self, tier: str) -> None:
        assert build_parser().parse_args(["--tier", tier]).tier == tier

    def test_an_unknown_tier_is_rejected(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--tier", "3"])

    def test_the_locked_flags_all_parse(self) -> None:
        args = build_parser().parse_args(
            [
                "--tier",
                "all",
                "--k",
                "7",
                "--sample",
                "3",
                "--seed",
                "1",
                "--llm",
                "fake",
                "--embedder",
                "fake",
                "--fail-under-hit",
                "0.8",
            ]
        )
        assert (args.k, args.sample, args.seed, args.fail_under_hit) == (7, 3, 1, 0.8)


class TestCli:
    @pytest.fixture(autouse=True)
    def store(self, monkeypatch: pytest.MonkeyPatch) -> InMemoryVectorStore:
        store = _store()
        monkeypatch.setattr(runner, "resolve_searchable_store", lambda **_: store)
        monkeypatch.setattr(
            runner.Retriever,
            "from_config",
            classmethod(
                lambda cls, *, store, embedder, config, reranker=None: Retriever(
                    store=store,
                    embedder=embedder,
                    config=RetrievalConfig(mode="sparse"),
                    reranker=reranker,
                )
            ),
        )
        return store

    @staticmethod
    def _golden(tmp_path: Path) -> Path:
        path = tmp_path / "golden.jsonl"
        path.write_text(
            "\n".join(
                json.dumps(record)
                for record in (
                    {
                        "id": "q1",
                        "question": "reciprocal rank fusion",
                        "expected_source_paths": ["sample/01-hybrid-search.md"],
                        "category": "exact_token",
                    },
                    {
                        "id": "q2",
                        "question": "cross-encoder",
                        "expected_source_paths": ["sample/02-reranking.md"],
                        "category": "conceptual",
                    },
                    {
                        "id": "q3",
                        "question": "how many concurrent requests are supported",
                        "expected_source_paths": [],
                        "category": "unanswerable",
                    },
                )
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    @staticmethod
    def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict[str, Any]]:
        code = runner.main(list(argv))
        output = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
        return code, json.loads(output[-1])

    def test_tier1_only_reports_retrieval(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, payload = self._run(
            capsys, "--tier", "1", "--golden", str(self._golden(tmp_path)), "--k", "2"
        )
        assert code == 0
        assert payload["ok"] is True
        assert payload["report_version"] == REPORT_VERSION
        assert payload["tier1"]["source_hit_at_k"] == 1.0
        assert "tier2" not in payload

    def test_tier2_only_reports_answers(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, payload = self._run(
            capsys, "--tier", "2", "--golden", str(self._golden(tmp_path)), "--k", "2"
        )
        assert code == 0
        assert payload["tier2"]["refusal_accuracy"] == 1.0
        assert payload["judge"] == FakeJudge().name
        assert "tier1" not in payload

    def test_all_runs_both_over_one_sample(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # The point of the unified runner: the two tiers describe the same cases,
        # so their numbers are comparable without reconciling two invocations.
        code, payload = self._run(capsys, "--golden", str(self._golden(tmp_path)), "--k", "2")
        assert code == 0
        assert payload["tier1"]["cases"] == payload["tier2"]["cases"] == 3
        assert payload["case_ids"] == ["q1", "q2", "q3"]

    def test_the_report_says_the_run_was_offline(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _, payload = self._run(capsys, "--golden", str(self._golden(tmp_path)))
        assert payload["offline_defaults"] is True
        assert payload["embedder"] == "fake"
        assert payload["llm"] == "fake"

    def test_sampling_is_recorded_and_reproducible(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        golden = str(self._golden(tmp_path))
        _, first = self._run(capsys, "--golden", golden, "--sample", "2", "--seed", "42")
        _, second = self._run(capsys, "--golden", golden, "--sample", "2", "--seed", "42")
        assert first["case_ids"] == second["case_ids"]
        assert first["scored_cases"] == 2
        assert first["golden_cases"] == 3

    def test_the_unanswerable_case_is_refused_not_answered(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _, payload = self._run(capsys, "--tier", "2", "--golden", str(self._golden(tmp_path)))
        refused = [case for case in payload["tier2"]["results"] if case["refused"]]
        assert [case["id"] for case in refused] == ["q3"]

    def test_a_met_gate_passes(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        code, payload = self._run(
            capsys,
            "--tier",
            "1",
            "--golden",
            str(self._golden(tmp_path)),
            "--fail-under-hit",
            "0.5",
        )
        assert code == 0
        assert payload["gate"]["passed"] is True

    def test_an_unmet_gate_exits_nonzero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        golden = tmp_path / "impossible.jsonl"
        golden.write_text(
            json.dumps(
                {
                    "id": "q1",
                    "question": "kubernetes autoscaling",
                    "expected_source_paths": ["sample/99-missing.md"],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        code, payload = self._run(
            capsys, "--tier", "1", "--golden", str(golden), "--fail-under-hit", "0.8"
        )
        assert code == 1
        assert payload["ok"] is False
        assert payload["gate"]["passed"] is False

    def test_no_gate_is_configured_by_default(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # ADR-0003 sets thresholds from a baseline run, not from ambition.
        _, payload = self._run(capsys, "--tier", "1", "--golden", str(self._golden(tmp_path)))
        assert "gate" not in payload

    def test_answers_can_be_kept_out_of_the_report_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _, payload = self._run(
            capsys, "--tier", "2", "--golden", str(self._golden(tmp_path)), "--no-answers"
        )
        assert all("answer" not in case for case in payload["tier2"]["results"])

    def test_the_report_can_be_written_to_disk(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        destination = tmp_path / "reports" / "eval.json"
        code, payload = self._run(
            capsys, "--golden", str(self._golden(tmp_path)), "--report", str(destination)
        )
        assert code == 0
        assert json.loads(destination.read_text(encoding="utf-8")) == payload

    def test_a_missing_golden_set_is_a_usage_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, payload = self._run(capsys, "--golden", str(tmp_path / "missing.jsonl"))
        assert code == 2
        assert payload["ok"] is False
        assert payload["error_type"] == "EvalError"

    def test_a_hosted_judge_without_the_opt_in_is_a_usage_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("RUN_LLM_EVALS", raising=False)
        code, payload = self._run(
            capsys, "--tier", "2", "--golden", str(self._golden(tmp_path)), "--judge", "openai"
        )
        assert code == 2
        assert payload["error_type"] == "JudgeError"

    def test_the_default_run_never_reaches_a_provider(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The property the whole tier depends on: no key, no network, no bill.
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        code, payload = self._run(capsys, "--golden", str(self._golden(tmp_path)))
        assert code == 0
        assert payload["tier2"]["model"] == "fake-extractive-v1"
