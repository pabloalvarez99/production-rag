"""The ingest command line: flags, exit codes, and the JSON contract.

Every case runs ``--dry-run`` or fails before any connection, so nothing here
reaches Qdrant or an embedding provider.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import structlog

from production_rag.ingest.cli import EXIT_OK, EXIT_USAGE, build_parser, main


@pytest.fixture(autouse=True)
def _restore_structlog() -> object:
    """Undo the CLI's global structlog configuration after each test."""
    yield
    structlog.reset_defaults()


CORPUS_BODY = " ".join(["Hybrid retrieval fuses dense and sparse candidate lists."] * 12)


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "raw"
    (root / "guide").mkdir(parents=True)
    (root / "guide" / "hybrid.md").write_text(f"# Hybrid\n\n{CORPUS_BODY}\n", encoding="utf-8")
    return root


def _last_json(capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    """Parse the last line of stdout, which is the documented machine contract."""
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    return json.loads(lines[-1])  # type: ignore[no-any-return]


# --- flag surface ----------------------------------------------------------


def test_defaults_to_the_fake_embedder() -> None:
    # A portfolio clone must run with no credentials at all.
    assert build_parser().parse_args([]).embedder == "fake"


def test_both_recreate_spellings_set_the_same_flag() -> None:
    # scripts/ingest.ps1 and the Makefile pass --recreate; the docs say
    # --recreate-collection. Accepting one and not the other breaks an operator
    # path that is only exercised when something has already gone wrong.
    parser = build_parser()
    assert parser.parse_args(["--recreate"]).recreate is True
    assert parser.parse_args(["--recreate-collection"]).recreate is True
    assert parser.parse_args([]).recreate is False


def test_incremental_is_unset_unless_disabled() -> None:
    # None means "use the config file", which is not the same as False.
    assert build_parser().parse_args([]).incremental is None
    assert build_parser().parse_args(["--no-incremental"]).incremental is False


def test_an_unknown_embedder_is_rejected_by_argparse() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--embedder", "word2vec"])


def test_there_is_no_flag_for_a_credential() -> None:
    # Regla 3: a key on a command line lands in shell history and in
    # `docker inspect`. It must come from the environment only.
    help_text = build_parser().format_help()
    assert "--api-key" not in help_text
    assert "--openai" not in help_text


# --- runs ------------------------------------------------------------------


def test_dry_run_succeeds_and_prints_a_json_summary(
    corpus: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["--source", str(corpus), "--dry-run", "--config", "does-not-exist.yaml"])
    assert code == EXIT_OK
    summary = _last_json(capsys)
    assert summary["ok"] is True
    assert summary["dry_run"] is True
    assert isinstance(summary["chunks_created"], int)
    assert summary["chunks_created"] > 0
    assert summary["chunks_upserted"] == 0


def test_collection_flag_wins_over_every_other_source(
    corpus: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["--source", str(corpus), "--dry-run", "--collection", "explicit_name"])
    assert _last_json(capsys)["collection"] == "explicit_name"


def test_collection_falls_back_to_the_environment(
    corpus: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("QDRANT_COLLECTION", "from_env")
    from production_rag.config import get_settings

    get_settings.cache_clear()
    main(["--source", str(corpus), "--dry-run"])
    assert _last_json(capsys)["collection"] == "from_env"


def test_a_missing_corpus_exits_2_with_a_json_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["--source", str(tmp_path / "absent"), "--dry-run"])
    assert code == EXIT_USAGE
    error = _last_json(capsys)
    assert error["ok"] is False
    assert error["error_type"] == "CorpusError"


def test_openai_without_a_credential_exits_2_before_connecting(
    corpus: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["--source", str(corpus), "--embedder", "openai", "--dry-run"])
    assert code == EXIT_USAGE
    assert _last_json(capsys)["error_type"] == "EmbeddingError"


def test_an_unparsable_config_file_exits_2(
    corpus: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "broken.yaml"
    bad.write_text("ingest: [unclosed\n", encoding="utf-8")
    code = main(["--source", str(corpus), "--config", str(bad), "--dry-run"])
    assert code == EXIT_USAGE
    assert _last_json(capsys)["error_type"] == "ConfigFileError"


def test_logs_go_to_stderr_so_stdout_stays_parsable(
    corpus: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["--source", str(corpus), "--dry-run", "--log-level", "INFO"])
    captured = capsys.readouterr()
    assert "ingest_completed" in captured.err
    assert len([line for line in captured.out.splitlines() if line.strip()]) == 1
