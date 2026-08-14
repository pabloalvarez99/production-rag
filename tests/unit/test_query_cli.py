from __future__ import annotations

import json
from collections.abc import Callable, Iterator

import pytest
import structlog

from production_rag.api.schemas import QueryDebug, QueryRequest, QueryResponse
from production_rag.config import Settings
from production_rag.query import cli as query_cli
from production_rag.retrieval.filters import FILTER_NOT_ALLOWED, FilterError

SettingsFactory = Callable[..., Settings]


@pytest.fixture(autouse=True)
def _fresh_structlog() -> Iterator[None]:
    """Rebind the logger to this test's streams.

    ``structlog``'s default logger caches the ``sys.stdout`` it saw first. Without
    this, an error path that logs writes to a ``capsys`` buffer another test
    already closed, and the failure looks like a CLI bug rather than a fixture
    one.
    """
    structlog.reset_defaults()
    yield
    structlog.reset_defaults()


def _last_json(capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    """The CLI's contract: whatever else is printed, the last stdout line is JSON."""
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    parsed: dict[str, object] = json.loads(lines[-1])
    return parsed


def _capture_payload(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> dict[str, object]:
    """Replace the executor with one that records the request it was handed."""
    observed: dict[str, object] = {}

    def fake_execute_query(payload: QueryRequest, **kwargs: object) -> QueryResponse:
        observed["payload"] = payload
        return QueryResponse(answer="Grounded answer [1].", refused=False)

    monkeypatch.setattr(query_cli, "get_settings", lambda: settings)
    monkeypatch.setattr(query_cli, "configure_cli_logging", lambda _: None)
    monkeypatch.setattr(query_cli, "execute_query", fake_execute_query)
    return observed


def test_repeated_filter_flags_become_one_filters_object(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    settings_factory: SettingsFactory,
) -> None:
    observed = _capture_payload(monkeypatch, settings_factory())

    assert (
        query_cli.main(
            [
                "--question",
                "Why use RRF?",
                "--filter",
                "source=sample",
                "--filter",
                "tags=bm25",
                "--filter",
                "tags=rrf",
            ]
        )
        == 0
    )

    payload = observed["payload"]
    assert isinstance(payload, QueryRequest)
    assert payload.filters == {"source": ["sample"], "tags": ["bm25", "rrf"]}


def test_no_filter_flag_leaves_the_request_unfiltered(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    settings_factory: SettingsFactory,
) -> None:
    observed = _capture_payload(monkeypatch, settings_factory())

    assert query_cli.main(["--question", "Why use RRF?"]) == 0

    payload = observed["payload"]
    assert isinstance(payload, QueryRequest)
    assert payload.filters is None


def test_a_rejected_filter_exits_two_with_the_shared_slug(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    settings_factory: SettingsFactory,
) -> None:
    """The executor's rejection is graded like any other bad invocation."""
    settings: Settings = settings_factory()

    def refusing_execute_query(payload: QueryRequest, **kwargs: object) -> QueryResponse:
        raise FilterError(
            "filtering on 'author' is not allowed",
            error_type=FILTER_NOT_ALLOWED,
            field="author",
        )

    monkeypatch.setattr(query_cli, "get_settings", lambda: settings)
    monkeypatch.setattr(query_cli, "configure_cli_logging", lambda _: None)
    monkeypatch.setattr(query_cli, "execute_query", refusing_execute_query)

    assert query_cli.main(["--question", "ok", "--filter", "author=pablo"]) == 2

    body = _last_json(capsys)
    assert body["ok"] is False
    assert body["error_type"] == "filter_not_allowed"


def test_a_malformed_filter_argument_exits_two(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    settings_factory: SettingsFactory,
) -> None:
    _capture_payload(monkeypatch, settings_factory())

    assert query_cli.main(["--question", "ok", "--filter", "source"]) == 2

    assert _last_json(capsys)["error_type"] == "filter_invalid_value"


@pytest.mark.parametrize("debug_requested", [False, True])
def test_query_cli_debug_is_opt_in(
    debug_requested: bool,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    settings_factory: SettingsFactory,
) -> None:
    settings: Settings = settings_factory()
    observed: dict[str, object] = {}

    def fake_execute_query(
        payload: QueryRequest,
        **kwargs: object,
    ) -> QueryResponse:
        observed["payload"] = payload
        observed.update(kwargs)
        response = QueryResponse(
            answer="Grounded answer [1].",
            citations=[],
            refused=False,
            refusal_reason=None,
        )
        if payload.debug:
            return response.model_copy(
                update={
                    "debug": QueryDebug(
                        timings_ms={"retrieve": 1.25},
                        invalid_markers=[],
                    )
                }
            )
        return response

    monkeypatch.setattr(query_cli, "get_settings", lambda: settings)
    monkeypatch.setattr(query_cli, "configure_cli_logging", lambda _: None)
    monkeypatch.setattr(query_cli, "execute_query", fake_execute_query)
    argv = ["--question", "Why use RRF?", "--llm", "fake"]
    if debug_requested:
        argv.append("--debug")

    assert query_cli.main(argv) == 0

    output = capsys.readouterr().out
    body = json.loads(output)
    payload = observed["payload"]
    assert isinstance(payload, QueryRequest)
    assert payload.debug is debug_requested
    if debug_requested:
        assert body["debug"] == {
            "timings_ms": {"retrieve": 1.25},
            "invalid_markers": [],
        }
    else:
        assert "debug" not in body
        assert "timings_ms" not in output
