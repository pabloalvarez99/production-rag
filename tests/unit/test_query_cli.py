from __future__ import annotations

import json
from collections.abc import Callable

import pytest

from production_rag.api.schemas import QueryDebug, QueryRequest, QueryResponse
from production_rag.config import Settings
from production_rag.query import cli as query_cli

SettingsFactory = Callable[..., Settings]


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
