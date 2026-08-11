"""Unit tests for the request context. Offline: structlog contextvars only."""

from __future__ import annotations

import pytest
import structlog

from production_rag.observability.context import (
    REQUEST_ID_KEY,
    current_request_id,
    new_request_id,
    request_context,
    resolve_request_id,
)


@pytest.fixture(autouse=True)
def _clean_context() -> None:
    """Never let one test's binding be another test's context."""
    structlog.contextvars.clear_contextvars()


class TestResolve:
    def test_a_supplied_id_is_kept(self) -> None:
        assert resolve_request_id("req-1") == "req-1"

    def test_surrounding_whitespace_is_stripped(self) -> None:
        assert resolve_request_id("  req-1\n") == "req-1"

    @pytest.mark.parametrize("raw", [None, "", "   "])
    def test_an_absent_id_is_minted(self, raw: str | None) -> None:
        assert resolve_request_id(raw)

    def test_minted_ids_are_unique(self) -> None:
        assert new_request_id() != new_request_id()


class TestBinding:
    def test_the_id_is_visible_inside_the_block(self) -> None:
        with request_context("req-1") as bound:
            assert bound == "req-1"
            assert current_request_id() == "req-1"

    def test_the_id_reaches_the_log_context(self) -> None:
        with request_context("req-1"):
            assert structlog.contextvars.get_contextvars()[REQUEST_ID_KEY] == "req-1"

    def test_extra_fields_bind_alongside(self) -> None:
        with request_context("req-1", surface="cli"):
            assert structlog.contextvars.get_contextvars()["surface"] == "cli"

    def test_a_missing_id_is_minted_and_returned(self) -> None:
        with request_context() as bound:
            assert bound == current_request_id()
            assert bound

    def test_the_context_is_clear_after_the_block(self) -> None:
        with request_context("req-1"):
            pass
        assert current_request_id() is None

    def test_a_failing_body_still_clears_the_context(self) -> None:
        # The case that matters: an id surviving a failure would be attached to
        # every later line on this thread, and those reports look correlated.
        with pytest.raises(RuntimeError), request_context("req-1"):
            raise RuntimeError("boom")
        assert current_request_id() is None

    def test_nesting_restores_the_outer_id(self) -> None:
        with request_context("outer"):
            with request_context("inner"):
                assert current_request_id() == "inner"
            assert current_request_id() == "outer"

    def test_an_unbound_context_reports_no_id(self) -> None:
        assert current_request_id() is None
