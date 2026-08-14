"""Unit tests for the streamed demo page, ``POST /ui/query/stream``.

The page and the JSON stream answer to different promises. This route's
terminal event carries **rendered HTML** produced by the same template the
swapped path uses, so the tests here are mostly about that identity: what a
reviewer sees must not depend on which toggle was set.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import cast

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from production_rag.api.routes import query as query_route
from production_rag.api.schemas import CitationOut, QueryRequest, QueryResponse
from production_rag.api.sse import SSE_MEDIA_TYPE
from production_rag.config import Settings
from production_rag.generation.llm import LLMError
from production_rag.generation.streaming import DeltaSink
from production_rag.retrieval.filters import FILTER_NOT_ALLOWED, FilterError

FORM_HEADERS = {"content-type": "application/x-www-form-urlencoded"}
QUESTION = "Why does hybrid search use reciprocal rank fusion?"


@dataclass
class FakeStreamingExecutor:
    """Publishes a fixed script of chunks, then returns a fixed outcome."""

    response: QueryResponse | None = None
    error: Exception | None = None
    chunks: tuple[str, ...] = ()
    calls: list[QueryRequest] = field(default_factory=list)

    def __call__(
        self,
        payload: QueryRequest,
        *,
        settings: Settings,
        request_id: str,
        on_delta: DeltaSink | None = None,
    ) -> QueryResponse:
        del settings, request_id
        self.calls.append(payload)
        for chunk in self.chunks:
            if on_delta is not None:
                on_delta(chunk)
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response


def _override(client: TestClient, executor: FakeStreamingExecutor) -> None:
    app = cast(FastAPI, client.app)
    app.dependency_overrides[query_route.get_streaming_query_executor] = lambda: executor


def _grounded() -> QueryResponse:
    return QueryResponse(
        answer="RRF combines ranked results [1].",
        citations=[
            CitationOut(
                marker=1,
                chunk_id="rrf",
                source_path="sample/hybrid.md",
                text="RRF combines ranked result lists.",
                rank=1,
            )
        ],
        refused=False,
        refusal_reason=None,
    )


def _refusal() -> QueryResponse:
    return QueryResponse(
        answer="I could not find support for that in the indexed documents.",
        citations=[],
        refused=True,
        refusal_reason="model_abstained",
    )


def _post(client: TestClient, body: str = f"question={QUESTION}") -> httpx.Response:
    # Cast: some envs type TestClient responses as httpx2 while stubs say httpx.
    return cast(
        httpx.Response,
        client.post("/ui/query/stream", headers=FORM_HEADERS, content=body),
    )


def _frames(body: bytes) -> list[tuple[str, dict[str, str]]]:
    """Every complete event as ``(name, payload)``."""
    events: list[tuple[str, dict[str, str]]] = []
    for frame in body.decode("utf-8").split("\n\n"):
        name = ""
        data = ""
        for line in frame.splitlines():
            if line.startswith("event: "):
                name = line.removeprefix("event: ")
            elif line.startswith("data: "):
                data = line.removeprefix("data: ")
        if name and data:
            events.append((name, json.loads(data)))
    return events


class TestStreamedPage:
    def test_deltas_arrive_before_the_rendered_fragment(self, client: TestClient) -> None:
        _override(
            client,
            FakeStreamingExecutor(response=_grounded(), chunks=("RRF ", "combines ", "results")),
        )

        response = _post(client)
        names = [name for name, _ in _frames(response.content)]

        assert response.status_code == 200
        assert response.headers["content-type"].startswith(SSE_MEDIA_TYPE)
        assert names == ["meta", "delta", "delta", "delta", "result"]

    def test_the_terminal_event_is_the_same_fragment_the_swap_path_renders(
        self, client: TestClient
    ) -> None:
        streamed_executor = FakeStreamingExecutor(response=_grounded(), chunks=("RRF ",))
        _override(client, streamed_executor)
        app = cast(FastAPI, client.app)
        app.dependency_overrides[query_route.get_query_executor] = lambda: streamed_executor

        swapped = client.post("/ui/query", headers=FORM_HEADERS, content=f"question={QUESTION}")
        streamed = _post(client)
        events = dict(_frames(streamed.content))

        # Request ids differ per request; everything else must be identical.
        assert _without_request_id(events["result"]["html"]) == _without_request_id(swapped.text)

    def test_a_refusal_replaces_the_draft(self, client: TestClient) -> None:
        _override(
            client,
            FakeStreamingExecutor(response=_refusal(), chunks=("INSUFFICIENT_CONTEXT",)),
        )

        events = dict(_frames(_post(client).content))

        assert 'data-outcome="refused"' in events["result"]["html"]
        # The sentinel was streamed as a draft and is not in what replaces it.
        assert "INSUFFICIENT_CONTEXT" not in events["result"]["html"]

    def test_a_pipeline_failure_is_rendered_as_a_fragment_not_an_error_event(
        self, client: TestClient
    ) -> None:
        _override(client, FakeStreamingExecutor(error=LLMError("provider down")))

        events = _frames(_post(client).content)
        names = [name for name, _ in events]

        assert names == ["meta", "result"]
        assert 'data-outcome="error"' in dict(events)["result"]["html"]

    def test_an_empty_store_keeps_its_own_rendering(self, client: TestClient) -> None:
        _override(
            client,
            FakeStreamingExecutor(error=RuntimeError("collection production_rag not found")),
        )
        events = dict(_frames(_post(client).content))
        assert 'data-outcome="empty-store"' in events["result"]["html"]


class TestRejectionsBeforeTheStream:
    def test_a_blank_question_is_a_422_fragment(self, client: TestClient) -> None:
        _override(client, FakeStreamingExecutor(response=_grounded()))

        response = client.post("/ui/query/stream", headers=FORM_HEADERS, content="question=   ")

        assert response.status_code == 422
        assert response.headers["content-type"].startswith("text/html")
        assert 'data-outcome="error"' in response.text

    def test_a_rejected_filter_is_a_422_fragment(self, client: TestClient) -> None:
        _override(client, FakeStreamingExecutor(response=_grounded()))

        response = client.post(
            "/ui/query/stream",
            headers=FORM_HEADERS,
            content=f"question={QUESTION}&filter_field=author&filter_value=someone",
        )

        assert response.status_code == 422
        assert 'data-outcome="invalid-filter"' in response.text
        assert FILTER_NOT_ALLOWED in response.text

    def test_a_filter_rejected_inside_the_executor_is_still_a_fragment(
        self, client: TestClient
    ) -> None:
        _override(
            client,
            FakeStreamingExecutor(
                error=FilterError("unknown field", field="author", error_type=FILTER_NOT_ALLOWED)
            ),
        )
        events = dict(_frames(_post(client).content))
        assert 'data-outcome="invalid-filter"' in events["result"]["html"]


class TestFilters:
    def test_an_allowlisted_filter_reaches_the_executor_and_is_echoed(
        self, client: TestClient
    ) -> None:
        executor = FakeStreamingExecutor(response=_grounded())
        _override(client, executor)

        response = client.post(
            "/ui/query/stream",
            headers=FORM_HEADERS,
            content=f"question={QUESTION}&filter_field=title&filter_value=Filtering",
        )
        events = dict(_frames(response.content))

        assert executor.calls[0].filters == {"title": "Filtering"}
        assert "Filtered: title = Filtering" in events["result"]["html"]


class TestForm:
    def test_the_form_offers_a_stream_toggle(self, client: TestClient) -> None:
        page = client.get("/").text
        assert 'id="stream"' in page
        assert "/ui/query/stream" in page

    def test_the_toggle_starts_off_under_the_default_profile(self, client: TestClient) -> None:
        # `generation.stream` is false in configs/default.yaml: the first thing a
        # reviewer sees is the plain request/response path the contract documents.
        page = client.get("/").text
        assert 'id="stream" name="stream" value="on">' in page


def _without_request_id(html: str) -> str:
    """Blank the per-request correlation id so two renders can be compared."""
    import re

    return re.sub(r"<code>[0-9a-f-]{8,}</code>", "<code>request-id</code>", html)
