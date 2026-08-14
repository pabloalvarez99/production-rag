"""Unit tests for ``POST /v1/query/stream``.

The route is asserted at the byte level rather than by parsing its own output
with its own encoder. A streamed endpoint is a wire format, its consumers are
written against those bytes, and a test that decodes with the same helper the
route encodes with would pass through any framing change — including one that
no browser can read.

Every executor here is a double, so nothing touches a store or a provider. What
is under test is the transport's promises: the request id arrives first, deltas
are provisional, exactly one terminal event closes the stream, a refusal is a
``result`` and a provider outage is an ``error``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

from fastapi import FastAPI
from fastapi.testclient import TestClient

from production_rag.api.routes import query as query_route
from production_rag.api.routes import query_stream as stream_route
from production_rag.api.schemas import CitationOut, QueryRequest, QueryResponse
from production_rag.api.sse import SSE_MEDIA_TYPE
from production_rag.config import Settings
from production_rag.generation.llm import LLMError
from production_rag.generation.streaming import DeltaSink
from production_rag.retrieval.filters import FILTER_NOT_ALLOWED, FilterError

REQUEST_ID = "test-stream-123"


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
        self.calls.append(payload)
        for chunk in self.chunks:
            if on_delta is not None:
                on_delta(chunk)
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response


def _grounded() -> QueryResponse:
    return QueryResponse(
        answer="RRF fuses rank positions [1].",
        citations=[
            CitationOut(
                marker=1,
                chunk_id="chunk-rrf",
                source_path="sample/01-hybrid-search.md",
                text="RRF combines ranked lists without calibrating score scales.",
                rank=1,
                title="Hybrid search",
                heading_path="Reciprocal rank fusion",
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


def _override(client: TestClient, executor: FakeStreamingExecutor) -> None:
    app = cast(FastAPI, client.app)
    app.dependency_overrides[query_route.get_streaming_query_executor] = lambda: executor


def _events(body: bytes) -> list[str]:
    """The event names, in order, as a client's dispatcher would see them."""
    return [
        line.removeprefix("event: ")
        for line in body.decode("utf-8").splitlines()
        if line.startswith("event: ")
    ]


class TestFraming:
    def test_the_stream_is_exact_bytes(self, client: TestClient) -> None:
        _override(
            client,
            FakeStreamingExecutor(response=_refusal(), chunks=("Reciprocal ", "rank")),
        )

        response = client.post(
            "/v1/query/stream",
            headers={"X-Request-ID": REQUEST_ID},
            json={"question": "How does hybrid retrieval fuse rankings?"},
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith(SSE_MEDIA_TYPE)
        assert response.content.startswith(
            b'event: meta\ndata: {"request_id":"test-stream-123"}\n\n'
            b'event: delta\ndata: {"text":"Reciprocal "}\n\n'
            b'event: delta\ndata: {"text":"rank"}\n\n'
            b"event: result\n"
        )

    def test_meta_comes_first_and_carries_the_request_id(self, client: TestClient) -> None:
        _override(client, FakeStreamingExecutor(response=_grounded()))

        response = client.post(
            "/v1/query/stream",
            headers={"X-Request-ID": REQUEST_ID},
            json={"question": "How does hybrid retrieval fuse rankings?"},
        )

        assert _events(response.content)[0] == "meta"
        assert response.headers["X-Request-ID"] == REQUEST_ID

    def test_proxy_buffering_is_disabled(self, client: TestClient) -> None:
        _override(client, FakeStreamingExecutor(response=_grounded()))
        response = client.post(
            "/v1/query/stream",
            headers={"X-Request-ID": REQUEST_ID},
            json={"question": "How does hybrid retrieval fuse rankings?"},
        )
        # Without this a proxy delivers the whole stream at the end, which looks
        # exactly like a slow backend and is diagnosed as one.
        assert response.headers["X-Accel-Buffering"] == "no"
        assert response.headers["Cache-Control"] == "no-cache"


class TestGroundedStream:
    def test_the_terminal_result_carries_the_citations(self, client: TestClient) -> None:
        _override(
            client,
            FakeStreamingExecutor(response=_grounded(), chunks=("RRF ", "fuses ", "rank [1].")),
        )

        response = client.post(
            "/v1/query/stream",
            headers={"X-Request-ID": REQUEST_ID},
            json={"question": "How does hybrid retrieval fuse rankings?"},
        )
        body = response.content.decode("utf-8")

        assert _events(response.content) == ["meta", "delta", "delta", "delta", "result"]
        assert '"chunk_id":"chunk-rrf"' in body
        assert '"refused":false' in body
        # Citations never travel on a delta: they do not exist until the whole
        # answer has been mapped onto the blocks that were in the prompt.
        assert '"marker"' not in body.split("event: result")[0]

    def test_the_result_reports_how_many_pieces_arrived(self, client: TestClient) -> None:
        _override(client, FakeStreamingExecutor(response=_grounded(), chunks=("a ", "b ", "c")))
        response = client.post(
            "/v1/query/stream",
            headers={"X-Request-ID": REQUEST_ID},
            json={"question": "How does hybrid retrieval fuse rankings?"},
        )
        assert '"deltas":3' in response.content.decode("utf-8")

    def test_a_provider_that_cannot_stream_is_reported_as_one_delta(
        self, client: TestClient
    ) -> None:
        _override(client, FakeStreamingExecutor(response=_grounded(), chunks=("whole answer",)))
        response = client.post(
            "/v1/query/stream",
            headers={"X-Request-ID": REQUEST_ID},
            json={"question": "How does hybrid retrieval fuse rankings?"},
        )
        assert '"deltas":1' in response.content.decode("utf-8")


class TestRefusalStream:
    def test_a_refusal_is_a_result_not_an_error(self, client: TestClient) -> None:
        _override(client, FakeStreamingExecutor(response=_refusal()))

        response = client.post(
            "/v1/query/stream",
            headers={"X-Request-ID": REQUEST_ID},
            json={"question": "How does hybrid retrieval fuse rankings?"},
        )
        body = response.content.decode("utf-8")

        assert _events(response.content) == ["meta", "result"]
        assert '"refused":true' in body
        assert '"refusal_reason":"model_abstained"' in body

    def test_a_refusal_after_deltas_still_ends_refused(self, client: TestClient) -> None:
        # The interesting shape: the model produced text, the guardrails could
        # not ground it, and the stream says so instead of serving the draft.
        _override(
            client,
            FakeStreamingExecutor(response=_refusal(), chunks=("INSUFFICIENT_CONTEXT",)),
        )

        response = client.post(
            "/v1/query/stream",
            headers={"X-Request-ID": REQUEST_ID},
            json={"question": "How does hybrid retrieval fuse rankings?"},
        )

        assert _events(response.content) == ["meta", "delta", "result"]
        assert '"refused":true' in response.content.decode("utf-8")


class TestFailures:
    def test_a_provider_outage_is_an_error_and_never_a_refusal(self, client: TestClient) -> None:
        _override(
            client,
            FakeStreamingExecutor(
                error=LLMError("generation request failed: APIError: upstream"),
                chunks=("Reciprocal ",),
            ),
        )

        response = client.post(
            "/v1/query/stream",
            headers={"X-Request-ID": REQUEST_ID},
            json={"question": "How does hybrid retrieval fuse rankings?"},
        )
        body = response.content.decode("utf-8")

        assert _events(response.content) == ["meta", "delta", "error"]
        assert f'"error_type":"{stream_route.ERROR_PROVIDER}"' in body
        assert '"refused":false' in body
        assert '"refusal_reason"' not in body

    def test_the_provider_message_is_not_forwarded_to_the_client(
        self, client: TestClient
    ) -> None:
        _override(
            client,
            FakeStreamingExecutor(error=LLMError("generation failed: key sk-secret rejected")),
        )
        response = client.post(
            "/v1/query/stream",
            headers={"X-Request-ID": REQUEST_ID},
            json={"question": "How does hybrid retrieval fuse rankings?"},
        )
        assert "sk-secret" not in response.content.decode("utf-8")

    def test_a_missing_pipeline_is_typed_rather_than_generic(self, client: TestClient) -> None:
        _override(
            client,
            FakeStreamingExecutor(
                error=query_route.QueryPipelineUnavailableError("query pipeline not installed")
            ),
        )
        response = client.post(
            "/v1/query/stream",
            headers={"X-Request-ID": REQUEST_ID},
            json={"question": "How does hybrid retrieval fuse rankings?"},
        )
        assert f'"error_type":"{stream_route.ERROR_PIPELINE_UNAVAILABLE}"' in (
            response.content.decode("utf-8")
        )

    def test_an_unclassified_failure_stays_generic(self, client: TestClient) -> None:
        _override(client, FakeStreamingExecutor(error=RuntimeError("collection is missing")))
        response = client.post(
            "/v1/query/stream",
            headers={"X-Request-ID": REQUEST_ID},
            json={"question": "How does hybrid retrieval fuse rankings?"},
        )
        body = response.content.decode("utf-8")
        assert f'"error_type":"{stream_route.ERROR_INTERNAL}"' in body
        assert "collection is missing" not in body

    def test_the_stream_ends_after_a_terminal_event(self, client: TestClient) -> None:
        _override(client, FakeStreamingExecutor(error=RuntimeError("boom"), chunks=("a ",)))
        response = client.post(
            "/v1/query/stream",
            headers={"X-Request-ID": REQUEST_ID},
            json={"question": "How does hybrid retrieval fuse rankings?"},
        )
        names = _events(response.content)
        assert names.count("error") + names.count("result") == 1
        assert names[-1] == "error"


class TestFilters:
    def test_a_rejected_filter_is_a_422_because_nothing_was_streamed(
        self, client: TestClient
    ) -> None:
        _override(client, FakeStreamingExecutor(response=_grounded()))

        response = client.post(
            "/v1/query/stream",
            headers={"X-Request-ID": REQUEST_ID},
            json={
                "question": "How does hybrid retrieval fuse rankings?",
                "filters": {"author": "someone"},
            },
        )

        assert response.status_code == 422
        assert response.json()["detail"]["error_type"] == FILTER_NOT_ALLOWED
        assert response.headers["content-type"].startswith("application/json")

    def test_an_allowlisted_filter_reaches_the_executor(self, client: TestClient) -> None:
        executor = FakeStreamingExecutor(response=_grounded())
        _override(client, executor)

        response = client.post(
            "/v1/query/stream",
            headers={"X-Request-ID": REQUEST_ID},
            json={
                "question": "How does hybrid retrieval fuse rankings?",
                "filters": {"title": "Filtering"},
            },
        )

        assert response.status_code == 200
        assert executor.calls[0].filters == {"title": "Filtering"}

    def test_a_filter_rejected_inside_the_executor_is_still_typed(
        self, client: TestClient
    ) -> None:
        # Reachable when the executor validates against a profile the route did
        # not see. The stream is already open, so it is an event, not a status.
        _override(
            client,
            FakeStreamingExecutor(
                error=FilterError("unknown field", field="title", error_type=FILTER_NOT_ALLOWED)
            ),
        )
        response = client.post(
            "/v1/query/stream",
            headers={"X-Request-ID": REQUEST_ID},
            json={"question": "How does hybrid retrieval fuse rankings?"},
        )
        assert f'"error_type":"{FILTER_NOT_ALLOWED}"' in response.content.decode("utf-8")


class TestContract:
    def test_the_json_route_is_untouched_by_the_stream_route(self, client: TestClient) -> None:
        app = cast(FastAPI, client.app)
        paths = app.openapi()["paths"]
        assert "/v1/query" in paths
        assert "/v1/query/stream" in paths
        # Same request body model on both, so a client that can call one can
        # call the other without learning a second schema.
        assert (
            paths["/v1/query"]["post"]["requestBody"]
            == paths["/v1/query/stream"]["post"]["requestBody"]
        )

    def test_the_stream_route_advertises_the_sse_media_type(self, client: TestClient) -> None:
        app = cast(FastAPI, client.app)
        responses = app.openapi()["paths"]["/v1/query/stream"]["post"]["responses"]
        assert SSE_MEDIA_TYPE in responses["200"]["content"]

    def test_an_empty_question_is_rejected_before_the_stream_opens(
        self, client: TestClient
    ) -> None:
        _override(client, FakeStreamingExecutor(response=_grounded()))
        response = client.post("/v1/query/stream", json={"question": "   "})
        assert response.status_code == 422
