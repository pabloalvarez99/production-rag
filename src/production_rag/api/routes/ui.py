"""Server-rendered query UI built on the public in-process query adapter."""

from __future__ import annotations

import queue
import re
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast
from urllib.parse import parse_qs

from fastapi import APIRouter, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from production_rag.api.deps import SettingsDep
from production_rag.api.middleware import get_request_id
from production_rag.api.routes.query import (
    QueryExecutorDep,
    StreamingQueryExecutor,
    StreamingQueryExecutorDep,
    deployment_filter_policy,
)
from production_rag.api.schemas import QueryRequest, QueryResponse
from production_rag.api.sse import (
    EVENT_DELTA,
    EVENT_META,
    EVENT_RESULT,
    SSE_HEADERS,
    SSE_MEDIA_TYPE,
    SSEEvent,
)
from production_rag.config import Settings
from production_rag.config_loader import ConfigFileError, load_yaml_config
from production_rag.retrieval.filters import FilterError

PACKAGE_DIR = Path(__file__).resolve().parents[2]
TEMPLATE_DIR = PACKAGE_DIR / "templates"
STATIC_DIR = PACKAGE_DIR / "static"
_MARKER = re.compile(r"\[(\d+)\]")
_EMPTY_STORE_SIGNALS = ("collection", "not found", "does not exist", "empty", "no points")

templates = Jinja2Templates(directory=TEMPLATE_DIR)
router = APIRouter(tags=["ui"], include_in_schema=False)


class AnswerSegment(TypedDict):
    """A safe answer fragment, optionally linked to a citation marker."""

    text: str
    marker: int | None


def _answer_segments(response: QueryResponse) -> list[AnswerSegment]:
    """Split answer markers without treating model output as HTML."""
    valid_markers = {citation.marker for citation in response.citations}
    segments: list[AnswerSegment] = []
    cursor = 0
    for match in _MARKER.finditer(response.answer):
        if match.start() > cursor:
            segments.append({"text": response.answer[cursor : match.start()], "marker": None})
        marker = int(match.group(1))
        segments.append(
            {
                "text": match.group(0),
                "marker": marker if marker in valid_markers else None,
            }
        )
        cursor = match.end()
    if cursor < len(response.answer):
        segments.append({"text": response.answer[cursor:], "marker": None})
    return segments


def _looks_like_empty_store(exc: Exception) -> bool:
    """Identify the temporary absent-corpus failures Qdrant commonly reports."""
    message = str(exc).lower()
    return "collection" in message and any(signal in message for signal in _EMPTY_STORE_SIGNALS[1:])


def _filter_error_response(
    request: Request,
    exc: FilterError,
    request_id: str,
) -> HTMLResponse:
    """Render a rejected filter as the typed 422 the API returns, not a 500."""
    return templates.TemplateResponse(
        request=request,
        name="result.html",
        context={
            "state": "invalid-filter",
            "message": str(exc),
            "error_type": exc.error_type,
            "request_id": request_id,
        },
        # The same code POST /v1/query answers with: the form is well formed and
        # unanswerable as written. HTMX still swaps it, because the fragment is
        # what the reviewer needs to see.
        status_code=422,
    )


def _stream_default(settings: Settings) -> bool:
    """Whether the form's stream toggle starts on, from ``generation.stream``.

    The endpoint itself is not behind this flag. A route is a contract and a
    contract that appears and disappears with a profile is worse than no route:
    a client cannot code against it. What the profile decides is what a reviewer
    sees first, and an unreadable profile falls back to the quiet default.
    """
    try:
        return load_yaml_config(settings.config_path).generation.stream
    except ConfigFileError:
        return False


@router.get("/", response_class=HTMLResponse)
def index(request: Request, settings: SettingsDep) -> HTMLResponse:
    """Render the offline-first query form."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "filter_fields": deployment_filter_policy(settings).filterable_fields,
            "stream_default": _stream_default(settings),
        },
    )


@dataclass(frozen=True, slots=True)
class _Submission:
    """A parsed, validated form, ready to run."""

    payload: QueryRequest
    applied_filter: str | None


def _read_submission(
    request: Request,
    body: bytes,
    *,
    settings: Settings,
    request_id: str,
) -> _Submission | HTMLResponse:
    """Turn the posted form into a validated request, or into a rejection.

    Shared by the swapped and the streamed route so the two cannot disagree
    about what the form means — which is the failure a second parser would
    eventually produce: a filter honoured on one path and dropped on the other.
    """
    form = parse_qs(body.decode("utf-8"), keep_blank_values=True)
    question = form.get("question", [""])[0]
    field = form.get("filter_field", [""])[0].strip()
    # One field and one value, because that is the whole shape the control has.
    # An unselected field means unfiltered, and the value is ignored rather than
    # remembered: a query is either narrowed or it is not.
    filters: dict[str, str | list[str]] | None = (
        {field: form.get("filter_value", [""])[0]} if field else None
    )
    try:
        # The demo path is pinned to the credential-free providers explicitly: the UI
        # promises "free / deterministic", so it must not inherit a schema default that
        # could later change to a billed provider.
        payload = QueryRequest(
            question=question,
            llm="fake",
            embedder="fake",
            debug=True,
            filters=filters,
        )
    except ValidationError:
        return templates.TemplateResponse(
            request=request,
            name="result.html",
            context={
                "state": "error",
                "message": "Enter a question before running the query.",
                "request_id": request_id,
            },
            status_code=422,
        )

    if payload.filters:
        # Validated here as well as inside the executor, and with the same policy
        # object: the fake executor the tests inject does no validation, and a
        # reviewer posting an unknown field by hand must see the typed rejection
        # rather than a generic failure fragment.
        try:
            deployment_filter_policy(settings).build(payload.filters)
        except FilterError as exc:
            return _filter_error_response(request, exc, request_id)

    return _Submission(
        payload=payload,
        # Echoed back so an answer can never be read as unfiltered when it was
        # narrowed. Absent means nothing was asked for.
        applied_filter=(f"{field} = {payload.filters[field]}" if payload.filters else None),
    )


def _outcome_fragment(
    request: Request,
    result: QueryResponse,
    *,
    request_id: str,
    applied_filter: str | None,
) -> HTMLResponse:
    """Render a completed run — grounded or refused — as the result fragment."""
    return templates.TemplateResponse(
        request=request,
        name="result.html",
        context={
            "state": "refused" if result.refused else "grounded",
            "result": result,
            "segments": _answer_segments(result),
            "request_id": request_id,
            "applied_filter": applied_filter,
        },
    )


def _failure_fragment(
    request: Request,
    exc: Exception,
    *,
    request_id: str,
) -> HTMLResponse:
    """Render a pipeline failure as a fragment the page can display."""
    if isinstance(exc, FilterError):
        return _filter_error_response(request, exc, request_id)
    empty_store = _looks_like_empty_store(exc)
    return templates.TemplateResponse(
        request=request,
        name="result.html",
        context={
            "state": "empty" if empty_store else "error",
            "message": (
                "The corpus is temporarily unavailable or empty. "
                "Index the sample corpus, then try again."
                if empty_store
                else "The query could not be completed because a service dependency failed."
            ),
            "request_id": request_id,
        },
        # The fragment is a successful UI render even when the pipeline failed;
        # HTMX swaps 2xx responses by default, making the failure visible.
        status_code=200,
    )


@router.post("/ui/query", response_class=HTMLResponse)
async def submit_query(
    request: Request,
    settings: SettingsDep,
    executor: QueryExecutorDep,
) -> HTMLResponse:
    """Run the same in-process query adapter as POST /v1/query and render it."""
    request_id = get_request_id(request)
    submission = _read_submission(
        request,
        await request.body(),
        settings=settings,
        request_id=request_id,
    )
    if isinstance(submission, HTMLResponse):
        return submission

    try:
        result = await run_in_threadpool(
            executor,
            submission.payload,
            settings=settings,
            request_id=request_id,
        )
    except Exception as exc:  # noqa: BLE001 - the HTML boundary must remain renderable
        return _failure_fragment(request, exc, request_id=request_id)

    return _outcome_fragment(
        request,
        result,
        request_id=request_id,
        applied_filter=submission.applied_filter,
    )


def _iter_ui_stream(
    request: Request,
    executor: StreamingQueryExecutor,
    submission: _Submission,
    *,
    settings: Settings,
    request_id: str,
) -> Iterator[bytes]:
    """Stream provisional text, then the fragment the page should actually show.

    The terminal event carries **rendered HTML**, not JSON for the browser to
    template. Two renderers for one outcome is how a streamed answer ends up
    formatted differently from a swapped one, and eventually how a citation
    marker becomes a link on one path and plain text on the other. There is one
    template, ``result.html``, and this route sends what it produced.

    There is deliberately no ``error`` event here, unlike
    ``POST /v1/query/stream``: a failure on this channel is also a fragment —
    the same one the swapped path renders — so the page has exactly one thing to
    do when the stream ends, and cannot end up with a draft left on screen next
    to an error it did not know how to display.
    """
    payload = submission.payload
    chunks: queue.SimpleQueue[str | None] = queue.SimpleQueue()
    outcome: list[tuple[str, object]] = []

    def run() -> None:
        try:
            response = executor(
                payload,
                settings=settings,
                request_id=request_id,
                on_delta=chunks.put,
            )
        except Exception as exc:  # noqa: BLE001 - rendered as a fragment, like the swap path
            outcome.append(("error", exc))
        else:
            outcome.append(("result", response))
        finally:
            chunks.put(None)

    worker = threading.Thread(target=run, name=f"ui-stream-{request_id}", daemon=True)
    yield SSEEvent(EVENT_META, {"request_id": request_id}).encode()
    worker.start()
    while True:
        chunk = chunks.get()
        if chunk is None:
            break
        yield SSEEvent(EVENT_DELTA, {"text": chunk}).encode()
    worker.join()

    kind, value = outcome[0]
    fragment = (
        _failure_fragment(request, cast(Exception, value), request_id=request_id)
        if kind == "error"
        else _outcome_fragment(
            request,
            cast(QueryResponse, value),
            request_id=request_id,
            applied_filter=submission.applied_filter,
        )
    )
    yield SSEEvent(
        EVENT_RESULT,
        {"html": bytes(fragment.body).decode("utf-8")},
    ).encode()


@router.post("/ui/query/stream")
async def submit_query_streamed(
    request: Request,
    settings: SettingsDep,
    executor: StreamingQueryExecutorDep,
) -> Response:
    """The same query, with the model's provisional text shown while it runs.

    The page renders deltas as a visibly unverified draft and replaces them with
    the terminal fragment. That replacement is the honest part of the feature:
    an answer the guardrails decline becomes a refusal in front of the reviewer
    rather than being quietly withheld, and a reviewer who watches it happen
    understands the refusal contract better than any paragraph explains it.
    """
    request_id = get_request_id(request)
    submission = _read_submission(
        request,
        await request.body(),
        settings=settings,
        request_id=request_id,
    )
    if isinstance(submission, HTMLResponse):
        # Before the first byte, so a bad form is still a status code and the
        # page can swap the fragment exactly as the non-streaming path does.
        return submission

    return StreamingResponse(
        _iter_ui_stream(
            request,
            executor,
            submission,
            settings=settings,
            request_id=request_id,
        ),
        media_type=SSE_MEDIA_TYPE,
        headers=SSE_HEADERS,
    )


__all__ = ["STATIC_DIR", "router"]
