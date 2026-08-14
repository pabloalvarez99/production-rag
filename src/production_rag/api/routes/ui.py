"""Server-rendered query UI built on the public in-process query adapter."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TypedDict
from urllib.parse import parse_qs

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from production_rag.api.deps import SettingsDep
from production_rag.api.middleware import get_request_id
from production_rag.api.routes.query import QueryExecutorDep, filter_policy
from production_rag.api.schemas import QueryRequest, QueryResponse
from production_rag.config import Settings
from production_rag.config_loader import ConfigFileError, load_yaml_config
from production_rag.retrieval.filters import FilterError, FilterPolicy

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


def _filter_policy(settings: Settings) -> FilterPolicy:
    """The same allowlist POST /v1/query enforces, for this deployment.

    Read from configuration rather than hard-coded in the template, so the field
    the UI offers and the field the API accepts cannot drift apart. A profile
    that cannot be parsed yields an empty policy: the control disappears and any
    filter posted by hand is rejected, which is the fail-closed direction.
    """
    try:
        config = load_yaml_config(settings.config_path)
    except ConfigFileError:
        return FilterPolicy()
    return filter_policy(config)


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


@router.get("/", response_class=HTMLResponse)
def index(request: Request, settings: SettingsDep) -> HTMLResponse:
    """Render the offline-first query form."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"filter_fields": _filter_policy(settings).filterable_fields},
    )


@router.post("/ui/query", response_class=HTMLResponse)
async def submit_query(
    request: Request,
    settings: SettingsDep,
    executor: QueryExecutorDep,
) -> HTMLResponse:
    """Run the same in-process query adapter as POST /v1/query and render it."""
    request_id = get_request_id(request)
    form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
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
            _filter_policy(settings).build(payload.filters)
        except FilterError as exc:
            return _filter_error_response(request, exc, request_id)

    try:
        result = await run_in_threadpool(
            executor,
            payload,
            settings=settings,
            request_id=request_id,
        )
    except FilterError as exc:
        return _filter_error_response(request, exc, request_id)
    except Exception as exc:  # noqa: BLE001 - the HTML boundary must remain renderable
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

    return templates.TemplateResponse(
        request=request,
        name="result.html",
        context={
            "state": "refused" if result.refused else "grounded",
            "result": result,
            "segments": _answer_segments(result),
            "request_id": request_id,
            # Echoed back so an answer can never be read as unfiltered when it
            # was narrowed. Absent means nothing was asked for.
            "applied_filter": (
                f"{field} = {payload.filters[field]}" if payload.filters else None
            ),
        },
    )


__all__ = ["STATIC_DIR", "router"]
