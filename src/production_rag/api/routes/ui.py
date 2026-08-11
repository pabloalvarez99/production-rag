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
from production_rag.api.routes.query import QueryExecutorDep
from production_rag.api.schemas import QueryRequest, QueryResponse

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


@router.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    """Render the offline-first query form."""
    return templates.TemplateResponse(request=request, name="index.html")


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
    try:
        payload = QueryRequest(question=question, llm="fake", debug=True)
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

    try:
        result = await run_in_threadpool(
            executor,
            payload,
            settings=settings,
            request_id=request_id,
        )
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
        },
    )


__all__ = ["STATIC_DIR", "router"]
