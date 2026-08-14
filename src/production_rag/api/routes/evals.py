"""Local free-path scorecard page. Not a hosted metrics product.

Serves the committed human-readable scorecard so a reviewer on the demo stack
can open ``/evals`` without hunting the repo tree. Labels in the HTML state
contract/plumbing, ``billed=false``, ``n``, and not-SOTA — this route does not
recompute numbers or invent a tariff.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse, HTMLResponse

router = APIRouter(tags=["evals"], include_in_schema=False)

# Repo layout: src/production_rag/api/routes/evals.py → parents[2] is package root.
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_CANDIDATES = (
    Path.cwd() / "docs" / "assets" / "scorecard.html",
    _PACKAGE_ROOT.parents[1] / "docs" / "assets" / "scorecard.html",
    Path("/app/docs/assets/scorecard.html"),
)


def _scorecard_path() -> Path | None:
    for candidate in _CANDIDATES:
        if candidate.is_file():
            return candidate
    return None


@router.get("/evals", response_class=HTMLResponse)
def evals_scorecard() -> FileResponse:
    """Serve the free-path scorecard HTML committed under docs/assets/."""
    path = _scorecard_path()
    if path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "scorecard.html is not present in this checkout. "
                "Run: python tools/render_scorecard_html.py"
            ),
        )
    return FileResponse(path, media_type="text/html; charset=utf-8")


__all__ = ["router"]
