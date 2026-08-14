"""Keep documented evidence from becoming a broken image.

``docs/demo.md`` named ``docs/assets/production-rag-demo.gif`` for as long as the
file did not exist. Nothing failed, because no test looked. These tests make an
asset reference a claim like any other: if the documentation points at a
capture, the capture is in the repository, and it is small enough that a
reviewer on a slow connection still sees it.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
ASSETS = ROOT / "docs" / "assets"
DEMO_ASSET = ASSETS / "production-rag-demo.gif"
SOCIAL_PREVIEW = ASSETS / "social-preview.png"
# GitHub renders a social preview at this size and crops anything else.
SOCIAL_PREVIEW_SIZE = (1280, 640)
# Generous enough that a legitimate recording fits, tight enough that an
# accidental full-page or full-colour export fails here instead of on a reviewer's
# connection.
MAX_ASSET_BYTES = 5_000_000
REFERENCE = re.compile(r"[(`\s\"]((?:\.\./)*(?:docs/)?assets/[A-Za-z0-9._/-]+)")


def _tracked_markdown() -> list[Path]:
    git = shutil.which("git")
    if git is None:
        pytest.fail("git executable is required to enumerate tracked Markdown")
    output = subprocess.run(  # noqa: S603 - resolved git executable and fixed arguments
        [git, "ls-files", "-z", "--", "*.md"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout
    return [ROOT / path.replace("\\", "/") for path in output.decode().split("\0") if path]


def _resolve(document: Path, reference: str) -> Path | None:
    """Resolve a documentation asset reference the way a reader would."""
    for candidate in ((document.parent / reference).resolve(), (ROOT / reference).resolve()):
        if candidate.exists():
            return candidate
    return None


def test_every_referenced_asset_exists() -> None:
    missing: list[str] = []
    for document in _tracked_markdown():
        text = document.read_text(encoding="utf-8")
        for match in REFERENCE.finditer(text):
            reference = match.group(1)
            if _resolve(document, reference) is None:
                missing.append(f"{document.relative_to(ROOT).as_posix()} -> {reference}")
    assert missing == [], "documented assets that do not exist:\n" + "\n".join(missing)


def test_the_demo_recording_is_committed_and_referenced() -> None:
    assert DEMO_ASSET.exists(), "docs/demo.md documents a recording that must be committed"
    referenced_in = {
        document.relative_to(ROOT).as_posix()
        for document in _tracked_markdown()
        if "production-rag-demo.gif" in document.read_text(encoding="utf-8")
    }
    assert "README.md" in referenced_in
    assert "docs/demo.md" in referenced_in


def _png_size(path: Path) -> tuple[int, int]:
    """Read a PNG's dimensions from its IHDR chunk, without an image library."""
    header = path.read_bytes()[:24]
    assert header[:8] == b"\x89PNG\r\n\x1a\n", f"{path.name} is not a PNG"
    return int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big")


def test_the_social_preview_is_committed_at_the_size_github_renders() -> None:
    assert SOCIAL_PREVIEW.exists(), "scripts/make_social_preview.py has not been run"
    assert _png_size(SOCIAL_PREVIEW) == SOCIAL_PREVIEW_SIZE


@pytest.mark.parametrize("asset", sorted(ASSETS.iterdir()))
def test_committed_assets_stay_small_enough_to_load(asset: Path) -> None:
    size = asset.stat().st_size
    assert size <= MAX_ASSET_BYTES, f"{asset.name} is too large to serve in a README"
