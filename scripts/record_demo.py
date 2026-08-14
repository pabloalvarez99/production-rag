r"""Record the animated free-path demo referenced by ``docs/demo.md``.

Run ``pip install -e \".[docs]\" && playwright install chromium`` once, then
``python scripts/record_demo.py``. The script reuses the capture pipeline in
``capture_ui.py``: the same fake providers, the same dedicated collection, the
same stabilised diagnostics. It drives the real UI at a fixed viewport, saves
one PNG per beat, and hands the sequence to ffmpeg.

The output is a slideshow of real renderings rather than a screen recording:
no cursor, no window chrome, no desktop behind it, and identical app inputs
produce an identical file.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from capture_ui import (
    COLLECTION,
    FILTER_FIELD,
    FILTER_VALUE,
    GROUNDED_QUESTION,
    REFUSAL_QUESTION,
    _run,
    _stabilize_dynamic_diagnostics,
    _wait_for_api,
)

if TYPE_CHECKING:
    from playwright.sync_api import Page

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
OUTPUT = ASSETS / "production-rag-demo.gif"
BASE_URL = "http://127.0.0.1:8000"
VIEWPORT = {"width": 1280, "height": 720}

# One entry per frame: (basename, seconds on screen). The grounded answer and
# the refusal hold longest because they are what the reviewer is asked to read;
# the typing beats exist only so the two questions are legibly different.
STORYBOARD: tuple[tuple[str, float], ...] = (
    ("01-empty", 1.6),
    ("02-typing-grounded", 1.2),
    ("03-grounded", 3.6),
    ("04-grounded-timings", 3.0),
    ("05-filter-set", 1.6),
    ("06-filtered", 3.4),
    ("07-typing-refusal", 1.2),
    ("08-refused", 3.8),
)


def _submit(
    page: Page,
    question: str,
    outcome: str,
    *,
    filter_field: str = "",
    filter_value: str = "",
) -> None:
    """Submit through the real form handler and wait for the typed outcome."""
    page.evaluate(
        """async ({question, filterField, filterValue}) => {
            const body = new URLSearchParams({
                question,
                filter_field: filterField,
                filter_value: filterValue,
            });
            const response = await fetch('/ui/query', {
                method: 'POST',
                headers: {'content-type': 'application/x-www-form-urlencoded'},
                body,
            });
            document.querySelector('#result').innerHTML = await response.text();
        }""",
        {"question": question, "filterField": filter_field, "filterValue": filter_value},
    )
    page.locator(f'[data-outcome="{outcome}"]').wait_for(state="visible")


def _shoot(page: Page, frames: Path, name: str) -> None:
    page.screenshot(path=frames / f"{name}.png", animations="disabled", caret="hide")


def _focus(page: Page, selector: str) -> None:
    """Put an element at the top of the viewport and let the scroll settle.

    The answer renders below the fold at this viewport, so a result frame shot
    from the top of the page is a picture of the empty form. Every frame after a
    submission scrolls to what it is meant to show.
    """
    page.locator(selector).evaluate(
        "node => node.scrollIntoView({block: 'start', behavior: 'instant'})"
    )
    page.wait_for_timeout(150)


def _record_frames(frames: Path) -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=1)
        page.goto(BASE_URL, wait_until="networkidle")

        question = page.locator("#question")
        _shoot(page, frames, "01-empty")

        # A prefix, not a keystroke animation: enough to show the question being
        # entered without paying twenty frames for it.
        question.fill(GROUNDED_QUESTION[: len(GROUNDED_QUESTION) // 2])
        _shoot(page, frames, "02-typing-grounded")

        question.fill(GROUNDED_QUESTION)
        _submit(page, GROUNDED_QUESTION, "grounded")
        _stabilize_dynamic_diagnostics(page)
        _focus(page, "#result")
        _shoot(page, frames, "03-grounded")

        page.locator(".diagnostics details").evaluate("details => details.open = true")
        page.locator(".diagnostics details[open]").wait_for(state="visible")
        _stabilize_dynamic_diagnostics(page)
        _focus(page, ".diagnostics")
        _shoot(page, frames, "04-grounded-timings")

        # The same question again, narrowed. Shown before the refusal so the
        # recording still ends on the refusal, which is the beat that carries.
        _focus(page, "#question")
        page.locator("#filter-field").select_option(FILTER_FIELD)
        page.locator("#filter-value").fill(FILTER_VALUE)
        _shoot(page, frames, "05-filter-set")

        _submit(
            page,
            GROUNDED_QUESTION,
            "grounded",
            filter_field=FILTER_FIELD,
            filter_value=FILTER_VALUE,
        )
        _stabilize_dynamic_diagnostics(page)
        _focus(page, "#result")
        _shoot(page, frames, "06-filtered")

        _focus(page, "#question")
        page.locator("#filter-field").select_option("")
        page.locator("#filter-value").fill("")
        question.fill(REFUSAL_QUESTION[: len(REFUSAL_QUESTION) // 2])
        _shoot(page, frames, "07-typing-refusal")

        question.fill(REFUSAL_QUESTION)
        _submit(page, REFUSAL_QUESTION, "refused")
        _stabilize_dynamic_diagnostics(page)
        _focus(page, "#result")
        _shoot(page, frames, "08-refused")

        browser.close()


def _assemble(frames: Path, output: Path, width: int) -> None:
    """Turn the frame sequence into one palette-optimised GIF."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to assemble the demo; install it and re-run")

    concat = frames / "storyboard.txt"
    lines: list[str] = []
    for name, seconds in STORYBOARD:
        frame = frames / f"{name}.png"
        if not frame.exists():
            raise RuntimeError(f"storyboard frame is missing: {frame.name}")
        lines.append(f"file '{frame.as_posix()}'")
        lines.append(f"duration {seconds}")
    # The concat demuxer ignores the final entry's duration, so the last frame is
    # repeated to give the refusal its full time on screen before the loop.
    lines.append(f"file '{(frames / f'{STORYBOARD[-1][0]}.png').as_posix()}'")
    concat.write_text("\n".join(lines) + "\n", encoding="utf-8")

    palette = frames / "palette.png"
    scale = f"scale={width}:-2:flags=lanczos"
    subprocess.run(  # noqa: S603 - resolved executable, generated file arguments
        [ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(concat),
         "-vf", f"{scale},palettegen=max_colors=128:stats_mode=diff", str(palette)],
        check=True,
    )
    subprocess.run(  # noqa: S603 - resolved executable, generated file arguments
        [ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(concat),
         "-i", str(palette), "-lavfi",
         f"{scale}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
         "-loop", "0", str(output)],
        check=True,
    )


def _report(output: Path) -> None:
    data = output.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    print(f"{output.relative_to(ROOT)}: {len(data) / 1_000_000:.2f} MB  sha256 {digest}")


def record(*, keep_stack: bool, width: int, output: Path) -> None:
    """Start the fake-provider stack, record the beats, and assemble the GIF."""
    try:
        import playwright.sync_api  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            'Playwright is documentation-only; install it with pip install -e ".[docs]"'
        ) from exc

    env = os.environ.copy()
    env.update(
        {
            "QDRANT_COLLECTION": COLLECTION,
            "OPENAI_API_KEY": "",
            "COHERE_API_KEY": "",
            "QDRANT_API_KEY": "",
        }
    )
    ASSETS.mkdir(parents=True, exist_ok=True)
    try:
        _run("docker", "compose", "up", "-d", "--build", "qdrant", env=env)
        _run(
            "docker", "compose", "run", "--rm", "api",
            "python", "-m", "production_rag.ingest",
            "--config", "configs/default.yaml",
            "--source", "data/raw/sample",
            "--embedder", "fake",
            "--collection", COLLECTION,
            "--recreate-collection",
            env=env,
        )
        _run("docker", "compose", "up", "-d", "api", env=env)
        _wait_for_api()

        with tempfile.TemporaryDirectory(prefix="production-rag-demo-") as scratch:
            frames = Path(scratch)
            _record_frames(frames)
            _assemble(frames, output, width)
        _report(output)
    finally:
        if not keep_stack:
            _run("docker", "compose", "down", env=env)


def main() -> int:
    """Parse CLI arguments and return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep-stack", action="store_true", help="leave Compose services running")
    parser.add_argument("--width", type=int, default=960, help="output width in pixels")
    parser.add_argument("--output", type=Path, default=OUTPUT, help="destination GIF")
    args = parser.parse_args()
    try:
        record(keep_stack=args.keep_stack, width=args.width, output=args.output)
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"recording failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
