r"""Rebuild the credential-free UI evidence in ``docs/assets``.

Run ``pip install -e \".[docs]\" && playwright install chromium`` once, then
``python scripts/capture_ui.py``. The script owns a dedicated Qdrant collection,
uses only fake providers, and deliberately stops Qdrant to capture the real
service-failure rendering before restoring the stack.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from playwright.sync_api import Page

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
BASE_URL = "http://127.0.0.1:8000"
COLLECTION = "production_rag_ui_capture"
GROUNDED_QUESTION = "Why does hybrid search use reciprocal rank fusion?"
REFUSAL_QUESTION = "Who won the Antarctic underwater chess championship?"
FILTER_FIELD = "tags"
FILTER_VALUE = "hybrid"
"""The narrowing captured in ``ui-filtered.png``.

Chosen against ``data/raw/sample`` rather than invented: every citation the
grounded question returns under this filter comes from ``01-hybrid-search.md``,
so the capture shows a filter that visibly did something. ``source`` is the
wrong field for this corpus — ingesting ``data/raw/sample`` makes it the corpus
root, so every point carries ``source: root`` and the filter would narrow
nothing."""
CAPTURES = {
    "grounded": ASSETS / "ui-grounded.png",
    "filtered": ASSETS / "ui-filtered.png",
    "refused": ASSETS / "ui-refusal.png",
    "stream": ASSETS / "ui-stream.png",
    "error": ASSETS / "ui-service-failure.png",
}


def _run(*args: str, env: dict[str, str]) -> None:
    subprocess.run(args, cwd=ROOT, env=env, check=True)  # noqa: S603


def _wait_for_api(timeout: float = 120.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{BASE_URL}/health", timeout=2) as response:  # noqa: S310
                if response.status == 200:
                    return
        except OSError:
            time.sleep(1)
    raise RuntimeError(f"API did not become healthy within {timeout:.0f} seconds")


def _submit(
    page: Page,
    question: str,
    outcome: str,
    *,
    filter_field: str = "",
    filter_value: str = "",
    stream: bool = False,
) -> None:
    page.goto(BASE_URL, wait_until="networkidle")
    # The stream toggle must be in every still: a hiring manager needs to see
    # that streaming is an opt-in on the same form, not a separate demo app.
    page.locator("#stream").wait_for(state="visible")
    page.locator("#question").fill(question)
    if filter_field:
        # Set through the controls a reviewer uses, so the capture shows the
        # filter that produced the answer rather than only its result.
        page.locator("#filter-field").select_option(filter_field)
        page.locator("#filter-value").fill(filter_value)
    if stream:
        page.locator("#stream").check()
        page.evaluate(
            """async ({question, filterField, filterValue}) => {
                const body = new URLSearchParams({
                    question,
                    filter_field: filterField,
                    filter_value: filterValue,
                });
                const result = document.querySelector('#result');
                const response = await fetch('/ui/query/stream', {
                    method: 'POST',
                    headers: {'content-type': 'application/x-www-form-urlencoded'},
                    body,
                });
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';
                while (true) {
                    const {value, done} = await reader.read();
                    if (done) break;
                    buffer += decoder.decode(value, {stream: true});
                    const frames = buffer.split('\\n\\n');
                    buffer = frames.pop() || '';
                    for (const frame of frames) {
                        let name = '';
                        let data = '';
                        for (const line of frame.split('\\n')) {
                            if (line.startsWith('event: ')) name = line.slice(7);
                            else if (line.startsWith('data: ')) data = line.slice(6);
                        }
                        if (name === 'delta' && data) {
                            const payload = JSON.parse(data);
                            if (!result.querySelector('[data-outcome="drafting"]')) {
                                const open =
                                    '<article class="outcome outcome-draft" '
                                    + 'data-outcome="drafting">';
                                result.innerHTML = open
                                    + '<p class="outcome-label">Draft · not verified</p>'
                                    + '<h2>The model is still writing</h2>'
                                    + '<div class="draft-text"></div></article>';
                            }
                            const draft = result.querySelector('.draft-text');
                            if (draft) draft.textContent += payload.text || '';
                        } else if (name === 'result' && data) {
                            const payload = JSON.parse(data);
                            result.innerHTML = payload.html || '';
                        }
                    }
                }
            }""",
            {
                "question": question,
                "filterField": filter_field,
                "filterValue": filter_value,
            },
        )
    else:
        page.locator("#stream").uncheck()
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
            {
                "question": question,
                "filterField": filter_field,
                "filterValue": filter_value,
            },
        )
    page.locator(f'[data-outcome="{outcome}"]').wait_for(state="visible")


def _stabilize_dynamic_diagnostics(page: Page) -> None:
    """Keep screenshots byte-stable without replacing any pipeline outcome."""
    page.locator(".diagnostics > span code").evaluate("node => node.textContent = 'capture-fixed'")
    timings = page.locator(".diagnostics dd")
    for index in range(timings.count()):
        timings.nth(index).evaluate("node => node.textContent = 'measured'")


def _open_timings_if_present(page: Page) -> None:
    """Expand pipeline timings when the fragment includes them (debug on)."""
    details = page.locator(".diagnostics details")
    if details.count() > 0:
        details.first.evaluate("node => { node.open = true }")


def _capture(page: Page, name: str) -> None:
    _stabilize_dynamic_diagnostics(page)
    page.screenshot(path=CAPTURES[name], full_page=True, animations="disabled", caret="hide")


def _print_hashes() -> None:
    for name, path in CAPTURES.items():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{name}: {digest}  {path.relative_to(ROOT)}")


def capture(*, keep_stack: bool) -> None:
    """Start the fake-provider stack and capture all documented UI outcomes."""
    try:
        from playwright.sync_api import sync_playwright
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
            # Captures must exercise the live pipeline. Compose defaults
            # CACHE_ENABLED=true for demos; a hit after Qdrant is stopped would
            # forge a grounded still instead of the dependency-failure render.
            "CACHE_ENABLED": "false",
        }
    )
    ASSETS.mkdir(parents=True, exist_ok=True)
    try:
        _run("docker", "compose", "up", "-d", "--build", "qdrant", env=env)
        # Rebuild api every capture run: templates/static live in the image, so a
        # stale layer would document yesterday's UI without failing any hash.
        _run("docker", "compose", "build", "api", env=env)
        _run(
            "docker",
            "compose",
            "run",
            "--rm",
            "api",
            "python",
            "-m",
            "production_rag.ingest",
            "--config",
            "configs/default.yaml",
            "--source",
            "data/raw/sample",
            "--embedder",
            "fake",
            "--collection",
            COLLECTION,
            "--recreate-collection",
            env=env,
        )
        _run("docker", "compose", "up", "-d", "api", env=env)
        _wait_for_api()

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1100}, device_scale_factor=1)

            _submit(page, GROUNDED_QUESTION, "grounded")
            _open_timings_if_present(page)
            if not page.locator("#stream").is_visible():
                raise RuntimeError("stream toggle must be visible in grounded capture")
            _capture(page, "grounded")

            _submit(
                page,
                GROUNDED_QUESTION,
                "grounded",
                filter_field=FILTER_FIELD,
                filter_value=FILTER_VALUE,
            )
            if not page.locator("#stream").is_visible():
                raise RuntimeError("stream toggle must be visible in filtered capture")
            _capture(page, "filtered")

            _submit(page, REFUSAL_QUESTION, "refused")
            if "score_threshold: 0.0" not in (ROOT / "configs/default.yaml").read_text():
                raise RuntimeError(
                    "refusal evidence requires configs/default.yaml score_threshold: 0.0"
                )
            if not page.locator("#stream").is_visible():
                raise RuntimeError("stream toggle must be visible in refusal capture")
            _capture(page, "refused")

            # Stream beat: same grounded question, toggle on. Terminal fragment is
            # the grounded outcome; the still shows the checked toggle so a
            # reviewer sees streaming without a second app.
            _submit(page, GROUNDED_QUESTION, "grounded", stream=True)
            _open_timings_if_present(page)
            if not page.locator("#stream").is_checked():
                raise RuntimeError("stream capture requires the stream toggle checked")
            _capture(page, "stream")

            _run("docker", "compose", "stop", "qdrant", env=env)
            _submit(page, GROUNDED_QUESTION, "error")
            _capture(page, "error")
            browser.close()
        _print_hashes()
    finally:
        if not keep_stack:
            _run("docker", "compose", "down", env=env)


def main() -> int:
    """Parse CLI arguments and return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep-stack", action="store_true", help="leave Compose services running")
    args = parser.parse_args()
    try:
        capture(keep_stack=args.keep_stack)
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"capture failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
