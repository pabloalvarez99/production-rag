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
CAPTURES = {
    "grounded": ASSETS / "ui-grounded.png",
    "refused": ASSETS / "ui-refusal.png",
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


def _submit(page: Page, question: str, outcome: str) -> None:
    page.goto(BASE_URL, wait_until="networkidle")
    page.locator("#question").fill(question)
    page.evaluate(
        """async question => {
            const body = new URLSearchParams({question});
            const response = await fetch('/ui/query', {
                method: 'POST',
                headers: {'content-type': 'application/x-www-form-urlencoded'},
                body,
            });
            document.querySelector('#result').innerHTML = await response.text();
        }""",
        question,
    )
    page.locator(f'[data-outcome="{outcome}"]').wait_for(state="visible")


def _stabilize_dynamic_diagnostics(page: Page) -> None:
    """Keep screenshots byte-stable without replacing any pipeline outcome."""
    page.locator(".diagnostics > span code").evaluate("node => node.textContent = 'capture-fixed'")
    timings = page.locator(".diagnostics dd")
    for index in range(timings.count()):
        timings.nth(index).evaluate("node => node.textContent = 'measured'")


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
        }
    )
    ASSETS.mkdir(parents=True, exist_ok=True)
    try:
        _run("docker", "compose", "up", "-d", "--build", "qdrant", env=env)
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
            page.locator(".diagnostics details").evaluate("details => details.open = true")
            _capture(page, "grounded")

            _submit(page, REFUSAL_QUESTION, "refused")
            if "score_threshold: 0.0" not in (ROOT / "configs/default.yaml").read_text():
                raise RuntimeError(
                    "refusal evidence requires configs/default.yaml score_threshold: 0.0"
                )
            _capture(page, "refused")

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
