#!/usr/bin/env python3
"""Render scorecard tokens into delimited Markdown regions, using only stdlib."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TOKEN = re.compile(r"\{\{scorecard:([^{}]+)}}")
START = "<!-- SCORECARD:START -->"
END = "<!-- SCORECARD:END -->"
DEFAULT_SCORECARD = ROOT / "data/eval/reports/scorecard.json"
DEFAULT_TEMPLATE = ROOT / "docs/_scorecard.md.in"
DEFAULT_OUTPUT = ROOT / "README.md"


class RenderError(ValueError):
    """A template or scorecard contract could not be resolved safely."""


def _number(value: Any, *, signed: bool = False) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RenderError(f"expected a number, got {value!r}")
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def _provenance(data: dict[str, Any], *, n: int) -> str:
    provenance = data["provenance"]
    date = str(data["generated_at"])[:10]
    commit = str(provenance["commit"])[:12]
    fields = (
        f"embedder={provenance['embedder']}; LLM={provenance['llm']}; "
        f"judge={provenance['judge']}; n={n}; date={date}; commit={commit}"
    )
    if provenance.get("billed") is False:
        return f"⚠️ **FAKE PROVIDERS — plumbing check only, not a quality claim.** {fields}"
    return fields


def _items(data: dict[str, Any]) -> int:
    try:
        return int(data["provenance"]["golden"]["items"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RenderError("scorecard is missing its golden-set size") from exc


def _metric(data: dict[str, Any], metric: str, config: str) -> str:
    """Render one value bare. Its provenance is stated once, by the footnote token."""
    try:
        value = data["configs"][config][metric]
    except (KeyError, TypeError) as exc:
        raise RenderError(f"unresolved metric token: {metric}:{config}") from exc
    return f"**{_number(value)}**"


def _comparison(data: dict[str, Any], a: str, b: str, metric: str) -> str:
    """Render one paired delta. This line keeps its own provenance: a delta gets quoted."""
    matches = [
        item
        for item in data.get("comparisons", [])
        if item.get("a") == a
        and item.get("b") == b
        and item.get("metric") == metric
        and item.get("scope") == "overall"
    ]
    if len(matches) != 1:
        raise RenderError(f"comparison must resolve exactly once: {a}:{b}:{metric}")
    item = matches[0]
    n = int(item["n"])
    low, high = item["ci95"]
    interval = f"95% CI {_number(low, signed=True)} to {_number(high, signed=True)}; n={n}"
    if item.get("reportable") is False:
        reason = str(item.get("reason") or "comparison is not reportable")
        claim = (
            f"**Directional only:** {a} - {b} = {_number(item['delta'], signed=True)} "
            f"({interval}); **not reportable** — {reason}."
        )
    else:
        claim = f"**{a} - {b} = {_number(item['delta'], signed=True)}** ({interval})."
    return f"{claim}<br><sub>{_provenance(data, n=n)}</sub>"


def resolve_token(data: dict[str, Any], expression: str) -> str:
    """Resolve one token expression or fail closed."""
    parts = expression.split(":")
    if parts == ["footnote"]:
        return _provenance(data, n=_items(data))
    if len(parts) == 2 and parts[0] == "provenance":
        field = parts[1]
        try:
            value = data["provenance"][field]
        except (KeyError, TypeError) as exc:
            raise RenderError(f"unresolved provenance token: {field}") from exc
        return f"**{value}**"
    if len(parts) == 2:
        return _metric(data, parts[0], parts[1])
    if len(parts) == 4 and parts[0] == "comparison":
        return _comparison(data, parts[1], parts[2], parts[3])
    raise RenderError(f"unsupported scorecard token: {expression}")


def render_template(template: str, data: dict[str, Any]) -> str:
    """Replace every token and reject malformed or unresolved leftovers."""
    rendered = TOKEN.sub(lambda match: resolve_token(data, match.group(1)), template)
    if "{{scorecard:" in rendered:
        raise RenderError("malformed or unresolved scorecard token remains")
    return rendered.rstrip() + "\n"


def replace_region(document: str, rendered: str) -> str:
    """Replace exactly one generated region without changing surrounding prose."""
    if document.count(START) != 1 or document.count(END) != 1:
        raise RenderError("output must contain exactly one scorecard region")
    before, remainder = document.split(START, 1)
    _, after = remainder.split(END, 1)
    return f"{before}{START}\n{rendered}{END}{after}"


def render_file(scorecard: Path, template: Path, output: Path, *, check: bool = False) -> bool:
    """Render one region; return whether the output was already current."""
    data = json.loads(scorecard.read_text(encoding="utf-8"))
    rendered = render_template(template.read_text(encoding="utf-8"), data)
    current = output.read_text(encoding="utf-8")
    expected = replace_region(current, rendered)
    current_ok = current == expected
    if check and not current_ok:
        raise RenderError(f"stale rendered region in {output}")
    if not check and not current_ok:
        output.write_text(expected, encoding="utf-8", newline="\n")
    return current_ok


def main() -> int:
    """Run the command-line renderer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorecard", type=Path, default=DEFAULT_SCORECARD)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        render_file(args.scorecard, args.template, args.output, check=args.check)
    except (OSError, json.JSONDecodeError, RenderError) as exc:
        print(f"render_docs: error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
