"""Apply the README claim-guard idea to measured numbers.

``test_readme_claims.py`` ties prose to code reality. These tests apply the same
idea to scorecard values: published Markdown must be derived from an artefact,
carry provenance, and fail closed when the artefact cannot support a claim.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, cast

import pytest
from tools.render_docs import (
    END,
    START,
    TOKEN,
    RenderError,
    render_file,
    render_template,
    resolve_token,
)

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tests/fixtures/scorecard_sample.json"
PUBLISHED_SCORECARD = ROOT / "data/eval/reports/scorecard.json"
TEMPLATE = ROOT / "docs/_scorecard.md.in"
README = ROOT / "README.md"
KNOWN_METRICS = (
    "hit_at_5|recall_at_5|mrr|ndcg_at_5|citation_precision|"
    "invalid_marker_rate|refusal_accuracy"
)
METRIC_NUMBER = re.compile(
    rf"(?:{KNOWN_METRICS}).{{0,100}}\b0\.\d+|\b0\.\d+.{{0,100}}(?:{KNOWN_METRICS})",
    re.IGNORECASE,
)
PROVENANCE_ALLOW = re.compile(r"^\s*<!-- provenance-allow:\s*(.*?)\s*-->\s*$")


def _scorecard() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(FIXTURE.read_text(encoding="utf-8")))


def _without_rendered_regions(text: str) -> str:
    while START in text:
        before, remainder = text.split(START, 1)
        if END not in remainder:
            pytest.fail("unterminated scorecard region")
        _, after = remainder.split(END, 1)
        text = before + after
    return text


def _metric_number_violations(relative: str, text: str) -> list[tuple[str, int, str]]:
    violations: list[tuple[str, int, str]] = []
    exempt_next = False
    for line_number, line in enumerate(text.splitlines(), start=1):
        marker = PROVENANCE_ALLOW.match(line)
        if marker:
            if not marker.group(1).strip():
                violations.append((relative, line_number, "provenance allow marker needs a reason"))
            exempt_next = True
            continue
        if METRIC_NUMBER.search(line) and not exempt_next:
            violations.append((relative, line_number, line.strip()))
        exempt_next = False
    return violations


def test_every_repository_scorecard_token_resolves() -> None:
    data = _scorecard()
    paths = list(ROOT.rglob("*.md")) + list(ROOT.rglob("*.md.in"))
    tokens = [
        match.group(1)
        for path in paths
        for match in TOKEN.finditer(path.read_text(encoding="utf-8"))
    ]
    assert tokens, "the documentation must exercise the renderer"
    for token in tokens:
        assert "{{scorecard:" not in resolve_token(data, token)


def test_readme_rendered_region_is_current() -> None:
    assert render_file(PUBLISHED_SCORECARD, TEMPLATE, README, check=True)


def test_rendering_is_idempotent(tmp_path: Path) -> None:
    output = tmp_path / "README.md"
    output.write_text(f"before\n{START}\nstale\n{END}\nafter\n", encoding="utf-8")
    assert not render_file(FIXTURE, TEMPLATE, output)
    once = output.read_text(encoding="utf-8")
    assert render_file(FIXTURE, TEMPLATE, output)
    assert output.read_text(encoding="utf-8") == once


def test_unresolved_token_is_a_hard_error() -> None:
    with pytest.raises(RenderError, match="unresolved metric token"):
        render_template("{{scorecard:not_a_metric:hybrid}}", _scorecard())


def test_non_reportable_comparison_is_never_a_bare_delta() -> None:
    rendered = resolve_token(_scorecard(), "comparison:hybrid:sparse:hit_at_5")
    assert "Directional only" in rendered
    assert "not reportable" in rendered
    assert "95% CI" in rendered
    assert "n=60" in rendered
    assert "FAKE PROVIDERS" in rendered


def test_metric_shaped_numbers_only_exist_in_rendered_regions() -> None:
    violations: list[tuple[str, int, str]] = []
    for path in [README, *ROOT.joinpath("docs").rglob("*.md")]:
        relative = path.relative_to(ROOT).as_posix()
        text = _without_rendered_regions(path.read_text(encoding="utf-8"))
        violations.extend(_metric_number_violations(relative, text))
    assert violations == []


def test_provenance_allow_marker_requires_a_reason() -> None:
    text = "<!-- provenance-allow: -->\nhit_at_5 0.5"
    assert _metric_number_violations("example.md", text) == [
        ("example.md", 1, "provenance allow marker needs a reason")
    ]
