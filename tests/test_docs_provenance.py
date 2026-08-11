"""Apply the README claim-guard idea to measured numbers.

``test_readme_claims.py`` ties prose to code reality. These tests apply the same
idea to scorecard values: published Markdown must be derived from an artefact,
carry provenance, and fail closed when the artefact cannot support a claim.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
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
    r"hit(?:_at_|\s+at\s+|@)\d+|recall(?:_at_|\s+at\s+|@)\d+|mrr|ndcg(?:_at_|\s+at\s+|@)?\d*|"
    r"citation_precision|citation precision|invalid_marker_rate|invalid marker rate|"
    r"refusal_accuracy|refusal accuracy"
)
NUMBER = r"(?:\b0\.\d+|\b\d+(?:\.\d+)?%|\b\d+(?:\.\d+)?\s+(?:percentage\s+)?points?\b)"
METRIC_NUMBER = re.compile(
    rf"(?:{KNOWN_METRICS})[\s\S]*?{NUMBER}|{NUMBER}[\s\S]*?(?:{KNOWN_METRICS})",
    re.IGNORECASE | re.MULTILINE,
)
PERCENTAGE_CLAIM = re.compile(
    r"(?:\b\d+(?:\.\d+)?%|\b\d+(?:\.\d+)?\s+(?:percentage\s+)?points?\b)",
    re.IGNORECASE,
)
PROVENANCE_ALLOW = re.compile(r"^\s*<!-- provenance-allow:\s*(.*?)\s*-->\s*$")
PROVENANCE_ALLOW_CATEGORIES = {
    "historical-measurement",
    "explanatory-example",
    "external-citation",
}
TRACKED_MARKDOWN_EXCLUSIONS = {
    "data/corpus/": "vendored third-party evaluation input, not project documentation",
    "data/raw/": "evaluation fixtures whose text is an input to tests, not project documentation",
}


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
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line_number = index + 1
        line = lines[index]
        marker = PROVENANCE_ALLOW.match(line)
        if marker:
            reason = marker.group(1).strip()
            category, separator, explanation = reason.partition(":")
            valid_reason = (
                category in PROVENANCE_ALLOW_CATEGORIES
                and bool(separator)
                and bool(explanation.strip())
            )
            if not valid_reason:
                violations.append(
                    (
                        relative,
                        line_number,
                        "provenance allow marker needs a known category and explanation",
                    )
                )
            exempt_next = True
            index += 1
            continue
        if not line.strip():
            index += 1
            continue
        if line.lstrip().startswith("|"):
            table_lines = [line]
            index += 1
            while index < len(lines) and lines[index].lstrip().startswith("|"):
                table_lines.append(lines[index])
                index += 1
            unit = "\n".join(table_lines)
        else:
            index += 1
            prose_lines = [line]
            while (
                index < len(lines)
                and lines[index].strip()
                and not lines[index].lstrip().startswith("|")
                and not PROVENANCE_ALLOW.match(lines[index])
            ):
                prose_lines.append(lines[index])
                index += 1
            unit = "\n".join(prose_lines)
        if (METRIC_NUMBER.search(unit) or PERCENTAGE_CLAIM.search(unit)) and not exempt_next:
            violations.append((relative, line_number, unit.strip()))
        exempt_next = False
    return violations


def _tracked_markdown_paths() -> list[Path]:
    git = shutil.which("git")
    if git is None:
        pytest.fail("git executable is required to enumerate tracked Markdown")
    output = subprocess.run(  # noqa: S603 - resolved git executable and fixed arguments
        [git, "ls-files", "-z", "--", "*.md", "*.md.in"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout
    paths: list[Path] = []
    for raw_path in output.decode().split("\0"):
        if not raw_path:
            continue
        relative = raw_path.replace("\\", "/")
        if any(relative.startswith(prefix) for prefix in TRACKED_MARKDOWN_EXCLUSIONS):
            continue
        paths.append(ROOT / relative)
    return paths


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
    for path in _tracked_markdown_paths():
        relative = path.relative_to(ROOT).as_posix()
        text = _without_rendered_regions(path.read_text(encoding="utf-8"))
        violations.extend(_metric_number_violations(relative, text))
    assert violations == []


def test_guard_scans_markdown_outside_docs_and_multiline_tables() -> None:
    tracked = {path.relative_to(ROOT).as_posix() for path in _tracked_markdown_paths()}
    assert "README.md" in tracked
    assert "data/eval/README.md" in tracked
    assert ".github/pull_request_template.md" in tracked
    table = "| Metric | Value |\n|---|---|\n| hit@5 |\n| 0.56 |"
    assert _metric_number_violations("table.md", table)


@pytest.mark.parametrize(
    "claim",
    ["hit@5 = 0.56", "Hit@5 improved 12 points", "nDCG reached 56%", "recall at 5: 0.8"],
)
def test_guard_recognizes_metric_aliases_and_percentages(claim: str) -> None:
    assert _metric_number_violations("claim.md", claim)


def test_provenance_allow_marker_requires_a_reason() -> None:
    text = "<!-- provenance-allow: -->\nhit_at_5 0.5"
    assert _metric_number_violations("example.md", text) == [
        (
            "example.md",
            1,
            "provenance allow marker needs a known category and explanation",
        )
    ]


@pytest.mark.parametrize(
    "reason",
    [
        "because it failed",
        "unknown-category: legacy number",
        "historical-measurement:",
    ],
)
def test_provenance_allow_marker_rejects_uncategorized_reasons(reason: str) -> None:
    text = f"<!-- provenance-allow: {reason} -->\nhit_at_5 0.5"
    assert _metric_number_violations("example.md", text)[0][1] == 1


@pytest.mark.parametrize("category", sorted(PROVENANCE_ALLOW_CATEGORIES))
def test_provenance_allow_marker_accepts_known_categories(category: str) -> None:
    text = f"<!-- provenance-allow: {category}: why this is not a current claim -->\nhit_at_5 0.5"
    assert _metric_number_violations("example.md", text) == []
