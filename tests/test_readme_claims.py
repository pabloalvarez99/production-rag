"""Keep the README honest where no other test looks.

The README is the one surface a reviewer checks and the one surface no other
test covers. These claims tie its feature language to code reality.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parents[1]
SRC = ROOT / "src"
README = ROOT / "README.md"
HONESTY_MARKERS = re.compile(
    r"\b(?:not used|optional|declared|not implemented|considered and rejected|"
    r"planned|next|no|absent|does not)\b|📋|⏳",
    re.IGNORECASE,
)


def source_count(pattern: str, *, exclude: tuple[str, ...] = ()) -> int:
    """Count a regex over Python source, excluding declaration-only files."""
    return sum(
        len(re.findall(pattern, path.read_text(encoding="utf-8"), re.MULTILINE))
        for path in SRC.rglob("*.py")
        if path.relative_to(SRC).as_posix() not in exclude
    )


@dataclass(frozen=True, slots=True)
class Claim:
    name: str
    readme_pattern: str
    code_pattern: str
    exclude: tuple[str, ...] = ()

    @property
    def exists(self) -> bool:
        return source_count(self.code_pattern, exclude=self.exclude) > 0


CLAIMS = (
    Claim("LlamaIndex", r"llamaindex", r"^(?:from|import)\s+llama_index\b"),
    Claim("Ragas", r"ragas", r"^(?:from|import)\s+ragas\b"),
    Claim(
        "streaming",
        r"\bstream(?:ing)?\b",
        r"\b(?:generation|config)\.stream\b",
        ("production_rag/config_loader.py",),
    ),
    Claim(
        "retrieval filters",
        r"(?:payload\s+)?filters?",
        r"\ballowed_fields\b",
        ("production_rag/config_loader.py",),
    ),
    Claim(
        "/metrics",
        r"/metrics",
        r"@\w+\.(?:get|route)\([^\n]*[\"']/metrics[\"']",
        ("production_rag/config_loader.py",),
    ),
)


def absent_feature_failures(
    readme_lines: list[str], claims: tuple[Claim, ...] = CLAIMS
) -> list[str]:
    failures: list[str] = []
    for claim in claims:
        if claim.exists:
            continue
        for number, line in enumerate(readme_lines, start=1):
            if re.search(claim.readme_pattern, line, re.IGNORECASE) and not HONESTY_MARKERS.search(
                line
            ):
                failures.append(f"{claim.name} at README.md:{number}: {line.strip()}")
    return failures


def test_absent_features_are_only_mentioned_honestly() -> None:
    readme_lines = README.read_text(encoding="utf-8").splitlines()
    failures = absent_feature_failures(readme_lines)
    assert not failures, "Absent features need an honesty marker:\n" + "\n".join(failures)


def test_closing_claim_predicates_match_the_implemented_surface() -> None:
    """Mutation baseline for the five optional surfaces.

    ``retrieval filters`` flipped to ``True`` in M7: ``allowed_fields`` is read by
    :mod:`production_rag.retrieval.filters` and enforced on every query path, so
    the README may now describe filters without an honesty marker. The other four
    stay ``False`` — that is what keeps this a baseline rather than a rubber
    stamp, and what makes an accidental flip visible in a diff.
    """
    assert {claim.name: claim.exists for claim in CLAIMS} == {
        "LlamaIndex": False,
        "Ragas": False,
        "streaming": False,
        "retrieval filters": True,
        "/metrics": False,
    }


def test_metrics_claim_fails_if_its_honesty_markers_are_removed() -> None:
    line = next(
        line for line in README.read_text(encoding="utf-8").splitlines() if "/metrics" in line
    )
    mutated = line.replace("**DECLARED**", "**LIVE**").replace(
        "no `/metrics` route", "`/metrics` route"
    )
    metrics_claim = next(claim for claim in CLAIMS if claim.name == "/metrics")
    assert absent_feature_failures([mutated], (metrics_claim,))
