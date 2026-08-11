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
    Claim("retrieval filters", r"(?:payload\s+)?filters?", r"\ballowed_fields\b"),
    Claim(
        "/metrics",
        r"/metrics",
        r"@\w+\.(?:get|route)\([^\n]*[\"']/metrics[\"']",
        ("production_rag/config_loader.py",),
    ),
)


def test_absent_features_are_only_mentioned_honestly() -> None:
    readme_lines = README.read_text(encoding="utf-8").splitlines()
    failures: list[str] = []
    for claim in CLAIMS:
        if claim.exists:
            continue
        for number, line in enumerate(readme_lines, start=1):
            if re.search(claim.readme_pattern, line, re.IGNORECASE) and not HONESTY_MARKERS.search(
                line
            ):
                failures.append(f"{claim.name} at README.md:{number}: {line.strip()}")
    assert not failures, "Absent features need an honesty marker:\n" + "\n".join(failures)
