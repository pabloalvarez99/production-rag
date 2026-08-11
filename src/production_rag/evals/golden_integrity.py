"""Standalone integrity checks for a source-labelled golden set."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from production_rag.config_loader import load_yaml_config
from production_rag.ingest.chunking import chunk_document
from production_rag.ingest.loaders import iter_documents


@dataclass(frozen=True, slots=True)
class IntegrityResult:
    """Integrity findings and corpus coverage counts."""

    items: int
    chunkable_documents: int
    errors: tuple[str, ...]


def check_golden_integrity(corpus: Path, golden: Path) -> IntegrityResult:
    """Validate labels against files that actually survive configured chunking."""
    config = load_yaml_config()
    documents = tuple(
        iter_documents(
            corpus,
            include_extensions=config.ingest.include_extensions,
            exclude_globs=config.ingest.exclude_globs,
        )
    )
    chunkable = {
        document.source_path
        for document in documents
        if chunk_document(document, config.ingest.chunking)[0]
    }
    disk_paths = {document.source_path for document in documents}
    errors: list[str] = []
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(golden.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_number}: invalid JSON: {exc.msg}")
            continue
        if not isinstance(value, dict):
            errors.append(f"line {line_number}: item must be an object")
            continue
        records.append(value)

    ids = [str(item.get("id", "")) for item in records]
    questions = [str(item.get("question", "")) for item in records]
    for item_id, count in Counter(ids).items():
        if not item_id:
            errors.append("item with missing id")
        elif count > 1:
            errors.append(f"{item_id}: duplicate id ({count} occurrences)")
    for question, count in Counter(questions).items():
        if not question:
            errors.append("item with missing question")
        elif count > 1:
            errors.append(f"duplicate question ({count} occurrences): {question}")

    slices: Counter[str] = Counter()
    for item in records:
        item_id = str(item.get("id", "<missing-id>"))
        category = str(item.get("category", ""))
        slices[category] += 1
        raw_paths = item.get("expected_source_paths", [])
        if not isinstance(raw_paths, list) or not all(isinstance(path, str) for path in raw_paths):
            errors.append(f"{item_id}: expected_source_paths must be a list of strings")
            continue
        paths = tuple(raw_paths)
        if item.get("answerable") is False and paths:
            errors.append(f"{item_id}: answerable=false requires empty expected_source_paths")
        for path in paths:
            if path not in disk_paths:
                errors.append(f"{item_id}: source does not exist: {path}")
            elif path not in chunkable:
                errors.append(f"{item_id}: source produces no chunks: {path}")
    for category, count in sorted(slices.items()):
        if count != 10:
            errors.append(f"slice {category!r}: expected 10 items, found {count}")

    return IntegrityResult(len(records), len(chunkable), tuple(errors))
