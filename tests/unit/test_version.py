"""One version, four places.

``__init__.__version__``, ``Settings.app_version``, the FastAPI app metadata and
``pyproject.toml`` all publish a version. They drift the moment someone bumps
one by hand, and the symptom is a health endpoint that lies about what is
deployed — so the agreement is asserted instead of documented.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from production_rag import __version__
from production_rag.config import Settings

EXPECTED_VERSION = "0.1.0"
PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def test_package_version() -> None:
    assert __version__ == EXPECTED_VERSION


def test_settings_version_matches_the_package(settings: Settings) -> None:
    assert settings.app_version == __version__


def test_app_metadata_matches_the_package(client: TestClient) -> None:
    """What the OpenAPI document advertises is what is installed."""
    assert client.get("/openapi.json").json()["info"]["version"] == __version__


def test_health_reports_the_package_version(client: TestClient) -> None:
    assert client.get("/health").json()["version"] == __version__


@pytest.mark.skipif(not PYPROJECT.is_file(), reason="running against an installed wheel")
def test_pyproject_version_matches_the_package() -> None:
    metadata = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))

    assert metadata["project"]["version"] == __version__
