"""The local /evals page serves the committed free-path scorecard HTML."""

from __future__ import annotations

from fastapi.testclient import TestClient

from production_rag.config import Settings
from production_rag.main import create_app


def test_evals_serves_scorecard_labels() -> None:
    client = TestClient(create_app(Settings(config_path=None)))
    response = client.get("/evals")
    assert response.status_code == 200
    body = response.text
    assert "billed" in body.lower()
    assert "false" in body
    assert "not SOTA" in body or "not-SOTA" in body or "not SOTA" in body.lower()
    assert "contract" in body.lower() or "plumbing" in body.lower()
    assert "n (golden items)" in body or "golden" in body.lower()
