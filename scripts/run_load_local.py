"""Local free-path load: 100 fake queries against an in-process executor.

Honesty: single process, no real Qdrant round-trip in this script by default —
it times the public HTTP surface with a fake executor so CI and clone demos can
produce ``docs/assets/load.json`` without inventing SOTA throughput.

Usage:
  python scripts/run_load_local.py
  python scripts/run_load_local.py --n 100 --out docs/assets/load.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from fastapi.testclient import TestClient

from production_rag.api.routes import query as query_route
from production_rag.api.schemas import CitationOut, QueryRequest, QueryResponse
from production_rag.config import Settings, get_settings
from production_rag.main import create_app

ROOT = Path(__file__).resolve().parents[1]


def _fake_executor(
    payload: QueryRequest,
    *,
    settings: Settings,
    request_id: str,
    embedder_kind: str | None = None,
    on_delta: object = None,
) -> QueryResponse:
    if on_delta is not None and callable(on_delta):
        on_delta("partial ")
        on_delta("answer ")
    return QueryResponse(
        answer=f"Load-path grounded reply for {payload.question[:48]} [1].",
        citations=[
            CitationOut(
                marker=1,
                chunk_id="load-c1",
                source_path="sample/01-hybrid-search.md",
                text="Hybrid search fuses ranks.",
                rank=1,
            )
        ],
        refused=False,
        refusal_reason=None,
    )


def main() -> int:
    """Run the load harness and write JSON latency stats."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "docs" / "assets" / "load.json",
    )
    args = parser.parse_args()
    get_settings.cache_clear()
    settings = Settings()
    app = create_app(settings)
    app.dependency_overrides[query_route.get_query_executor] = lambda: _fake_executor
    app.dependency_overrides[query_route.get_streaming_query_executor] = (
        lambda: _fake_executor
    )
    client = TestClient(app)
    latencies_ms: list[float] = []
    for index in range(args.n):
        started = time.perf_counter()
        response = client.post(
            "/v1/query",
            json={
                "question": f"Load query {index}: why hybrid search?",
                "llm": "fake",
                "embedder": "fake",
            },
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        if response.status_code != 200:
            raise SystemExit(f"load query failed: {response.status_code} {response.text}")
        latencies_ms.append(elapsed)
    latencies_ms.sort()

    def pct(p: float) -> float:
        if not latencies_ms:
            return 0.0
        index = round(p * (len(latencies_ms) - 1))
        rank = min(len(latencies_ms) - 1, max(0, index))
        return round(latencies_ms[rank], 3)

    payload = {
        "schema_version": 1,
        "n": args.n,
        "path": "POST /v1/query",
        "providers": {"embedder": "fake", "llm": "fake", "executor": "in-process-fake"},
        "billed": False,
        "hardware_note": (
            "single process + TestClient + fake executor; not SOTA throughput; "
            "not a capacity plan; local free-path plumbing only"
        ),
        "qdrant": "not dialed by this load harness",
        "latency_ms": {
            "p50": pct(0.50),
            "p95": pct(0.95),
            "min": round(min(latencies_ms), 3),
            "max": round(max(latencies_ms), 3),
            "mean": round(statistics.fmean(latencies_ms), 3),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
