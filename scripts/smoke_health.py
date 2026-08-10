#!/usr/bin/env python3
"""Smoke test for a running production-rag stack.

Dependency-free on purpose: it uses only the standard library, so it can run
from any interpreter -- the host, CI, or inside the API container -- without
installing the project first. That matters because this script is the check
you reach for when the install itself is what you suspect.

Checks performed:

    1. GET {base}/health      -- liveness, must return 2xx
    2. GET {base}/v1/health   -- versioned readiness, must return 2xx
    3. GET {qdrant}/readyz    -- vector store readiness, must return 2xx
    4. GET {qdrant}/collections -- informational: is the collection ingested?

Exit codes:

    0  every required check passed
    1  at least one required check failed
    2  bad usage / unreachable configuration

Usage:

    python scripts/smoke_health.py
    python scripts/smoke_health.py --base-url http://localhost:8000 --json
    python scripts/smoke_health.py --retries 20 --retry-delay 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from typing import Any

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_COLLECTION = "production_rag"


@dataclass
class CheckResult:
    name: str
    url: str
    ok: bool
    status: int | None
    elapsed_ms: int
    required: bool
    detail: str


def probe(name: str, url: str, *, required: bool, timeout: float) -> CheckResult:
    """Issue one GET and normalise the outcome into a CheckResult."""
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            elapsed = int((time.perf_counter() - started) * 1000)
            return CheckResult(
                name=name,
                url=url,
                ok=200 <= response.status < 300,
                status=response.status,
                elapsed_ms=elapsed,
                required=required,
                detail=body.strip()[:400],
            )
    except urllib.error.HTTPError as exc:
        elapsed = int((time.perf_counter() - started) * 1000)
        return CheckResult(
            name=name,
            url=url,
            ok=False,
            status=exc.code,
            elapsed_ms=elapsed,
            required=required,
            detail=f"HTTP {exc.code} {exc.reason}",
        )
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        elapsed = int((time.perf_counter() - started) * 1000)
        return CheckResult(
            name=name,
            url=url,
            ok=False,
            status=None,
            elapsed_ms=elapsed,
            required=required,
            detail=str(exc),
        )


def run_checks(args: argparse.Namespace) -> list[CheckResult]:
    base = args.base_url.rstrip("/")
    qdrant = args.qdrant_url.rstrip("/")

    specs: list[tuple[str, str, bool]] = [
        ("api:/health", f"{base}/health", True),
        ("api:/v1/health", f"{base}/v1/health", True),
        ("qdrant:/readyz", f"{qdrant}/readyz", True),
        ("qdrant:/collections", f"{qdrant}/collections", False),
    ]

    results: list[CheckResult] = []
    for name, url, required in specs:
        # Retries exist for the boot window only: a cold uvicorn refuses
        # connections for a few seconds after the container reports running.
        attempt = 0
        while True:
            result = probe(name, url, required=required, timeout=args.timeout)
            attempt += 1
            if result.ok or attempt > args.retries:
                break
            time.sleep(args.retry_delay)
        results.append(result)
    return results


def collection_present(results: list[CheckResult], collection: str) -> bool | None:
    """None when the collections endpoint was unreachable."""
    for result in results:
        if result.name != "qdrant:/collections":
            continue
        if not result.ok:
            return None
        try:
            payload: Any = json.loads(result.detail)
        except json.JSONDecodeError:
            return collection in result.detail
        names = {
            entry.get("name")
            for entry in payload.get("result", {}).get("collections", [])
            if isinstance(entry, dict)
        }
        return collection in names
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Smoke-test a running production-rag stack.",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="API base URL")
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL, help="Qdrant base URL")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="expected collection name")
    parser.add_argument("--timeout", type=float, default=10.0, help="per-request timeout, seconds")
    parser.add_argument("--retries", type=int, default=0, help="retries per check on failure")
    parser.add_argument("--retry-delay", type=float, default=3.0, help="delay between retries, seconds")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    if args.retries < 0 or args.timeout <= 0:
        parser.error("--retries must be >= 0 and --timeout must be > 0")
        return 2

    results = run_checks(args)
    has_collection = collection_present(results, args.collection)
    required_failures = [r for r in results if r.required and not r.ok]
    exit_code = 1 if required_failures else 0

    if args.json:
        print(
            json.dumps(
                {
                    "ok": exit_code == 0,
                    "collection": args.collection,
                    "collection_present": has_collection,
                    "checks": [asdict(r) for r in results],
                },
                indent=2,
            )
        )
        return exit_code

    width = max(len(r.name) for r in results)
    for result in results:
        verdict = "PASS" if result.ok else ("FAIL" if result.required else "WARN")
        status = result.status if result.status is not None else "---"
        print(f"{verdict:4}  {result.name:<{width}}  {status:>4}  {result.elapsed_ms:>5} ms")

    for result in results:
        if not result.ok:
            print(f"      {result.name}: {result.detail}", file=sys.stderr)

    if has_collection is True:
        print(f"collection '{args.collection}' present.")
    elif has_collection is False:
        print(f"collection '{args.collection}' not found yet -- run the ingest job.")

    print("smoke: OK" if exit_code == 0 else "smoke: FAIL")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
