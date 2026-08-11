#!/usr/bin/env python3
"""Smoke test for a running production-rag stack.

Dependency-free on purpose: it uses only the standard library, so it can run
from any interpreter -- the host, CI, or inside the API container -- without
installing the project first. That matters because this script is the check
you reach for when the install itself is what you suspect.

Checks performed by default:

    1. GET {base}/health        -- liveness, must return 2xx
    2. GET {qdrant}/readyz      -- vector store readiness, must return 2xx
    3. GET {qdrant}/collections -- informational: is the collection ingested?

Liveness is the default probe on purpose. It is the one surface that must answer
without any dependency being up, so a failure here means the process is wrong
rather than the stack around it. Readiness reports on dependencies and is opt-in
via --ready, which adds:

    GET {base}{prefix}/ready    -- readiness, must return 2xx

Exit codes:

    0  every required check passed
    1  at least one required check failed
    2  bad usage / unreachable configuration

Usage:

    python scripts/smoke_health.py
    python scripts/smoke_health.py --ready
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
from dataclasses import asdict, dataclass
from typing import Any

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_COLLECTION = "production_rag"
DEFAULT_API_PREFIX = "/v1"


@dataclass
class CheckResult:
    """Outcome of a single HTTP probe, normalised so failures and successes print alike."""

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
        # S310: the URL comes from a CLI flag with an http(s) default, and this
        # script is dependency-free on purpose, so requests is not an option.
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
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
    """Probe every configured surface in order and return one result per check."""
    base = args.base_url.rstrip("/")
    qdrant = args.qdrant_url.rstrip("/")
    prefix = "/" + args.api_prefix.strip("/") if args.api_prefix.strip("/") else ""

    specs: list[tuple[str, str, bool]] = [
        ("api:/health", f"{base}/health", True),
    ]
    if args.ready:
        specs.append((f"api:{prefix}/ready", f"{base}{prefix}/ready", True))
    specs += [
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
    """Parse arguments, run the probes, print a verdict, and return the exit code."""
    parser = argparse.ArgumentParser(
        description="Smoke-test a running production-rag stack.",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="API base URL")
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL, help="Qdrant base URL")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="expected collection name")
    parser.add_argument("--api-prefix", default=DEFAULT_API_PREFIX, help="versioned route prefix")
    parser.add_argument(
        "--ready",
        action="store_true",
        help="also probe the versioned readiness route (dependency state)",
    )
    parser.add_argument("--timeout", type=float, default=10.0, help="per-request timeout, seconds")
    parser.add_argument("--retries", type=int, default=0, help="retries per check on failure")
    parser.add_argument(
        "--retry-delay", type=float, default=3.0, help="delay between retries, seconds"
    )
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
