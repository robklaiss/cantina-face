#!/usr/bin/env python3
"""Simple checkout latency benchmark for Cantina Face APIs."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
TIMEOUT = 15


def _build_url(base: str, path: str) -> str:
    if not path.startswith("/"):
        path = "/" + path
    return base.rstrip("/") + path


def _http_request(url: str, headers: Dict[str, str], data: bytes | None = None, method: str = "GET") -> bytes:
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return resp.read()
    except urllib.error.HTTPError as err:
        body = err.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"{method} {url} failed: {err.code} {err.reason}\n{body}") from err


def _request_json(base: str, path: str, headers: Dict[str, str]) -> Dict:
    url = _build_url(base, path)
    payload = _http_request(url, headers)
    return json.loads(payload.decode("utf-8"))


def _post_form(base: str, path: str, form: Dict[str, str]) -> Dict:
    url = _build_url(base, path)
    data = urllib.parse.urlencode(form).encode("utf-8")
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    payload = _http_request(url, headers, data=data, method="POST")
    return json.loads(payload.decode("utf-8"))


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    values_sorted = sorted(values)
    k = (len(values_sorted) - 1) * pct
    f = int(k)
    c = min(f + 1, len(values_sorted) - 1)
    if f == c:
        return values_sorted[int(k)]
    d0 = values_sorted[f] * (c - k)
    d1 = values_sorted[c] * (k - f)
    return d0 + d1


def bench_sequence(base_url: str, token: str, student_id: str, iterations: int) -> Dict[str, List[float]]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }
    metrics: Dict[str, List[float]] = {
        "products": [],
        "student_detail": [],
        "student_orders": [],
    }

    for _ in range(iterations):
        start = time.perf_counter()
        _request_json(base_url, "/api/products", headers)
        metrics["products"].append((time.perf_counter() - start) * 1000)

        start = time.perf_counter()
        _request_json(base_url, f"/api/students/{student_id}", headers)
        metrics["student_detail"].append((time.perf_counter() - start) * 1000)

        start = time.perf_counter()
        _request_json(base_url, f"/api/students/{student_id}/scheduled-orders?status_filter=pending", headers)
        metrics["student_orders"].append((time.perf_counter() - start) * 1000)

    return metrics


def resolve_student(base_url: str, token: str, provided_id: str | None) -> str:
    if provided_id:
        return provided_id

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }
    data = _request_json(base_url, "/api/students?limit=1", headers)
    if not data:
        raise RuntimeError("No students available in the database. Specify --student-id explicitly.")
    return data[0]["id"]


def authenticate(base_url: str, username: str, password: str) -> str:
    payload = _post_form(base_url, "/auth/token", {"username": username, "password": password})
    token = payload.get("access_token")
    if not token:
        raise RuntimeError("Authentication failed: missing access_token in response")
    return token


def summarize(label: str, samples: List[float]) -> str:
    if not samples:
        return f"{label:<20} | no samples"
    avg = statistics.fmean(samples)
    p95 = percentile(samples, 0.95)
    worst = max(samples)
    return f"{label:<20} | count={len(samples):<3d} avg={avg:7.2f}ms p95={p95:7.2f}ms max={worst:7.2f}ms"


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark checkout-related Cantina Face endpoints.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Server base URL (default: %(default)s)")
    parser.add_argument("--username", required=True, help="User email for /auth/token")
    parser.add_argument("--password", required=True, help="Password for /auth/token")
    parser.add_argument("--student-id", help="Existing student ID to benchmark (default: first available)")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations per endpoint (default: %(default)s)")

    args = parser.parse_args()

    print(f"Connecting to {args.base_url} ...", flush=True)
    token = authenticate(args.base_url, args.username, args.password)
    student_id = resolve_student(args.base_url, token, args.student_id)
    print(f"Using student_id={student_id}")

    metrics = bench_sequence(args.base_url, token, student_id, max(1, args.iterations))
    print("\nBenchmark results:")
    for label, samples in metrics.items():
        print("  " + summarize(label, samples))

    health_url = _build_url(args.base_url, "/api/health/timing")
    try:
        health_payload = _http_request(health_url, {"Accept": "application/json"})
        print("\nCurrent /api/health/timing snapshot:")
        print(health_payload.decode("utf-8"))
    except Exception as exc:
        print(f"\nWarning: could not read /api/health/timing: {exc}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Benchmark interrupted", file=sys.stderr)
        raise
