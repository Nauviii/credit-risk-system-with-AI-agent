"""Measure serving latency properly: percentiles over many warm requests, not one cold one.

A single request through Swagger includes first-call warm-up - Polars, LightGBM and the
JSON stack all pay a one-off cost - and reads far slower than steady state. The p95 target
is a property of a warmed service under load, so measure it that way.

Runs in-process against the ASGI app, which removes network and browser overhead and
isolates what the service itself costs. Add network time separately when sizing infra.
"""

import argparse
import statistics
import time
from pathlib import Path

from fastapi.testclient import TestClient

from credit_risk.serving import app as app_module

_EXAMPLE = app_module._EXAMPLE_APPLICATION


def _percentiles(timings: list[float]) -> dict:
    ordered = sorted(timings)
    return {
        "n": len(ordered),
        "p50_ms": round(statistics.median(ordered), 2),
        "p95_ms": round(ordered[int(len(ordered) * 0.95)], 2),
        "p99_ms": round(ordered[min(int(len(ordered) * 0.99), len(ordered) - 1)], 2),
        "max_ms": round(ordered[-1], 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default="artifacts/champion")
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()

    app_module.BUNDLE_PATH = Path(args.bundle)
    with TestClient(app_module.app) as client:
        payload = {"application": _EXAMPLE}

        for _ in range(args.warmup):
            client.post("/score", json=payload)

        timings = []
        for _ in range(args.requests):
            started = time.perf_counter()
            response = client.post("/score", json=payload)
            timings.append((time.perf_counter() - started) * 1000)
            response.raise_for_status()

        single = _percentiles(timings)
        print(f"single /score over {single['n']} warm requests")
        for key in ("p50_ms", "p95_ms", "p99_ms", "max_ms"):
            print(f"  {key}: {single[key]}")
        print(f"  target: p95 < 200 ms -> {'PASS' if single['p95_ms'] < 200 else 'FAIL'}")

        batch_payload = {"applications": [_EXAMPLE] * args.batch_size}
        client.post("/score/batch", json=batch_payload)
        batch_timings = []
        for _ in range(max(args.requests // 10, 10)):
            started = time.perf_counter()
            client.post("/score/batch", json=batch_payload)
            batch_timings.append((time.perf_counter() - started) * 1000)

        batch = _percentiles(batch_timings)
        per_row = batch["p50_ms"] / args.batch_size
        print(f"\nbatch of {args.batch_size}")
        for key in ("p50_ms", "p95_ms"):
            print(f"  {key}: {batch[key]}")
        print(f"  per application: {per_row:.3f} ms")
        print(f"  speedup vs looping /score: {single['p50_ms'] / per_row:.1f}x")


if __name__ == "__main__":
    main()