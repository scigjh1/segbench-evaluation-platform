"""Benchmark SegBench on the deterministic built-in regression suite."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from server import sample_evaluation  # noqa: E402


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(np.ceil(len(ordered) * fraction)) - 1)
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--tolerance", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmark_results" / "local_cpu.json",
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.runs < 1:
        parser.error("warmup must be >= 0 and runs must be >= 1")

    for _ in range(args.warmup):
        sample_evaluation(args.tolerance)

    timings_ms: list[float] = []
    latest_result = None
    for _ in range(args.runs):
        started = time.perf_counter()
        latest_result = sample_evaluation(args.tolerance)
        timings_ms.append((time.perf_counter() - started) * 1000)

    assert latest_result is not None
    sample_pairs_per_run = latest_result["candidate"]["summary"]["samples"]
    masks_per_run = sample_pairs_per_run * 2
    mean_ms = statistics.fmean(timings_ms)
    report = {
        "timestampUtc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "os": platform.platform(),
            "processor": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "device": "CPU",
        },
        "protocol": {
            "imageWidth": 512,
            "imageHeight": 512,
            "samplePairsPerRun": sample_pairs_per_run,
            "masksEvaluatedPerRun": masks_per_run,
            "warmupRuns": args.warmup,
            "measuredRuns": args.runs,
            "boundaryTolerancePx": args.tolerance,
            "scope": "end-to-end Python evaluation including PNG decode, six metrics, aggregation and regression deltas",
        },
        "latencyMsPerFullRun": {
            "mean": round(mean_ms, 2),
            "median": round(statistics.median(timings_ms), 2),
            "p95": round(percentile(timings_ms, 0.95), 2),
            "min": round(min(timings_ms), 2),
            "max": round(max(timings_ms), 2),
        },
        "throughputMasksPerSecond": round(masks_per_run * 1000 / mean_ms, 1),
        "quality": {
            "baseline": latest_result["baseline"]["summary"],
            "candidate": latest_result["candidate"]["summary"],
            "improvement": latest_result["delta"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
