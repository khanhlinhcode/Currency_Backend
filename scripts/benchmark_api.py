"""Measure end-to-end prediction latency against a deployed API."""

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import statistics
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[math.ceil(fraction * len(ordered)) - 1]


def summarize(values):
    if not values:
        return {"mean": None, "p50": None, "p95": None, "p99": None}
    return {
        "mean": round(statistics.mean(values), 2),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def send_prediction(url, image_bytes, filename, timeout, token, run_id):
    boundary = f"currency-benchmark-{uuid.uuid4().hex}"
    payload = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode() + image_bytes + f"\r\n--{boundary}--\r\n".encode()
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}", "X-Benchmark-Run": run_id}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
            return {
                "status": response.status,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "model_version": result.get("model_version"),
                "inference_ms": round(result.get("inference_time", 0) * 1000, 2),
                "queue_ms": round(result.get("queue_time", 0) * 1000, 2),
                "processing_ms": round(result.get("processing_time", 0) * 1000, 2),
            }
    except urllib.error.HTTPError as error:
        return {"status": error.code, "error": error.read(300).decode(errors="replace")}
    except (urllib.error.URLError, TimeoutError) as error:
        return {"status": None, "error": str(error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, nargs="+", help="Representative JPEG images to upload in rotation")
    parser.add_argument("--url", required=True, help="API base URL, for example https://api.example.com")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--token", default="", help="Optional analyst token; avoid storing it in shell history")
    parser.add_argument("--token-env", default="", help="Environment variable containing an analyst token")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.runs < 1 or args.concurrency < 1 or args.warmups < 0:
        parser.error("runs and concurrency must be positive; warmups cannot be negative")
    for image in args.images:
        if not image.is_file() or image.suffix.lower() not in {".jpg", ".jpeg"}:
            parser.error(f"Use an existing JPEG file: {image}")
    samples = [(image.name, image.read_bytes()) for image in args.images]
    token = args.token or (os.getenv(args.token_env, "") if args.token_env else "")
    run_id = str(uuid.uuid4())
    url = args.url.rstrip("/") + "/predict"
    warmup_results = [
        send_prediction(url, samples[index % len(samples)][1], samples[index % len(samples)][0], args.timeout, token, run_id)
        for index in range(args.warmups)
    ]

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(send_prediction, url, samples[index % len(samples)][1], samples[index % len(samples)][0], args.timeout, token, run_id)
            for index in range(args.runs)
        ]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    wall_seconds = time.perf_counter() - started
    successful = [result for result in results if result.get("status") == 200]
    latencies = [result["latency_ms"] for result in successful]
    status_counts = Counter(str(result.get("status") or "network_error") for result in results)
    report = {
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_run_id": run_id,
        "url": url,
        "runs": args.runs,
        "warmups": args.warmups,
        "concurrency": args.concurrency,
        "image_count": len(samples),
        "unique_image_count": len({hashlib.sha256(content).digest() for _, content in samples}),
        "image_bytes_min": min(len(content) for _, content in samples),
        "image_bytes_max": max(len(content) for _, content in samples),
        "warmup_errors": sum(result.get("status") != 200 for result in warmup_results),
        "successes": len(successful),
        "errors": args.runs - len(successful),
        "success_rate": round(len(successful) / args.runs, 6),
        "status_counts": dict(sorted(status_counts.items())),
        "error_examples": [result for result in results if result.get("status") != 200][:5],
        "model_versions": sorted({result["model_version"] for result in successful if result["model_version"]}),
        "wall_seconds": round(wall_seconds, 3),
        "throughput_rps": round(len(successful) / wall_seconds, 3),
        "latency_ms": summarize(latencies),
        "server_processing_ms": summarize([result["processing_ms"] for result in successful]),
        "queue_ms": summarize([result["queue_ms"] for result in successful]),
        "inference_ms": summarize([result["inference_ms"] for result in successful]),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["errors"] or report["warmup_errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
