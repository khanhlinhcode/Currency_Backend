"""Apply explicit, provisional staging SLOs to an HTTP benchmark report."""

import argparse
import json
from pathlib import Path


def check(report, targets, expected_version):
    def metric(section, name):
        return (report.get(section) or {}).get(name)

    checks = {
        "enough measured requests": report.get("runs", 0) >= targets["min_requests"],
        "representative images": report.get("unique_image_count", 0) >= targets["min_unique_images"],
        "expected concurrency": report.get("concurrency") == targets["required_concurrency"],
        "warmups succeeded": report.get("warmup_errors", 0) == 0,
        "success rate": report.get("success_rate", 0) >= targets["min_success_rate"],
        "one expected model version": report.get("model_versions") == [expected_version],
        "client p95": metric("latency_ms", "p95") is not None and metric("latency_ms", "p95") <= targets["max_client_p95_ms"],
        "client p99": metric("latency_ms", "p99") is not None and metric("latency_ms", "p99") <= targets["max_client_p99_ms"],
        "processing p95": metric("server_processing_ms", "p95") is not None and metric("server_processing_ms", "p95") <= targets["max_server_processing_p95_ms"],
        "queue p95": metric("queue_ms", "p95") is not None and metric("queue_ms", "p95") <= targets["max_queue_p95_ms"],
        "inference p95": metric("inference_ms", "p95") is not None and metric("inference_ms", "p95") <= targets["max_inference_p95_ms"],
        "throughput": report.get("throughput_rps", 0) >= targets["min_throughput_rps"],
    }
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--targets", type=Path, default=Path("docs/staging-targets.json"))
    parser.add_argument("--manifest", type=Path, default=Path("models/model-manifest.json"))
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    targets = json.loads(args.targets.read_text(encoding="utf-8"))
    expected_version = json.loads(args.manifest.read_text(encoding="utf-8"))["version"]
    checks = check(report, targets, expected_version)
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
