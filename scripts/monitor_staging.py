"""Probe staging and save a timestamped operational snapshot for CI artifacts."""

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def request_json(url, *, data=None, json_body=None, token=None, timeout=120):
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    if json_body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(json_body, ensure_ascii=False).encode()
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(url, data=data, headers=headers)
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response), round((time.perf_counter() - started) * 1000, 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, default=Path("results/staging-monitor.json"))
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--require-auth", action="store_true")
    args = parser.parse_args()
    base = args.url.rstrip("/")
    report = {"checked_at": datetime.now(timezone.utc).isoformat(), "url": base, "checks": {}}
    try:
        ready, latency = request_json(base + "/ready", timeout=args.timeout)
        report["checks"]["ready"] = ready.get("status") == "ready"
        report["ready_latency_ms"] = latency
        report["model_version"] = ready.get("model_version")
        username, password = os.getenv("STAGING_USERNAME"), os.getenv("STAGING_PASSWORD")
        if username and password:
            body = urllib.parse.urlencode({"username": username, "password": password}).encode()
            login, latency = request_json(base + "/login", data=body, timeout=args.timeout)
            token = login["access_token"]
            report["checks"]["login"] = bool(token)
            report["login_latency_ms"] = latency
            stats, latency = request_json(base + "/stats", token=token, timeout=args.timeout)
            report["checks"]["stats"] = stats.get("model_version") == report["model_version"]
            report["stats_latency_ms"] = latency
            report["operational"] = {key: stats.get(key) for key in (
                "total_predictions", "suspicious_count", "reviewed_count", "reviewed_accuracy",
                "inference_latency_sample_size", "inference_latency", "queue_latency", "processing_latency",
            )}
            model, _ = request_json(base + "/model-info", token=token, timeout=args.timeout)
            report["checks"]["model_info"] = model.get("model_version") == report["model_version"]
            report["release_status"] = model.get("release_status")
            report["dataset_version"] = model.get("dataset_version")
            if login.get("is_admin"):
                runs, _ = request_json(base + "/mlops/runs", token=token, timeout=args.timeout)
                report["checks"]["mlops_runs"] = isinstance(runs.get("runs"), list)
                report["archived_run_count"] = runs.get("total", len(runs.get("runs", [])))
                current, _ = request_json(base + "/mlops/monitoring/current", token=token, timeout=args.timeout)
                report["checks"]["monitoring"] = current.get("status") in {"healthy", "insufficient_data"}
                report["monitoring_status"] = current.get("status")
                report["model_quality_status"] = current.get("drift_status")
                report["operational_signals"] = {key: current.get(key) for key in (
                    "api_5xx_24h", "storage_bytes", "processing_p95_ms", "processing_p99_ms",
                    "low_confidence_rate", "distribution_js_divergence", "operational_sample_size",
                    "reviewed_count", "reviewed_accuracy",
                )}
                saved, _ = request_json(base + "/mlops/monitoring", json_body=current,
                                        token=token, timeout=args.timeout)
                report["checks"]["snapshot_saved"] = bool(saved.get("snapshot_id"))
        else:
            report["auth_checks_skipped"] = "STAGING_USERNAME and STAGING_PASSWORD are not set"
            if args.require_auth:
                report["checks"]["auth_configured"] = False
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError, KeyError) as error:
        report["checks"]["request"] = False
        report["error"] = str(error)
    report["healthy"] = bool(report["checks"]) and all(report["checks"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["healthy"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
