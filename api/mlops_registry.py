"""Idempotent metadata writes for immutable private MLOps artifacts."""

import hashlib
import json
import re
from datetime import datetime, timezone

from .models import MLOpsBenchmark, MLOpsRun, MLOpsMonitor


def parse_time(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Timestamp must include timezone")
    return parsed


def canonical_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def validate_run_record(record):
    run_id = record.get("run_id", "")
    if not re.fullmatch(r"[A-Za-z0-9-]{1,100}", run_id):
        raise ValueError("Invalid run ID")
    if record.get("status") not in {"diagnostic", "candidate", "approved"}:
        raise ValueError("Invalid run status")
    if not record.get("model_version") or not record.get("dataset_version"):
        raise ValueError("Missing model or dataset version")
    if not re.fullmatch(r"[0-9a-f]{64}", record.get("model_sha256", "")):
        raise ValueError("Missing model artifact hash")
    parse_time(record["recorded_at"])
    files = record.get("files", {})
    if not isinstance(files, dict) or not {"manifest", "audit", "evaluation", "benchmark"} <= set(files):
        raise ValueError("Missing run evidence files")
    for item in files.values():
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", item.get("file", "")) or not re.fullmatch(r"[0-9a-f]{64}", item.get("sha256", "")):
            raise ValueError("Invalid run evidence file")
    if record["status"] != "diagnostic":
        if (record.get("schema_version") != 2 or not record.get("group_split_verified")
            or not re.fullmatch(r"[0-9a-f]{64}", record.get("dataset_manifest_sha256") or "")
            or not re.fullmatch(r"[0-9a-f]{64}", record.get("quality_gates_sha256") or "")
            or not {"training_history", "training_config", "quality_gates", "release_gate", "model_artifact"} <= set(files)):
            raise ValueError("Candidate run lacks reproducibility evidence")
    if record["status"] == "approved" and record.get("release_gate_passed") is not True:
        raise ValueError("Approved run failed quality gate")


def register_run(db, record, environment, object_prefix, record_sha256):
    validate_run_record(record)
    run_id = record["run_id"]
    existing = db.get(MLOpsRun, run_id)
    if existing:
        if existing.record_sha256 != record_sha256 or existing.object_prefix != object_prefix:
            raise ValueError(f"Run ID already exists with different evidence: {run_id}")
        return existing
    summary = {key: record.get(key) for key in (
        "run_id", "status", "recorded_at", "model_version", "model_sha256", "dataset_version", "dataset_manifest_sha256",
        "group_split_verified", "accuracy", "macro_f1", "ece_10_bins", "accepted_coverage",
        "accepted_accuracy", "inference_p95_ms", "release_gate_passed", "quality_gates_sha256",
        "git_commit", "git_dirty",
    )}
    item = MLOpsRun(run_id=run_id, environment=environment, status=record["status"],
                    model_version=record["model_version"], dataset_version=record["dataset_version"],
                    recorded_at=parse_time(record["recorded_at"]), object_prefix=object_prefix,
                    record_sha256=record_sha256, summary_json=json.dumps(summary, ensure_ascii=False))
    db.add(item)
    db.commit()
    return item


def register_benchmark(db, benchmark_id, report, environment, object_key, report_sha256):
    existing = db.get(MLOpsBenchmark, benchmark_id)
    if existing:
        if existing.report_sha256 != report_sha256 or existing.object_key != object_key:
            raise ValueError(f"Benchmark ID already exists with different evidence: {benchmark_id}")
        return existing
    summary = {key: report.get(key) for key in (
        "runs", "concurrency", "unique_image_count", "success_rate", "throughput_rps",
        "latency_ms", "server_processing_ms", "queue_ms", "inference_ms", "model_versions",
        "measured_at", "benchmark_run_id", "image_bytes_min", "image_bytes_max",
        "started_at", "finished_at", "errors", "warmup_errors",
    )}
    item = MLOpsBenchmark(benchmark_id=benchmark_id, environment=environment,
        model_version=(report.get("model_versions") or ["unknown"])[0],
        recorded_at=datetime.now(timezone.utc), object_key=object_key,
        report_sha256=report_sha256, summary_json=json.dumps(summary, ensure_ascii=False))
    db.add(item)
    db.commit()
    return item


def register_monitor(db, snapshot, environment):
    content = canonical_bytes(snapshot)
    snapshot_id = hashlib.sha256(content).hexdigest()[:24]
    existing = db.get(MLOpsMonitor, snapshot_id)
    if existing:
        return existing
    item = MLOpsMonitor(snapshot_id=snapshot_id, environment=environment,
        checked_at=parse_time(snapshot["checked_at"]), status=snapshot["status"],
        snapshot_json=json.dumps(snapshot, ensure_ascii=False))
    db.add(item)
    db.commit()
    return item
