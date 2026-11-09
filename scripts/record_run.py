"""Store a reproducible local record of one model evaluation run."""

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args):
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("models/model-manifest.json"))
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--training-history", type=Path)
    parser.add_argument("--training-config", type=Path)
    parser.add_argument("--gates-file", type=Path)
    parser.add_argument("--release-gate", type=Path)
    parser.add_argument("--model", type=Path, help="Exact model artifact to archive with this run")
    parser.add_argument("--status", choices=["diagnostic", "candidate", "approved"], required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/runs"))
    parser.add_argument("--run-id", help="Optional pipeline run ID to keep all reports linked")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    if evaluation["model_version"] != manifest["version"] or evaluation["model_sha256"] != manifest["sha256"]:
        raise SystemExit("Evaluation does not match the model manifest")
    if benchmark["model_version"] != manifest["version"]:
        raise SystemExit("Benchmark does not match the model manifest")
    if args.status != "diagnostic":
        if (
            not audit.get("dataset_manifest_verified")
            or not audit.get("human_review_complete")
            or not audit["group_split_verified"]
            or not audit["class_sets_match"]
            or audit["exact_cross_split_duplicates"]
            or audit["conflicting_label_groups"]
            or evaluation["dataset_version"] != audit["dataset_version"]
        ):
            raise SystemExit("Non-diagnostic run requires a verified grouped dataset")
        if not all((args.training_history, args.training_config, args.gates_file, args.release_gate, args.model)):
            raise SystemExit("Non-diagnostic run requires training history, config, model, locked policy and gate report")
        gate_report = json.loads(args.release_gate.read_text(encoding="utf-8"))
        if (gate_report.get("model_version") != manifest["version"]
            or gate_report.get("dataset_version") != audit["dataset_version"]
            or gate_report.get("gates_sha256") != sha256(args.gates_file)):
            raise SystemExit("Release gate report does not match inputs")
    else:
        gate_report = None
    if args.status == "approved" and manifest["release_status"] != "approved":
        raise SystemExit("Model manifest is not approved")
    if args.status == "approved" and not gate_report["passed"]:
        raise SystemExit("Release gate did not pass")
    if args.model and sha256(args.model) != manifest["sha256"]:
        raise SystemExit("Model artifact hash mismatch")

    run_id = args.run_id or (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    if not run_id.replace("-", "").isalnum():
        raise SystemExit("Invalid run ID")
    destination = args.output_dir / run_id
    files = {"manifest": args.manifest, "audit": args.audit, "evaluation": args.evaluation, "benchmark": args.benchmark}
    if args.training_history:
        files["training_history"] = args.training_history
    if args.training_config:
        files["training_config"] = args.training_config
    if args.gates_file:
        files["quality_gates"] = args.gates_file
    if args.release_gate:
        files["release_gate"] = args.release_gate
    if args.model:
        files["model_artifact"] = args.model
    if destination.exists():
        existing = json.loads((destination / "run.json").read_text(encoding="utf-8"))
        if (existing.get("status") != args.status or existing.get("model_sha256") != manifest["sha256"]
            or set(existing.get("files", {})) != set(files)
            or any(existing["files"][name]["sha256"] != sha256(path) for name, path in files.items())):
            raise SystemExit(f"Run ID already exists with different evidence: {run_id}")
        print(f"Run {run_id} already recorded with identical evidence")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(run_id + ".partial-" + uuid.uuid4().hex[:8])
    staging.mkdir(parents=True, exist_ok=False)
    copied = {}
    for name, source in files.items():
        target = staging / (name + source.suffix)
        shutil.copy2(source, target)
        copied[name] = {"file": target.name, "sha256": sha256(target)}

    record = {
        "schema_version": 2,
        "run_id": run_id,
        "status": args.status,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "model_version": manifest["version"],
        "model_sha256": manifest["sha256"],
        "dataset_version": audit["dataset_version"],
        "dataset_manifest_sha256": audit.get("dataset_manifest_sha256"),
        "evaluation_dataset_version": evaluation["dataset_version"],
        "group_split_verified": audit["group_split_verified"],
        "accuracy": evaluation["accuracy"],
        "macro_f1": evaluation["macro_f1"],
        "ece_10_bins": evaluation["ece_10_bins"],
        "accepted_coverage": evaluation["accepted_coverage"],
        "accepted_accuracy": evaluation["accepted_accuracy"],
        "inference_p95_ms": benchmark["p95_ms"],
        "release_gate_passed": gate_report["passed"] if gate_report else None,
        "quality_gates_sha256": sha256(args.gates_file) if args.gates_file else None,
        "training_config": json.loads(args.training_config.read_text(encoding="utf-8")) if args.training_config else None,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_dirty": bool(git_value("status", "--porcelain")),
        "files": copied,
    }
    (staging / "run.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    staging.rename(destination)
    print(f"Recorded {args.status} run {run_id} at {destination}")


if __name__ == "__main__":
    main()
