"""Audit, train, evaluate, benchmark and record one isolated model candidate."""

import argparse
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path


def run_step(name, command, environment=None):
    print(f"\n== {name} ==", flush=True)
    result = subprocess.run(command, env=environment, check=False)
    if result.returncode:
        raise SystemExit(f"Pipeline stopped at {name} (exit {result.returncode})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True, help="Audited grouped train/val/test root")
    parser.add_argument("--version", required=True, help="New model release version")
    parser.add_argument("--benchmark-image", type=Path, required=True)
    parser.add_argument("--benchmark-runs", type=int, default=100)
    parser.add_argument("--runs-root", type=Path, default=Path("results/pipeline"))
    parser.add_argument("--archive-to-s3", action="store_true", help="Archive the completed run in private staging storage")
    args = parser.parse_args()
    if args.benchmark_runs < 30 or not args.benchmark_image.is_file():
        parser.error("Provide an existing benchmark image and at least 30 benchmark runs")
    if not all((args.dataset_root / split).is_dir() for split in ("train", "val", "test")):
        parser.error("Dataset root must contain train, val and test directories")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    work = args.runs_root / run_id
    work.mkdir(parents=True, exist_ok=False)
    model_dir = Path("models/candidates") / run_id
    model_path = model_dir / "efficientnetb0_best.keras"
    manifest_path = work / "model-manifest.json"
    audit_path = work / "dataset-audit.json"
    history_path = work / "training-history.csv"
    evaluation_path = work / "release-evaluation.json"
    benchmark_path = work / "benchmark.json"
    python = sys.executable

    run_step("dataset audit", [python, "training/audit_dataset.py", "--root", str(args.dataset_root), "--output", str(audit_path)])
    train_environment = dict(os.environ, DATA_DIR=str(args.dataset_root), MODEL_DIR=str(model_dir), HISTORY_PATH=str(history_path))
    run_step("train", [python, "training/train_efficientnetb0.py"], train_environment)
    run_step("candidate manifest", [python, "scripts/create_candidate.py", "--model", str(model_path), "--audit", str(audit_path), "--train-dir", str(args.dataset_root / "train"), "--version", args.version, "--output", str(manifest_path)])
    run_step("held-out test evaluation", [python, "training/evaluate_release.py", "--manifest", str(manifest_path), "--model", str(model_path), "--test-dir", str(args.dataset_root / "test"), "--audit", str(audit_path), "--output", str(evaluation_path)])
    benchmark_environment = dict(os.environ, MODEL_PATH=str(model_path), MODEL_MANIFEST_PATH=str(manifest_path))
    run_step("warm inference benchmark", [python, "-m", "training.benchmark_inference", str(args.benchmark_image), "--runs", str(args.benchmark_runs), "--output", str(benchmark_path)], benchmark_environment)
    run_step("record run", [python, "scripts/record_run.py", "--manifest", str(manifest_path), "--audit", str(audit_path), "--evaluation", str(evaluation_path), "--benchmark", str(benchmark_path), "--training-history", str(history_path), "--status", "candidate", "--run-id", run_id])
    if args.archive_to_s3:
        run_step("archive run", [python, "-m", "scripts.archive_run", str(Path("results/runs") / run_id), "--target", "staging"])
    print(f"\nCandidate {args.version} completed: {manifest_path}")
    print("Review the metrics and apply scripts/promote_release.py only after the release gates pass")


if __name__ == "__main__":
    main()
