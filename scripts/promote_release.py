"""Approve a model only when dataset, test and latency gates pass."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("models/model-manifest.json"))
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--min-accuracy", type=float, required=True)
    parser.add_argument("--min-macro-f1", type=float, required=True)
    parser.add_argument("--min-class-recall", type=float, required=True)
    parser.add_argument("--max-ece", type=float, required=True)
    parser.add_argument("--min-accepted-accuracy", type=float, required=True)
    parser.add_argument("--max-p95-ms", type=float, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))

    checks = {
        "group split verified": audit["group_split_verified"],
        "class sets match": audit["class_sets_match"],
        "no exact cross-split duplicates": audit["exact_cross_split_duplicates"] == 0,
        "no conflicting labels": audit["conflicting_label_groups"] == 0,
        "dataset version matches": evaluation["dataset_version"] == audit["dataset_version"],
        "model version matches": evaluation["model_version"] == manifest["version"],
        "model checksum matches": evaluation["model_sha256"] == manifest["sha256"],
        "test accuracy": evaluation["accuracy"] >= args.min_accuracy,
        "test macro F1": evaluation["macro_f1"] >= args.min_macro_f1,
        "per-class recall": all(
            evaluation["per_class"][label]["recall"] >= args.min_class_recall
            and evaluation["per_class"][label]["support"] > 0
            for label in manifest["class_names"]
        ),
        "confidence calibration": evaluation["ece_10_bins"] <= args.max_ece,
        "accepted accuracy": evaluation["accepted_accuracy"] is not None and evaluation["accepted_accuracy"] >= args.min_accepted_accuracy,
        "benchmark sample size": benchmark["runs"] >= 30,
        "benchmark model version": benchmark["model_version"] == manifest["version"],
        "benchmark p95": benchmark["p95_ms"] <= args.max_p95_ms,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise SystemExit("Release rejected: " + ", ".join(failures))

    manifest["dataset_version"] = audit["dataset_version"]
    manifest["test_metrics"] = {
        "test_samples": evaluation["test_samples"],
        "accuracy": evaluation["accuracy"],
        "macro_f1": evaluation["macro_f1"],
        "ece_10_bins": evaluation["ece_10_bins"],
        "accepted_coverage": evaluation["accepted_coverage"],
        "accepted_accuracy": evaluation["accepted_accuracy"],
        "min_class_recall": min(evaluation["per_class"][label]["recall"] for label in manifest["class_names"]),
    }
    manifest["benchmark"] = benchmark
    manifest["release_status"] = "approved"
    manifest["approved_at"] = datetime.now(timezone.utc).isoformat()
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Approved model {manifest['version']} with dataset {manifest['dataset_version'][:12]}")


if __name__ == "__main__":
    main()
