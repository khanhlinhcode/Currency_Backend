"""Check a pre-registered quality policy and approve only a verified model."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from quality_gates import evaluate_checks, load_gates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("models/model-manifest.json"))
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--gates-file", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=Path("results/release-gate.json"))
    parser.add_argument("--check-only", action="store_true", help="Write evidence without modifying the model manifest")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    gates, gates_sha256 = load_gates(args.gates_file, audit["dataset_version"])
    checks = evaluate_checks(manifest, audit, evaluation, benchmark, gates, gates_sha256)
    failures = [name for name, passed in checks.items() if not passed]
    decision = {
        "schema_version": 1, "checked_at": datetime.now(timezone.utc).isoformat(),
        "model_version": manifest["version"], "dataset_version": audit["dataset_version"],
        "gates_sha256": gates_sha256, "checks": checks, "passed": not failures,
        "failures": failures,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.check_only:
        print(f"Quality decision: {'passed' if not failures else 'rejected'}; report: {args.report}")
        return
    if failures:
        raise SystemExit("Release rejected: " + ", ".join(failures))
    manifest["dataset_version"] = audit["dataset_version"]
    manifest["dataset_manifest_sha256"] = audit["dataset_manifest_sha256"]
    manifest["quality_gates_sha256"] = gates_sha256
    manifest["test_metrics"] = {
        "test_samples": evaluation["test_samples"], "accuracy": evaluation["accuracy"],
        "macro_f1": evaluation["macro_f1"], "ece_10_bins": evaluation["ece_10_bins"],
        "accepted_coverage": evaluation["accepted_coverage"],
        "accepted_accuracy": evaluation["accepted_accuracy"],
        "min_class_recall": min(evaluation["per_class"][label]["recall"] for label in manifest["class_names"]),
    }
    manifest["benchmark"] = benchmark
    manifest["release_status"] = "approved"
    manifest["approved_at"] = datetime.now(timezone.utc).isoformat()
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Approved model {manifest['version']} with dataset {manifest['dataset_version']}")


if __name__ == "__main__":
    main()
