import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.check_staging_benchmark import check
from scripts.drift_report import jensen_shannon


ROOT = Path(__file__).resolve().parents[1]


class ReleaseGateTest(unittest.TestCase):
    def test_raw_curation_quarantines_conflicts_without_changing_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            for relative, content in (
                ("VN/200/a.jpg", b"same-note"),
                ("VN/200/b.jpg", b"same-note"),
                ("VN/200/c.jpg", b"conflict-note"),
                ("VN/500/d.jpg", b"conflict-note"),
            ):
                path = raw / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            inventory = root / "inventory.json"
            review = root / "review.csv"
            curated = root / "curated"
            report = root / "curation.json"
            groups = root / "groups.csv"
            subprocess.run(
                [sys.executable, str(ROOT / "training/inventory_raw.py"), "--raw", str(raw), "--output", str(inventory), "--review-csv", str(review)],
                check=True, capture_output=True, text=True,
            )
            subprocess.run(
                [sys.executable, str(ROOT / "training/curate_raw.py"), "--raw", str(raw), "--review-csv", str(review), "--output", str(curated), "--group-template", str(groups), "--report", str(report)],
                check=True, capture_output=True, text=True,
            )
            result = json.loads(report.read_text())
            self.assertEqual(result["curated_images"], 1)
            self.assertEqual(result["quarantined_images"], 2)
            self.assertEqual(result["exact_duplicate_copies_removed"], 1)
            self.assertEqual(len(list(curated.rglob("*.jpg"))), 1)
            self.assertEqual((raw / "VN/200/a.jpg").read_bytes(), b"same-note")
            self.assertNotEqual(os.stat(raw / "VN/200/a.jpg").st_ino, os.stat(curated / "VN/200/a.jpg").st_ino)

    def test_group_audit_rejects_manifest_that_does_not_cover_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "processed"
            assignments = []
            for split, filename in (("train", "a.jpg"), ("val", "b.jpg"), ("test", "c.jpg")):
                source = f"VN/000200/{filename}"
                output_name = hashlib.sha256(source.encode()).hexdigest()[:12] + ".jpg"
                destination = root / split / "VN_000200" / output_name
                destination.parent.mkdir(parents=True)
                destination.write_bytes(split.encode())
                assignments.append({"split": split, "label": "VN_000200", "group_id": filename, "files": [source]})
            manifest = {"group_type": "physical_banknote_or_capture_session", "assignments": assignments}
            (root / "split-manifest.json").write_text(json.dumps(manifest))
            report = Path(directory) / "audit.json"

            def audit():
                return subprocess.run(
                    [sys.executable, str(ROOT / "training/audit_dataset.py"), "--root", str(root), "--output", str(report)],
                    capture_output=True, text=True, check=False,
                )

            self.assertEqual(audit().returncode, 0)
            self.assertTrue(json.loads(report.read_text())["group_split_verified"])
            assignments[0]["files"] = ["VN/000200/different.jpg"]
            (root / "split-manifest.json").write_text(json.dumps(manifest))
            self.assertNotEqual(audit().returncode, 0)
            self.assertFalse(json.loads(report.read_text())["split_manifest_integrity"])
            assignments[0]["files"] = ["VN/000200/a.jpg"]
            manifest["group_type"] = "unverified_group_ids"
            (root / "split-manifest.json").write_text(json.dumps(manifest))
            self.assertNotEqual(audit().returncode, 0)
            self.assertFalse(json.loads(report.read_text())["group_split_verified"])

    def test_staging_gate_rejects_small_or_slow_samples(self):
        targets = json.loads((ROOT / "docs/staging-targets.json").read_text())
        report = {
            "runs": 100, "unique_image_count": 3, "concurrency": 4, "warmup_errors": 0, "success_rate": 1.0,
            "model_versions": ["1.1.0"], "throughput_rps": 5,
            "latency_ms": {"p95": 1000, "p99": 1500},
            "server_processing_ms": {"p95": 800},
            "queue_ms": {"p95": 300},
            "inference_ms": {"p95": 400},
        }
        self.assertTrue(all(check(report, targets, "1.1.0").values()))
        report["runs"] = 5
        report["latency_ms"]["p95"] = 3000
        checks = check(report, targets, "1.1.0")
        self.assertFalse(checks["enough measured requests"])
        self.assertFalse(checks["client p95"])

    def test_model_promotion_requires_clean_data_and_measured_quality(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            manifest = {"version": "test-1", "sha256": "a" * 64, "release_status": "candidate", "class_names": ["A", "B"]}
            audit = {"dataset_manifest_verified": True, "dataset_manifest_sha256": "d" * 64, "human_review_complete": True, "group_split_verified": True, "class_sets_match": True, "exact_cross_split_duplicates": 0, "conflicting_label_groups": 0, "dataset_version": "dataset-1"}
            policy = {"schema_version": 1, "status": "locked", "dataset_version": "dataset-1",
                      "approved_by": "reviewer", "locked_at": "2026-01-01T00:00:00Z",
                      "thresholds": {"min_accuracy": 0.95, "min_macro_f1": 0.93,
                                     "min_class_recall": 0.85, "max_ece": 0.1,
                                     "min_accepted_accuracy": 0.97, "max_p95_ms": 250}}
            gates_path = folder / "gates.json"
            gates_path.write_text(json.dumps(policy))
            evaluation = {
                "evaluated_at": "2026-01-02T00:00:00Z",
                "gates_sha256": hashlib.sha256(gates_path.read_bytes()).hexdigest(),
                "dataset_version": "dataset-1", "model_version": "test-1", "model_sha256": "a" * 64,
                "accuracy": 0.98, "macro_f1": 0.97, "ece_10_bins": 0.02,
                "accepted_accuracy": 0.98, "accepted_coverage": 0.95, "test_samples": 100,
                "per_class": {label: {"recall": 0.95, "support": 50} for label in ("A", "B")},
            }
            benchmark = {"runs": 100, "model_version": "test-1", "p95_ms": 100}
            paths = {name: folder / (name + ".json") for name in ("manifest", "audit", "evaluation", "benchmark")}

            def promote():
                for name, value in (("manifest", manifest), ("audit", audit), ("evaluation", evaluation), ("benchmark", benchmark)):
                    paths[name].write_text(json.dumps(value))
                return subprocess.run([
                    sys.executable, str(ROOT / "scripts/promote_release.py"),
                    "--manifest", str(paths["manifest"]), "--audit", str(paths["audit"]),
                    "--evaluation", str(paths["evaluation"]), "--benchmark", str(paths["benchmark"]),
                    "--gates-file", str(gates_path), "--report", str(folder / "decision.json"),
                ], capture_output=True, text=True, check=False)

            self.assertEqual(promote().returncode, 0)
            self.assertEqual(json.loads(paths["manifest"].read_text())["release_status"], "approved")
            audit["exact_cross_split_duplicates"] = 1
            rejected = promote()
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("no exact cross-split duplicates", rejected.stderr)

    def test_distribution_shift_is_zero_only_for_matching_distribution(self):
        self.assertEqual(jensen_shannon({"A": 2, "B": 2}, {"A": 1, "B": 1}), 0)
        self.assertGreater(jensen_shannon({"A": 4}, {"B": 4}), 0.9)


if __name__ == "__main__":
    unittest.main()
