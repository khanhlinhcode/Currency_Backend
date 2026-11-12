import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.finalize_data_review import finalize
from training.evaluate_onnx import release_grade_preflight


ROOT = Path(__file__).resolve().parents[1]


class ONNXReleaseGradePreflightTest(unittest.TestCase):
    def test_clean_v2_passes_and_changed_image_or_legacy_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            for number in range(3):
                image = raw / "VN" / "200" / f"note-{number}.jpg"
                image.parent.mkdir(parents=True, exist_ok=True)
                image.write_bytes(f"independent-physical-note-{number}".encode())
            review = root / "review.csv"
            subprocess.run([sys.executable, str(ROOT / "training/inventory_raw.py"),
                            "--raw", str(raw), "--review-csv", str(review),
                            "--output", str(root / "inventory.json")],
                           check=True, capture_output=True)
            with review.open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            for index, row in enumerate(rows):
                row.update(group_id=f"note-{index}", reviewed_by="human-reviewer",
                           reviewed_at="2026-11-11T00:00:00Z")
            with review.open("w", newline="", encoding="utf-8") as target:
                writer = csv.DictWriter(target, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            snapshot = finalize(raw, review, root / "snapshots", "test-v2")
            processed = snapshot / "processed"
            audit = json.loads((snapshot / "audit.json").read_text())
            manifest = {"dataset_version": audit["dataset_version"],
                        "dataset_manifest_sha256": audit["dataset_manifest_sha256"]}
            gates = {"schema_version": 1, "status": "locked", "dataset_version": "test-v2",
                     "approved_by": "human-reviewer", "locked_at": "2026-01-01T00:00:00Z",
                     "thresholds": {"min_accuracy": 0.9, "min_macro_f1": 0.9,
                                    "min_class_recall": 0.8, "max_ece": 0.1,
                                    "min_accepted_accuracy": 0.9, "max_p95_ms": 250}}
            policy = root / "gates.json"
            policy.write_text(json.dumps(gates), encoding="utf-8")
            self.assertEqual(release_grade_preflight(processed, manifest, audit, policy),
                             hashlib.sha256(policy.read_bytes()).hexdigest())
            legacy = dict(audit, dataset_version="legacy-unversioned", human_review_complete=False)
            with self.assertRaisesRegex(ValueError, "verified grouped v2"):
                release_grade_preflight(processed, manifest, legacy, policy)
            next(processed.rglob("*.jpg")).write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "Fresh dataset audit failed"):
                release_grade_preflight(processed, manifest, audit, policy)


if __name__ == "__main__":
    unittest.main()
