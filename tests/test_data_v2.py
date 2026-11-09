import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DatasetV2Test(unittest.TestCase):
    def run_script(self, script, *args, okay=True):
        result = subprocess.run([sys.executable, str(ROOT / script), *map(str, args)],
                                capture_output=True, text=True, check=False)
        if okay and result.returncode:
            self.fail(result.stderr + result.stdout)
        return result

    def test_review_survives_inventory_and_manifest_detects_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            for index in range(3):
                path = raw / "VN" / "200" / f"note-{index}.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"distinct-{index}".encode())
            review = root / "review.csv"
            inventory = root / "inventory.json"
            self.run_script("training/inventory_raw.py", "--raw", raw, "--review-csv", review, "--output", inventory)
            with review.open(newline="") as source:
                rows = list(csv.DictReader(source))
            for index, row in enumerate(rows):
                row.update(group_id=f"physical-note-{index}", reviewed_by="human-reviewer",
                           reviewed_at="2026-11-08T12:00:00Z")
            with review.open("w", newline="") as target:
                writer = csv.DictWriter(target, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            self.run_script("training/inventory_raw.py", "--raw", raw, "--review-csv", review, "--output", inventory)
            with review.open(newline="") as source:
                self.assertEqual(next(csv.DictReader(source))["group_id"], "physical-note-0")
            queue = root / "queue.csv"
            status = root / "status.json"
            self.run_script("training/review_queue.py", "--review-csv", review, "--queue", queue, "--report", status, "--require-complete")
            self.assertEqual(json.loads(status.read_text())["pending_rows"], 0)
            curated = root / "curated"
            groups = root / "groups.csv"
            curation = root / "curation.json"
            self.run_script("training/curate_raw.py", "--raw", raw, "--review-csv", review,
                            "--output", curated, "--group-template", groups, "--report", curation)
            processed = root / "processed"
            self.run_script("training/split_grouped.py", "--raw", curated, "--groups", groups,
                            "--output", processed, "--dataset-version", "2.0.0",
                            "--curation-report", curation, "--attested-physical-groups")
            audit = root / "audit.json"
            self.run_script("training/audit_dataset.py", "--root", processed, "--output", audit)
            self.assertTrue(json.loads(audit.read_text())["dataset_manifest_verified"])
            self.assertEqual(json.loads(audit.read_text())["dataset_version"], "2.0.0")
            next(processed.rglob("*.jpg")).write_bytes(b"tampered")
            self.assertNotEqual(self.run_script("training/audit_dataset.py", "--root", processed,
                                                "--output", audit, okay=False).returncode, 0)
            self.assertFalse(json.loads(audit.read_text())["dataset_manifest_verified"])

    def test_unreviewed_conflict_cannot_be_published(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            for relative in ("VN/200/a.jpg", "VN/500/b.jpg"):
                path = raw / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"same-content")
            review = root / "review.csv"
            self.run_script("training/inventory_raw.py", "--raw", raw, "--review-csv", review,
                            "--output", root / "inventory.json")
            curated = root / "curated"
            groups = root / "groups.csv"
            curation = root / "curation.json"
            self.run_script("training/curate_raw.py", "--raw", raw, "--review-csv", review,
                            "--output", curated, "--group-template", groups, "--report", curation)
            result = self.run_script("training/split_grouped.py", "--raw", curated, "--groups", groups,
                                     "--output", root / "processed", "--dataset-version", "2.0.0",
                                     "--curation-report", curation, "--attested-physical-groups", okay=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Human review incomplete", result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()
