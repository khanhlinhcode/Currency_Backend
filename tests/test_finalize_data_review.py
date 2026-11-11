import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.finalize_data_review import finalize


ROOT = Path(__file__).resolve().parents[1]


class FinalizeReviewTest(unittest.TestCase):
    def test_incomplete_review_stops_and_complete_review_builds_versioned_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            for index in range(3):
                path = raw / "VN" / "200" / f"note-{index}.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"different-banknote-{index}".encode())
            review = root / "review.csv"
            subprocess.run([sys.executable, str(ROOT / "training/inventory_raw.py"),
                            "--raw", str(raw), "--review-csv", str(review),
                            "--output", str(root / "inventory.json")], check=True, capture_output=True)
            output = root / "snapshots"
            with self.assertRaisesRegex(ValueError, "Human review incomplete"):
                finalize(raw, review, output, "2.0.0")
            self.assertFalse((output / "2.0.0").exists())
            with review.open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            for index, row in enumerate(rows):
                row.update(group_id=f"physical-note-{index}", reviewed_by="human-reviewer",
                           reviewed_at="2026-11-11T00:00:00Z")
            with review.open("w", newline="", encoding="utf-8") as target:
                writer = csv.DictWriter(target, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            final = finalize(raw, review, output, "2.0.0")
            self.assertTrue((final / "processed/dataset-manifest.json").exists())
            self.assertTrue(json.loads((final / "audit.json").read_text())["dataset_manifest_verified"])
            self.assertEqual(json.loads((final / "snapshot.json").read_text())["version"], "2.0.0")
            with self.assertRaises(FileExistsError):
                finalize(raw, review, output, "2.0.0")


if __name__ == "__main__":
    unittest.main()
