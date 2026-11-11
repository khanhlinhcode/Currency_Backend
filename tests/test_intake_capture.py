import csv
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.finalize_data_review import finalize
from training.intake_capture import CAPTURE_FIELDS, export_template, intake, verify_intake
from training.review_queue import generate_review_queue


ROOT = Path(__file__).resolve().parents[1]


class CaptureIntakeTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        pictures = [("s1/front.jpg", (240, 100, 50)),
                    ("s2/other.jpg", (30, 200, 80)),
                    ("s3/third.jpg", (25, 40, 220))]
        for name, color in pictures:
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (64, 48), color).save(path, format="JPEG")
        duplicate = self.source / "s1/front-copy.jpg"
        shutil.copyfile(self.source / "s1/front.jpg", duplicate)
        self.capture_csv = self.root / "capture.csv"
        self.rows = []
        for name, group in (("s1/front.jpg", "note-001"),
                            ("s1/front-copy.jpg", "note-001"),
                            ("s2/other.jpg", "note-002"),
                            ("s3/third.jpg", "note-003")):
            self.rows.append({"source_path": name,
                              "sha256": hashlib.sha256((self.source / name).read_bytes()).hexdigest(),
                              "label": "VN_000200", "group_id": group,
                              "source_ref": f"ledger-1/{group}", "captured_by": "camera-operator",
                              "captured_at": "2026-11-11T09:00:00+07:00",
                              "reviewed_by": "reviewer-1", "reviewed_at": "2026-11-11T10:00:00+07:00",
                              "review_notes": "Checked original banknote"})
        self.write_rows()
        self.output = self.root / "intakes" / "batch-001"

    def write_rows(self):
        with self.capture_csv.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=CAPTURE_FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)

    def run_intake(self, apply=False):
        return intake(self.source, self.capture_csv, self.output, "batch-001",
                      ROOT / "models/model-manifest.json", apply=apply)

    def test_template_hashes_images_without_assigning_labels_or_groups(self):
        template = self.root / "blank-capture.csv"
        self.assertEqual(export_template(self.source, template), 4)
        with template.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        self.assertEqual({row["source_path"] for row in rows},
                         {row["source_path"] for row in self.rows})
        self.assertTrue(all(not row["label"] and not row["group_id"] for row in rows))
        self.assertTrue(all(row["sha256"] == hashlib.sha256(
            (self.source / row["source_path"]).read_bytes()).hexdigest() for row in rows))

    def test_valid_batch_reaches_audited_v2_snapshot_without_inferred_groups(self):
        checked = self.run_intake()
        self.assertEqual((checked["images"], checked["physical_groups"]), (4, 3))
        self.assertFalse(self.output.exists())
        applied = self.run_intake(apply=True)
        self.assertTrue(self.output.is_dir())
        self.assertTrue(verify_intake(self.output)["verified"])
        self.assertEqual(applied["manifest_sha256"], json.loads(
            (self.output / "intake-manifest.json").read_text())["manifest_sha256"])
        status = generate_review_queue(self.output / "review.csv", self.root / "queue.csv",
                                       self.root / "status.json")
        self.assertEqual(status["pending_rows"], 0)
        snapshot = finalize(self.output / "raw", self.output / "review.csv",
                            self.root / "versions", "capture-v2")
        audit = json.loads((snapshot / "audit.json").read_text())
        self.assertTrue(audit["dataset_manifest_verified"])
        self.assertTrue(audit["group_split_verified"])
        self.assertEqual(audit["duplicate_groups"], [])

    def test_verification_detects_changed_intake_image(self):
        self.run_intake(apply=True)
        image = next((self.output / "raw").rglob("*.jpg"))
        image.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Intake image changed"):
            verify_intake(self.output)

    def test_duplicate_copy_with_different_group_is_rejected(self):
        self.rows[1]["group_id"] = "note-999"
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "Exact image copies disagree"):
            self.run_intake(apply=True)
        self.assertFalse(self.output.exists())

    def test_missing_source_record_and_bad_hash_are_rejected(self):
        self.rows.pop()
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "does not cover source images"):
            self.run_intake(apply=True)
        self.rows[0]["sha256"] = "0" * 64
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.run_intake(apply=True)
        self.assertFalse(self.output.exists())

    def test_group_across_labels_and_missing_provenance_are_rejected(self):
        self.rows[2]["group_id"] = "note-001"
        self.rows[2]["label"] = "VN_000500"
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "group_id spans classes"):
            self.run_intake()
        self.rows[2]["group_id"] = "note-002"
        self.rows[2]["label"] = "VN_000200"
        self.rows[2]["source_ref"] = ""
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "source_ref"):
            self.run_intake()


if __name__ == "__main__":
    unittest.main()
