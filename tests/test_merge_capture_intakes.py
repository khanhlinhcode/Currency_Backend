import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.finalize_data_review import finalize
from training.intake_capture import CAPTURE_FIELDS, intake
from training.merge_capture_intakes import merge_intakes, verify_merge
from training.review_queue import generate_review_queue


ROOT = Path(__file__).resolve().parents[1]


def jpeg(color):
    target = io.BytesIO()
    Image.new("RGB", (64, 48), color).save(target, format="JPEG")
    return target.getvalue()


class CaptureMergeTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.merge_output = self.root / "merged" / "campaign-v2"

    def make_intake(self, batch, entries):
        source_root = self.root / "sources" / batch
        source_root.mkdir(parents=True)
        rows = []
        for path_text, content, group in entries:
            path = source_root / path_text
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            rows.append({"source_path": path_text, "sha256": hashlib.sha256(content).hexdigest(),
                         "label": "VN_000200", "group_id": group,
                         "source_ref": f"ledger/{batch}/{group}",
                         "captured_by": "camera-operator", "captured_at": "2026-11-11T09:00:00+07:00",
                         "reviewed_by": "reviewer-1", "reviewed_at": "2026-11-11T10:00:00+07:00",
                         "review_notes": "Checked source ledger"})
        source_csv = self.root / f"{batch}.csv"
        with source_csv.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=CAPTURE_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        output = self.root / "intakes" / batch
        intake(source_root, source_csv, output, batch, ROOT / "models/model-manifest.json", apply=True)
        return output

    def test_verified_merge_finalizes_v2_dataset(self):
        first = self.make_intake("batch-001", [("batch-001/a.jpg", jpeg((240, 10, 30)), "b1-note-1"),
                                                   ("batch-001/b.jpg", jpeg((20, 220, 20)), "b1-note-2")])
        second = self.make_intake("batch-002", [("batch-002/c.jpg", jpeg((20, 20, 220)), "b2-note-3")])
        checked = merge_intakes([first, second], self.merge_output, "campaign-v2")
        self.assertEqual((checked["images"], checked["groups_per_class"]["VN_000200"]), (3, 3))
        self.assertFalse(self.merge_output.exists())
        applied = merge_intakes([first, second], self.merge_output, "campaign-v2", apply=True)
        self.assertTrue(verify_merge(self.merge_output)["verified"])
        self.assertEqual(applied["manifest_sha256"], json.loads(
            (self.merge_output / "merge-manifest.json").read_text())["manifest_sha256"])
        repeated = self.root / "merged" / "campaign-v2-repeated"
        repeated_result = merge_intakes([second, first], repeated, "campaign-v2", apply=True)
        self.assertEqual(applied["manifest_sha256"], repeated_result["manifest_sha256"])
        self.assertEqual((self.merge_output / "review.csv").read_bytes(),
                         (repeated / "review.csv").read_bytes())
        status = generate_review_queue(self.merge_output / "review.csv", self.root / "queue.csv",
                                       self.root / "status.json")
        self.assertEqual(status["pending_rows"], 0)
        dataset = finalize(self.merge_output / "raw", self.merge_output / "review.csv",
                           self.root / "datasets", "capture-merged-v2")
        audit = json.loads((dataset / "audit.json").read_text())
        self.assertTrue(audit["dataset_manifest_verified"])
        self.assertTrue(audit["group_split_verified"])
        self.assertEqual(audit["duplicate_groups"], [])

    def test_cross_campaign_group_and_source_path_collisions_are_rejected(self):
        first = self.make_intake("batch-001", [("shared/a.jpg", jpeg((240, 10, 30)), "note-1")])
        second = self.make_intake("batch-002", [("different/b.jpg", jpeg((20, 220, 20)), "note-1")])
        with self.assertRaisesRegex(ValueError, "group_id collision"):
            merge_intakes([first, second], self.merge_output, "campaign-v2", apply=True)
        self.assertFalse(self.merge_output.exists())

        third = self.make_intake("batch-003", [("shared/a.jpg", jpeg((20, 20, 220)), "note-3")])
        with self.assertRaisesRegex(ValueError, "Duplicate source_path"):
            merge_intakes([first, third], self.merge_output, "campaign-v2", apply=True)

    def test_cross_campaign_exact_content_reuse_is_rejected(self):
        content = jpeg((240, 10, 30))
        first = self.make_intake("batch-001", [("batch-001/a.jpg", content, "b1-note")])
        second = self.make_intake("batch-002", [("batch-002/b.jpg", content, "b2-note")])
        with self.assertRaisesRegex(ValueError, "Exact image content reused"):
            merge_intakes([first, second], self.merge_output, "campaign-v2", apply=True)
        self.assertFalse(self.merge_output.exists())

    def test_tampered_input_or_merged_output_is_rejected(self):
        first = self.make_intake("batch-001", [("batch-001/a.jpg", jpeg((240, 10, 30)), "b1-note")])
        second = self.make_intake("batch-002", [("batch-002/b.jpg", jpeg((20, 220, 20)), "b2-note")])
        image = next((first / "raw").rglob("*.jpg"))
        original = image.read_bytes()
        image.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "Intake image changed"):
            merge_intakes([first, second], self.merge_output, "campaign-v2", apply=True)
        image.write_bytes(original)
        merge_intakes([first, second], self.merge_output, "campaign-v2", apply=True)
        merged_image = next((self.merge_output / "raw").rglob("*.jpg"))
        merged_image.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "Merged image changed"):
            verify_merge(self.merge_output)


if __name__ == "__main__":
    unittest.main()
