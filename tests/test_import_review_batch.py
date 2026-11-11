import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from training.import_review_batch import IMPORT_FIELDS, export_template, import_batch


ROOT = Path(__file__).resolve().parents[1]


class BatchReviewImportTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.raw = self.root / "raw"
        for relative, content in (("VN/200/a.jpg", b"same-note-photo"),
                                  ("VN/200/b.jpg", b"same-note-photo"),
                                  ("VN/500/c.jpg", b"other-photo")):
            path = self.raw / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        self.review = self.root / "review.csv"
        result = subprocess.run([sys.executable, str(ROOT / "training/inventory_raw.py"),
                                 "--raw", str(self.raw), "--review-csv", str(self.review),
                                 "--output", str(self.root / "inventory.json")],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.template = self.root / "source-log.csv"
        self.assertEqual(export_template(self.review, self.template), 3)
        self.queue = self.root / "queue.csv"
        self.status = self.root / "status.json"
        self.report = self.root / "import-report.json"

    def fill(self, *, group_b="session-17", source_ref="capture-log/row-17"):
        with self.template.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        for row in rows:
            if row["path"].endswith("c.jpg"):
                continue
            row.update(group_id=group_b if row["path"].endswith("b.jpg") else "session-17",
                       source_ref=source_ref, reviewed_by="owner-1",
                       reviewed_at="2026-11-11T12:00:00+07:00")
        with self.template.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=IMPORT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def run_import(self, apply=False):
        return import_batch(self.raw, self.review, self.template, self.queue,
                            self.status, self.report, apply)

    def test_check_apply_and_replay_are_safe(self):
        self.fill()
        before = hashlib.sha256(self.review.read_bytes()).hexdigest()
        check = self.run_import()
        self.assertEqual((check["new_decisions"], check["remaining_group_ids"]), (2, 1))
        self.assertEqual(hashlib.sha256(self.review.read_bytes()).hexdigest(), before)
        applied = self.run_import(apply=True)
        self.assertEqual(applied["new_decisions"], 2)
        self.assertTrue(Path(applied["backup_path"]).is_file())
        self.assertEqual(json.loads(self.status.read_text())["pending_group_ids"], 1)
        after = hashlib.sha256(self.review.read_bytes()).hexdigest()
        replay = self.run_import(apply=True)
        self.assertEqual(replay["new_decisions"], 0)
        self.assertEqual(hashlib.sha256(self.review.read_bytes()).hexdigest(), after)

    def test_conflicting_group_or_missing_source_rejects_whole_batch(self):
        self.fill(group_b="session-18")
        before = self.review.read_bytes()
        with self.assertRaisesRegex(ValueError, "Exact image copies disagree"):
            self.run_import(apply=True)
        self.assertEqual(self.review.read_bytes(), before)
        self.fill(source_ref="")
        with self.assertRaisesRegex(ValueError, "Source reference and reviewer"):
            self.run_import(apply=True)
        self.assertEqual(self.review.read_bytes(), before)

    def test_changed_raw_image_rejects_batch(self):
        self.fill()
        (self.raw / "VN/200/a.jpg").write_bytes(b"changed-after-inventory")
        with self.assertRaisesRegex(ValueError, "Raw image changed"):
            self.run_import(apply=True)
        self.assertEqual(export_template(self.review, self.root / "second-template.csv"), 3)

    def test_exact_copy_is_not_auto_attested_from_hash(self):
        self.fill()
        with self.template.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        for row in rows:
            if row["path"].endswith("b.jpg"):
                row.update(group_id="", source_ref="", reviewed_by="", reviewed_at="")
        with self.template.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=IMPORT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        result = self.run_import(apply=True)
        self.assertEqual((result["new_decisions"], result["remaining_group_ids"]), (1, 2))
        with self.review.open(newline="", encoding="utf-8") as source:
            by_path = {row["path"]: row for row in csv.DictReader(source)}
        self.assertEqual(by_path["VN/200/b.jpg"]["group_id"], "")


if __name__ == "__main__":
    unittest.main()
