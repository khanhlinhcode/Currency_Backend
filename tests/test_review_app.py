import csv
import json
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from training.review_app import ReviewStore, make_server


ROOT = Path(__file__).resolve().parents[1]


class HumanReviewAppTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.raw = self.root / "raw"
        for relative in ("VN/200/note-a.jpg", "VN/500/note-b.jpg"):
            target = self.raw / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"same-source-image")
        self.review = self.root / "group-review.csv"
        result = subprocess.run([sys.executable, str(ROOT / "training/inventory_raw.py"),
                                 "--raw", str(self.raw), "--review-csv", str(self.review),
                                 "--output", str(self.root / "inventory.json")],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.store = ReviewStore(self.raw, self.review, self.root / "queue.csv", self.root / "status.json")
        self.server = make_server(self.store, 0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)
        self.temporary.cleanup()

    def request(self, path, data=None, headers=None):
        request = urllib.request.Request(self.base + path, data=data, headers=headers or {})
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.read()

    def save(self, path, label, group, *, csrf=None):
        row = self.store.by_path[path]
        body = urllib.parse.urlencode({"csrf": csrf if csrf is not None else self.server.csrf,
            "path": path, "expected_reviewed_at": row["reviewed_at"],
            "reviewed_label": label, "group_id": group, "reviewed_by": "reviewer-1",
            "review_notes": "Checked source record", "action": "stay"}).encode()
        return self.request("/save", body, {"Origin": self.base})

    def test_conflict_review_is_saved_and_queue_reaches_zero(self):
        first, second = sorted(self.store.by_path)
        status, page = self.request("/")
        self.assertEqual(status, 200)
        self.assertIn(b"Duy", page)
        status, data = self.request("/image?path=" + urllib.parse.quote(first))
        self.assertEqual((status, data), (200, b"same-source-image"))
        self.assertEqual(self.save(first, "VN_200", "session-1", csrf="wrong")[0], 403)
        self.assertEqual(self.save(first, "VN_200", "session-1")[0], 200)
        self.assertEqual(self.save(second, "VN_500", "session-1")[0], 409)
        self.assertEqual(self.save(second, "VN_200", "session-2")[0], 409)
        self.assertEqual(self.save(second, "VN_200", "session-1")[0], 200)
        with self.review.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        self.assertEqual({row["group_id"] for row in rows}, {"session-1"})
        self.assertTrue(all(row["reviewed_at"].endswith("+00:00") for row in rows))
        self.assertEqual(json.loads((self.root / "status.json").read_text())["pending_rows"], 0)
        self.assertTrue(self.review.with_suffix(".initial-backup.csv").exists())

    def test_external_edits_and_raw_image_changes_block_review(self):
        first = sorted(self.store.by_path)[0]
        (self.raw / first).write_bytes(b"changed")
        self.assertEqual(self.request("/image?path=" + urllib.parse.quote(first))[0], 409)
        self.assertEqual(self.save(first, "VN_200", "session-1")[0], 409)
        (self.raw / first).write_bytes(b"same-source-image")
        with self.review.open("a", encoding="utf-8") as target:
            target.write("\n")
        self.assertEqual(self.save(first, "VN_200", "session-1")[0], 409)


if __name__ == "__main__":
    unittest.main()
