import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from api.database import Base
from api.mlops_registry import canonical_bytes, register_benchmark, register_monitor, register_run, validate_run_record
from api.models import MLOpsBenchmark, MLOpsMonitor, MLOpsRun


class MLOpsRegistryTest(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite://")
        Base.metadata.create_all(self.engine)

    def tearDown(self):
        self.engine.dispose()

    def test_run_is_idempotent_and_conflicts_on_changed_evidence(self):
        record = {"run_id": "run-001", "status": "diagnostic", "recorded_at": "2026-11-08T10:00:00Z",
                  "model_version": "1.1.0", "model_sha256": "a" * 64, "dataset_version": "legacy-unversioned",
                  "accuracy": 0.96, "macro_f1": 0.95, "group_split_verified": False,
                  "files": {name: {"file": f"{name}.json", "sha256": "a" * 64}
                            for name in ("manifest", "audit", "evaluation", "benchmark")}}
        with Session(self.engine) as db:
            self.assertEqual(register_run(db, record, "staging", "staging/runs/run-001/", "a" * 64).run_id, "run-001")
            register_run(db, record, "staging", "staging/runs/run-001/", "a" * 64)
            self.assertEqual(db.query(MLOpsRun).count(), 1)
            with self.assertRaises(ValueError):
                register_run(db, record, "staging", "staging/runs/run-001/", "b" * 64)

    def test_archived_diagnostic_schema_and_invalid_approval(self):
        from pathlib import Path
        fixture = Path(__file__).resolve().parents[1] / "docs/evidence/2026-11-08-diagnostic-run.json"
        record = json.loads(fixture.read_text(encoding="utf-8"))
        validate_run_record(record)
        record["status"] = "approved"
        with self.assertRaises(ValueError):
            validate_run_record(record)

    def test_benchmark_and_monitor_are_queryable(self):
        report = {"runs": 100, "concurrency": 4, "model_versions": ["1.1.0"],
                  "success_rate": 1, "throughput_rps": 2.5, "latency_ms": {"p95": 2100}}
        snapshot = {"checked_at": datetime.now(timezone.utc).isoformat(),
                    "status": "insufficient_data", "drift_status": "insufficient_data"}
        with Session(self.engine) as db:
            register_benchmark(db, "bench-001", report, "staging", "staging/benchmarks/bench-001.json", "b" * 64)
            first = register_monitor(db, snapshot, "staging")
            second = register_monitor(db, snapshot, "staging")
            self.assertEqual(first.snapshot_id, second.snapshot_id)
            self.assertEqual(db.query(MLOpsBenchmark).count(), 1)
            self.assertEqual(db.query(MLOpsMonitor).count(), 1)
            self.assertEqual(hashlib.sha256(canonical_bytes(snapshot)).hexdigest()[:24], first.snapshot_id)

    def test_record_run_replay_is_idempotent(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            data = {
                "manifest": {"version": "diag-1", "sha256": "a" * 64, "release_status": "candidate"},
                "audit": {"dataset_version": "legacy-unversioned", "group_split_verified": False},
                "evaluation": {"model_version": "diag-1", "model_sha256": "a" * 64,
                               "dataset_version": "legacy-unversioned", "accuracy": 0.7, "macro_f1": 0.6,
                               "ece_10_bins": 0.1, "accepted_coverage": 0.5, "accepted_accuracy": 0.8},
                "benchmark": {"model_version": "diag-1", "p95_ms": 100},
            }
            for name, content in data.items():
                (folder / f"{name}.json").write_text(json.dumps(content))
            command = [sys.executable, str(root / "scripts/record_run.py"),
                       "--manifest", str(folder / "manifest.json"), "--audit", str(folder / "audit.json"),
                       "--evaluation", str(folder / "evaluation.json"), "--benchmark", str(folder / "benchmark.json"),
                       "--status", "diagnostic", "--run-id", "repeat-001", "--output-dir", str(folder / "runs")]
            subprocess.run(command, check=True, capture_output=True)
            first = (folder / "runs/repeat-001/run.json").read_bytes()
            subprocess.run(command, check=True, capture_output=True)
            self.assertEqual((folder / "runs/repeat-001/run.json").read_bytes(), first)
            data["evaluation"]["accuracy"] = 0.8
            (folder / "evaluation.json").write_text(json.dumps(data["evaluation"]))
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)


if __name__ == "__main__":
    unittest.main()
