import base64
import hashlib
import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from api.database import Base
from api.inference import MODEL_MANIFEST
from api.models import DevicePredictionEvent, DeviceRegistration, MLOpsRun, ReleaseEvent
from api.release_control import _switch, device_events, verified_release
from scripts.device_updater import sync_once
from scripts.release_signature import sign_payload


class FakeS3:
    def __init__(self, objects):
        self.objects = objects

    def get_object(self, Bucket, Key):
        from io import BytesIO
        return {"Body": BytesIO(self.objects[Key])}


class FakeResponse:
    def __init__(self, payload):
        from io import BytesIO
        self.stream = BytesIO(payload)

    def __enter__(self):
        return self.stream

    def __exit__(self, *_args):
        self.stream.close()


class ReleaseControlTest(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite://")
        Base.metadata.create_all(self.engine)

    def tearDown(self):
        self.engine.dispose()

    def test_legacy_run_cannot_be_activated_and_generation_is_compare_and_swap(self):
        with Session(self.engine) as db:
            db.add(MLOpsRun(run_id="legacy", environment="development", status="diagnostic",
                model_version="1.1.0", dataset_version="legacy-unversioned",
                recorded_at=datetime(2026, 11, 10, tzinfo=timezone.utc), object_prefix="development/runs/legacy/",
                record_sha256="a" * 64, summary_json="{}"))
            db.commit()
            with patch("api.release_control.STORAGE_BACKEND", "s3"):
                with self.assertRaises(HTTPException) as error:
                    _switch(db, "development", "legacy", 0, "admin", "initial release", "activate")
                self.assertEqual(error.exception.status_code, 409)
                with patch("api.release_control.verified_release", return_value={}):
                    first = _switch(db, "development", "approved-1", 0, "admin", "passed gate", "activate")
                    self.assertEqual(first["generation"], 1)
                    self.assertEqual(db.query(ReleaseEvent).count(), 1)
                    with self.assertRaises(HTTPException):
                        _switch(db, "development", "approved-2", 0, "admin", "passed gate", "activate")
                    second = _switch(db, "development", "approved-2", 1, "admin", "new model", "activate")
                    self.assertEqual(second["previous_run_id"], "approved-1")
                    self.assertEqual(db.query(ReleaseEvent).count(), 2)

    def test_archived_approval_requires_clean_audit(self):
        model = b"fake model content"
        digest = hashlib.sha256(model).hexdigest()
        payloads = {
            "manifest": {**MODEL_MANIFEST, "version": "2.0.0", "sha256": digest,
                         "release_status": "approved", "dataset_version": "2.0.0",
                         "dataset_manifest_sha256": "b" * 64},
            "audit": {"dataset_version": "2.0.0", "dataset_manifest_sha256": "b" * 64,
                      "dataset_manifest_verified": True, "human_review_complete": False,
                      "class_sets_match": True,
                      "group_split_verified": True, "exact_cross_split_duplicates": 0,
                      "conflicting_label_groups": 0},
            "release_gate": {"passed": True, "checks": {"audit": True},
                             "model_version": "2.0.0", "dataset_version": "2.0.0",
                             "gates_sha256": "c" * 64},
            "evaluation": {"accuracy": 0.99, "model_version": "2.0.0",
                           "model_sha256": digest, "dataset_version": "2.0.0"},
            "quality_gates": {"locked": True},
        }
        files = {name: {"file": name + ".json", "sha256": hashlib.sha256(json.dumps(value).encode()).hexdigest()}
                 for name, value in payloads.items()}
        files["model_artifact"] = {"file": "model.onnx", "sha256": digest}
        record = {"status": "approved", "run_id": "approved-1", "model_version": "2.0.0",
                  "model_sha256": digest, "dataset_version": "2.0.0",
                  "dataset_manifest_sha256": "b" * 64, "quality_gates_sha256": "c" * 64,
                  "release_gate_passed": True, "files": files}
        objects = {"development/runs/approved-1/" + name + ".json": json.dumps(value).encode()
                   for name, value in payloads.items()}
        objects["development/runs/approved-1/model.onnx"] = model
        record_bytes = json.dumps(record).encode()
        objects["development/runs/approved-1/run.json"] = record_bytes
        with Session(self.engine) as db:
            db.add(MLOpsRun(run_id="approved-1", environment="development", status="approved",
                model_version="2.0.0", dataset_version="2.0.0",
                recorded_at=datetime(2026, 11, 10, tzinfo=timezone.utc), object_prefix="development/runs/approved-1/",
                record_sha256=hashlib.sha256(record_bytes).hexdigest(), summary_json="{}"))
            db.commit()
            with patch("api.release_control._s3_client", return_value=FakeS3(objects)):
                with self.assertRaises(HTTPException):
                    verified_release(db, "approved-1")
                payloads["audit"]["human_review_complete"] = True
                audit_bytes = json.dumps(payloads["audit"]).encode()
                objects["development/runs/approved-1/audit.json"] = audit_bytes
                record["files"]["audit"]["sha256"] = hashlib.sha256(audit_bytes).hexdigest()
                record_bytes = json.dumps(record).encode()
                objects["development/runs/approved-1/run.json"] = record_bytes
                db.get(MLOpsRun, "approved-1").record_sha256 = hashlib.sha256(record_bytes).hexdigest()
                db.commit()
                self.assertEqual(verified_release(db, "approved-1")["model_sha256"], digest)

    def test_device_pointer_changes_only_after_verified_download(self):
        data = b"model bytes"
        digest = hashlib.sha256(data).hexdigest()
        contract = {"schema_version": 1, "channel": "staging", "model_format": "onnx",
                    **{name: MODEL_MANIFEST[name] for name in (
                        "class_names", "input_size", "preprocessing",
                        "resize_interpolation", "confidence_threshold")}}
        manifest = {**MODEL_MANIFEST, "release_status": "approved", "version": "2.0.0",
                    "dataset_version": "2.0.0", "sha256": digest}
        release = {"status": "available", "run_id": "approved-1", "generation": 1,
                   "channel": "staging", "model_version": "2.0.0", "dataset_version": "2.0.0",
                   "model_filename": "model.onnx", "model_sha256": digest,
                   "artifact_path": "/devices/id/artifact/approved-1", "manifest": manifest}
        self_test = [sys.executable, "-c",
                     "import sys; assert open(sys.argv[1], 'rb').read() == b'model bytes'"]
        signing_key = Ed25519PrivateKey.generate()
        private_b64 = base64.b64encode(signing_key.private_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption())).decode()
        public_b64 = base64.b64encode(signing_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)).decode()
        calls = []
        tamper_signature = False
        artifact_data = data

        def fake_request(url, _token, payload=None):
            calls.append((url, payload))
            if url.endswith("/manifest"):
                signed = sign_payload(release, private_b64)
                if tamper_signature:
                    signed["model_sha256"] = "0" * 64
                return FakeResponse(json.dumps(signed).encode())
            if "/devices/id/artifact/" in url:
                return FakeResponse(artifact_data)
            return FakeResponse(b"{}")

        with tempfile.TemporaryDirectory() as temporary, patch("scripts.device_updater.request", fake_request):
            state_dir = Path(temporary)
            self.assertEqual(sync_once("https://example.test", "id", "token", state_dir,
                                       self_test, public_b64, contract), "updated")
            pointer = json.loads((state_dir / "current.json").read_text())
            self.assertEqual(pointer["model_sha256"], digest)
            self.assertEqual(sync_once("https://example.test", "id", "token", state_dir,
                                       self_test, public_b64, contract), "up_to_date")
            self.assertEqual(sum(url.endswith("/devices/id/artifact/approved-1") for url, _ in calls), 1)
            release["manifest"]["class_names"] = ["wrong"]
            with self.assertRaisesRegex(ValueError, "pinned device contract"):
                sync_once("https://example.test", "id", "token", state_dir,
                          self_test, public_b64, contract)
            release["manifest"]["class_names"] = MODEL_MANIFEST["class_names"]
            with self.assertRaisesRegex(ValueError, "self-test"):
                sync_once("https://example.test", "id", "token", state_dir,
                          None, public_b64, contract)
            tamper_signature = True
            with self.assertRaises(ValueError):
                sync_once("https://example.test", "id", "token", state_dir,
                          self_test, public_b64, contract)
            tamper_signature = False
            release["generation"] = 0
            with self.assertRaisesRegex(ValueError, "Stale"):
                sync_once("https://example.test", "id", "token", state_dir,
                          self_test, public_b64, contract)
            release["generation"] = 2
            release["model_sha256"] = "f" * 64
            release["manifest"]["sha256"] = "f" * 64
            with self.assertRaises(ValueError):
                sync_once("https://example.test", "id", "token", state_dir,
                          self_test, public_b64, contract)
            self.assertEqual(json.loads((state_dir / "current.json").read_text()), pointer)
            release["run_id"] = "approved-2"
            release["artifact_path"] = "/devices/id/artifact/approved-2"
            release["model_sha256"] = hashlib.sha256(b"model v2").hexdigest()
            release["manifest"]["sha256"] = release["model_sha256"]
            artifact_data = b"model v2"
            with self.assertRaisesRegex(RuntimeError, "self-test failed"):
                sync_once("https://example.test", "id", "token", state_dir,
                          self_test, public_b64, contract)
            self.assertEqual(json.loads((state_dir / "current.json").read_text()), pointer)
            self.assertEqual(Path(pointer["model_path"]).read_bytes(), data)

    def test_device_telemetry_is_idempotent_and_requires_approved_run(self):
        import uuid
        device = DeviceRegistration(id=str(uuid.uuid4()), name="desk camera", channel="development",
            environment="development", token_sha256="a" * 64, is_active=True,
            created_by="admin", created_at=datetime.now(timezone.utc))
        with Session(self.engine) as db:
            db.add(device)
            db.add(MLOpsRun(run_id="approved-1", environment="development", status="approved",
                model_version="2.0.0", dataset_version="2.0.0",
                recorded_at=datetime.now(timezone.utc), object_prefix="development/runs/approved-1/",
                record_sha256="a" * 64, summary_json="{}"))
            db.add(ReleaseEvent(channel="development", environment="development", action="activate",
                from_run_id=None, to_run_id="approved-1", generation=1, actor="admin",
                reason="approved model", occurred_at=datetime.now(timezone.utc)))
            db.commit()
            event = {"event_id": str(uuid.uuid4()), "run_id": "approved-1",
                     "prediction": "INR_10", "confidence": 0.8,
                     "observed_at": datetime.now(timezone.utc).isoformat()}
            self.assertEqual(device_events({"events": [event]}, device, db)["accepted"], 1)
            self.assertEqual(device_events({"events": [event]}, device, db)["duplicates"], 1)
            self.assertEqual(db.query(DevicePredictionEvent).count(), 1)
            event["run_id"] = "unapproved"
            with self.assertRaises(HTTPException):
                device_events({"events": [event]}, device, db)
