"""Load approved channel releases without a code redeploy.

The bootstrap model remains available while a new artifact is downloaded and
validated. Each prediction uses one immutable model/manifest snapshot.
"""

import hashlib
import logging
import os
import threading
import time
from pathlib import Path

from .config import ENVIRONMENT, S3_BUCKET
from .database import SessionLocal
from .inference import CLASS_NAMES, MODEL_MANIFEST
from .model_loader import OnnxModel, load_model
from .models import ReleaseChannel
from .release_control import contract_matches, verified_release
from .storage import _s3_client

logger = logging.getLogger(__name__)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RuntimeRelease:
    def __init__(self):
        self._lock = threading.RLock()
        self._model = load_model()
        self._manifest = MODEL_MANIFEST
        self._run_id = None
        self.last_error = None

    def snapshot(self):
        with self._lock:
            return self._model, self._manifest

    def run_id(self):
        with self._lock:
            return self._run_id

    def sync(self):
        if ENVIRONMENT not in {"staging", "production"}:
            return False
        with SessionLocal() as db:
            channel = db.get(ReleaseChannel, ENVIRONMENT)
            if not channel or not channel.active_run_id or channel.active_run_id == self.run_id():
                return False
            release = verified_release(db, channel.active_run_id, verify_model=False)
            manifest = release["manifest"]
            if not contract_matches(manifest):
                raise RuntimeError("Approved release changes the API inference contract")
            record = release["record"]
            item = record["files"]["model_artifact"]
            suffix = Path(item["file"]).suffix
            if suffix not in {".onnx", ".keras"}:
                raise RuntimeError("Unsupported approved model format")
            path = Path("/tmp/currency-ai-approved") / (record["run_id"] + suffix)
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists() or sha256(path) != release["model_sha256"]:
                temporary = path.with_suffix(path.suffix + ".partial")
                try:
                    _s3_client().download_file(S3_BUCKET, release["row"].object_prefix + item["file"], str(temporary))
                    if sha256(temporary) != release["model_sha256"]:
                        raise RuntimeError("Downloaded approved model checksum mismatch")
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
            if suffix == ".onnx":
                loaded = OnnxModel(path)
            else:
                import tensorflow as tf
                loaded = tf.keras.models.load_model(path, compile=False)
            if loaded.output_shape[-1] != len(CLASS_NAMES):
                raise RuntimeError("Approved model output count mismatch")
            # Do not replace a newer channel switch that occurred while loading.
            db.refresh(channel)
            if channel.active_run_id != record["run_id"]:
                return False
            with self._lock:
                self._model = loaded
                self._manifest = manifest
                self._run_id = record["run_id"]
                self.last_error = None
            logger.info("approved_model_loaded run=%s version=%s sha256=%s",
                        record["run_id"], manifest["version"], release["model_sha256"])
            return True

    def poll_forever(self, interval=60):
        while True:
            try:
                self.sync()
            except Exception as error:
                self.last_error = str(error)
                logger.exception("approved_model_sync_failed")
            time.sleep(interval)
