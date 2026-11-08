"""Fetch a pinned artifact and validate its output contract before serving."""

import hashlib
import logging
import shutil
import urllib.request

from .config import ENVIRONMENT, MODEL_PATH, MODEL_S3_BUCKET, MODEL_S3_KEY, MODEL_URL
from .inference import CLASS_NAMES, MODEL_MANIFEST

logger = logging.getLogger(__name__)
_model = None


class OnnxModel:
    def __init__(self, path):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError("ONNX model must have one input and one output")
        shape = inputs[0].shape
        if shape[1:] != [224, 224, 3] or outputs[0].shape[-1] != len(CLASS_NAMES):
            raise RuntimeError("ONNX model dimensions do not match manifest")
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        self.output_shape = outputs[0].shape

    def predict(self, tensor, verbose=0):
        return self.session.run([self.output_name], {self.input_name: tensor})[0]


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_artifact():
    expected = MODEL_MANIFEST["sha256"]
    if MODEL_PATH.exists() and _sha256(MODEL_PATH) == expected:
        return
    if not MODEL_URL and not MODEL_S3_KEY:
        raise RuntimeError("Pinned model artifact missing or invalid; set MODEL_S3_KEY or MODEL_URL")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = MODEL_PATH.with_suffix(".download")
    try:
        if MODEL_S3_KEY:
            from .storage import _s3_client
            _s3_client().download_file(MODEL_S3_BUCKET, MODEL_S3_KEY, str(temporary))
        else:
            with urllib.request.urlopen(MODEL_URL, timeout=120) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output)
        if _sha256(temporary) != expected:
            raise RuntimeError("Downloaded model checksum mismatch")
        temporary.replace(MODEL_PATH)
    finally:
        temporary.unlink(missing_ok=True)

def load_model():
    global _model

    if _model is None:
        if ENVIRONMENT == "production" and (
            MODEL_MANIFEST.get("release_status") != "approved"
            or MODEL_MANIFEST.get("test_metrics") is None
            or MODEL_MANIFEST.get("dataset_version") == "legacy-unversioned"
        ):
            raise RuntimeError("Production requires an approved model with verified dataset and test metrics")
        _ensure_artifact()
        if MODEL_PATH.suffix == ".onnx":
            loaded = OnnxModel(MODEL_PATH)
        elif MODEL_PATH.suffix == ".keras":
            import tensorflow as tf
            loaded = tf.keras.models.load_model(MODEL_PATH, compile=False)
        else:
            raise RuntimeError("Unsupported model artifact format")
        if loaded.output_shape[-1] != len(CLASS_NAMES):
            raise RuntimeError("Model output size does not match manifest class_names")
        _model = loaded
        logger.info("model_loaded version=%s sha256=%s", MODEL_MANIFEST["version"], MODEL_MANIFEST["sha256"])

    return _model
