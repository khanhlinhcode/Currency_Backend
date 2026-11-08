import json

import numpy as np
from PIL import Image

from .config import CONFIDENCE_THRESHOLD, IMG_SIZE, MODEL_MANIFEST_PATH


def load_manifest():
    with MODEL_MANIFEST_PATH.open(encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    labels = manifest.get("class_names")
    if not isinstance(labels, list) or not labels or len(labels) != len(set(labels)):
        raise RuntimeError("Model manifest must contain unique class_names")
    if manifest.get("input_size") != list(IMG_SIZE):
        raise RuntimeError("Model manifest input_size does not match API preprocessing")
    if manifest.get("preprocessing") != "tensorflow.keras.applications.efficientnet.preprocess_input":
        raise RuntimeError("Model manifest preprocessing does not match API preprocessing")
    if manifest.get("resize_interpolation") != "nearest":
        raise RuntimeError("Model manifest resize interpolation must match training")
    if manifest.get("confidence_threshold") != CONFIDENCE_THRESHOLD:
        raise RuntimeError("Model manifest confidence_threshold does not match API configuration")
    return manifest


MODEL_MANIFEST = load_manifest()
CLASS_NAMES = MODEL_MANIFEST["class_names"]
MODEL_VERSION = MODEL_MANIFEST["version"]

CURRENCY_VALUE = {
    # ================= INR =================
    "INR_10": 10,
    "INR_20": 20,
    "INR_50": 50,
    "INR_100": 100,
    "INR_200": 200,
    "INR_500": 500,
    "INR_2000": 2000,

    # ================= USA =================
    "USA_1 Dollar": 1,
    "USA_2 Dollar": 2,
    "USA_5 Dollar": 5,
    "USA_10 Dollar": 10,
    "USA_50 Dollar": 50,
    "USA_100 Dollar": 100,

    # ================= VN =================
    "VN_000000": 0,
    "VN_000200": 200,
    "VN_000500": 500,
    "VN_001000": 1000,
    "VN_002000": 2000,
    "VN_005000": 5000,
    "VN_010000": 10000,
    "VN_020000": 20000,
    "VN_050000": 50000,
    "VN_100000": 100000,
    "VN_200000": 200000,
    "VN_500000": 500000,
}
if set(CURRENCY_VALUE) != set(CLASS_NAMES):
    raise RuntimeError("Currency values and model class names differ")
def preprocess_image(image: Image.Image):
    image = image.convert("RGB").resize(IMG_SIZE, Image.Resampling.NEAREST)
    arr = np.asarray(image, dtype=np.float32)
    # EfficientNet's Keras preprocess_input is an identity function: input
    # rescaling is built into the exported network. Keep raw 0..255 float32.
    arr = np.expand_dims(arr, axis=0)
    return arr
