import tensorflow as tf
from pathlib import Path
from .config import MODEL_PATH

_model = None

def load_model():
    global _model

    if _model is None:
        model_path = Path(MODEL_PATH)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at: {model_path}"
            )

        print(f"🔄 Loading model from {model_path} ...")
        _model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully")

    return _model