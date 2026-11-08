"""Runtime configuration shared by the API and model loader."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
ENVIRONMENT = os.getenv("APP_ENV", "development").lower()
if ENVIRONMENT not in {"development", "staging", "production"}:
    raise RuntimeError("APP_ENV must be development, staging or production")
SECRET_KEY = os.getenv("SECRET_KEY", "development-only-secret-change-before-production")
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(BASE_DIR / "models/efficientnetb0_best.keras")))
MODEL_MANIFEST_PATH = Path(os.getenv("MODEL_MANIFEST_PATH", str(BASE_DIR / "models/model-manifest.json")))
MODEL_URL = os.getenv("MODEL_URL", "")
MODEL_S3_KEY = os.getenv("MODEL_S3_KEY", "")
MODEL_S3_BUCKET = os.getenv("MODEL_S3_BUCKET", os.getenv("S3_BUCKET", ""))
RAW_DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'currency_ai.db'}")
DATABASE_URL = RAW_DATABASE_URL.replace("postgres://", "postgresql+psycopg://", 1).replace("postgresql://", "postgresql+psycopg://", 1)
CORS_ORIGINS = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if origin.strip()]
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))
INFERENCE_CONCURRENCY = max(1, int(os.getenv("INFERENCE_CONCURRENCY", "1")))
STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "local").lower()
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", str(BASE_DIR / "api/uploads")))
S3_BUCKET = os.getenv("S3_BUCKET", "")
S3_REGION = os.getenv("S3_REGION", "auto")
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "")
S3_ACCESS_KEY_ID = os.getenv("S3_ACCESS_KEY_ID", "")
S3_SECRET_ACCESS_KEY = os.getenv("S3_SECRET_ACCESS_KEY", "")


def validate_production_config():
    if ENVIRONMENT not in {"staging", "production"}:
        return
    required = {
        "DATABASE_URL": DATABASE_URL.startswith("postgresql"),
        "CORS_ORIGINS": bool(os.getenv("CORS_ORIGINS")) and bool(CORS_ORIGINS) and "*" not in CORS_ORIGINS,
        "SECRET_KEY": len(os.getenv("SECRET_KEY", "")) >= 32,
        "ADMIN_USERNAME": bool(os.getenv("ADMIN_USERNAME")),
        "ADMIN_PASSWORD": len(os.getenv("ADMIN_PASSWORD", "")) >= 12,
        "STORAGE_BACKEND=s3": STORAGE_BACKEND == "s3",
        "S3_BUCKET": bool(S3_BUCKET),
        "S3_ACCESS_KEY_ID": bool(S3_ACCESS_KEY_ID),
        "S3_SECRET_ACCESS_KEY": bool(S3_SECRET_ACCESS_KEY),
        "MODEL_S3_KEY or MODEL_URL": bool(MODEL_S3_KEY or MODEL_URL),
    }
    missing = [name for name, valid in required.items() if not valid]
    if missing:
        raise RuntimeError("Invalid deployed configuration: " + ", ".join(missing))
