import csv
import asyncio
import io
import json
import logging
import math
import hashlib
import re
import time
import uuid
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from botocore.exceptions import ClientError
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from PIL import Image
from sqlalchemy import Integer, and_, func, or_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .auth import create_access_token
from .config import CONFIDENCE_THRESHOLD, CORS_ORIGINS, ENVIRONMENT, INFERENCE_CONCURRENCY, STORAGE_BACKEND, UPLOAD_DIR, validate_production_config
from .database import SessionLocal, engine
from .dependencies import get_current_user, get_optional_current_user
from .inference import CLASS_NAMES, CURRENCY_VALUE, preprocess_image
from .models import ApiErrorEvent, DevicePredictionEvent, DeviceRegistration, MLOpsBenchmark, MLOpsMonitor, MLOpsRun, PredictionHistory, ReleaseChannel, User
from .mlops_registry import canonical_bytes, register_benchmark, register_monitor, register_run
from .monitoring import assess as assess_monitoring, jensen_shannon
from .schemas import PredictionResponse, PredictionReview, UserCreate, UserOut, UserUpdate
from .security import hash_password, verify_password
from .users import DEFAULT_ADMIN_PASSWORD, DEFAULT_ADMIN_USERNAME
from .storage import _s3_client, delete_image, image_url as stored_image_url, save_image, valid_image_signature
from .release_control import router as release_router
from .runtime_release import RuntimeRelease


# =====================================================
# CONSTANTS
# =====================================================
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 20_000_000
DEFAULT_PAGE_SIZE = 10
MAX_PAGE_SIZE = 100
ALLOWED_ROLES = {"admin", "analyst"}


# =====================================================
# LOGGER
# =====================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("currency_ai")


# =====================================================
# DATABASE SCHEMA IS MANAGED BY ALEMBIC
# =====================================================
validate_production_config()


# =====================================================
# INIT FASTAPI
# =====================================================
app = FastAPI(
    title="Currency AI Recognition System",
    version="2.0.0",
    description="AI-powered currency recognition API with versioned model releases",
)
app.include_router(release_router)


def record_api_error(request_id: str, path: str, status_code: int):
    try:
        with SessionLocal() as db:
            db.add(ApiErrorEvent(request_id=request_id, path=path[:300], status_code=status_code,
                                 occurred_at=datetime.utcnow()))
            db.commit()
    except Exception:
        logger.exception("failed_to_record_api_error request_id=%s", request_id)


@app.middleware("http")
async def request_logging(request: Request, call_next):
    request_id = str(uuid.uuid4())
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        await run_in_threadpool(record_api_error, request_id, request.url.path, 500)
        logger.exception("http_request_failed request_id=%s path=%s", request_id, request.url.path)
        raise
    if response.status_code >= 500:
        await run_in_threadpool(record_api_error, request_id, request.url.path, response.status_code)
    response.headers["X-Request-ID"] = request_id
    if request.url.path not in {"/health", "/ready"}:
        logger.info(
            "http_request request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
            request_id, request.method, request.url.path, response.status_code,
            (time.perf_counter() - started) * 1000,
        )
    return response


# =====================================================
# CORS (DEV MODE)
# =====================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================
# LOAD MODEL ONCE
# =====================================================
runtime_release = RuntimeRelease()
inference_slots = asyncio.Semaphore(INFERENCE_CONCURRENCY)


@app.get("/images/{image_key:path}")
def get_private_image(image_key: str, expires: int, signature: str):
    """Serve one stored prediction image through a short-lived signed API URL."""
    if STORAGE_BACKEND != "s3" or not valid_image_signature(image_key, expires, signature):
        raise HTTPException(status_code=403, detail="Invalid or expired image URL")
    from .config import S3_BUCKET

    try:
        stored = _s3_client().get_object(Bucket=S3_BUCKET, Key=image_key)
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") in {"404", "NoSuchKey", "NotFound"}:
            raise HTTPException(status_code=404, detail="Image not found") from error
        raise

    def chunks():
        body = stored["Body"]
        try:
            yield from body.iter_chunks(chunk_size=65536)
        finally:
            body.close()

    return StreamingResponse(
        chunks(), media_type=stored.get("ContentType") or "image/jpeg",
        headers={"Cache-Control": "private, max-age=60"},
    )


# =====================================================
# UPLOAD DIRECTORY + STATIC MOUNT
# =====================================================
if STORAGE_BACKEND == "local" and ENVIRONMENT == "development":
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


# =====================================================
# HELPERS
# =====================================================
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def normalize_username(username: str) -> str:
    return username.strip().lower()


def safe_csv_cell(value):
    text_value = str(value or "")
    return "'" + text_value if text_value.lstrip().startswith(("=", "+", "-", "@")) else text_value


def normalize_role(role: str) -> str:
    normalized = role.strip().lower()
    if normalized not in ALLOWED_ROLES:
        raise HTTPException(status_code=400, detail="Role must be admin or analyst")
    return normalized


def infer_currency_category(prediction: str) -> str:
    label = (prediction or "").upper()
    if label.startswith("VN"):
        return "vietnam"
    if label.startswith("USA"):
        return "usa"
    if label.startswith("INR"):
        return "india"

    if "_" in label:
        return label.split("_", 1)[0].lower()

    return "unknown"


def parse_top3(top3_json: Optional[str]):
    if not top3_json:
        return []

    try:
        parsed = json.loads(top3_json)
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


def serialize_history_record(record: PredictionHistory):
    image_url = stored_image_url(record.image_path) if record.image_path else None
    top3 = parse_top3(record.top3_json)

    return {
        "id": record.id,
        "filename": record.filename,
        "image_url": image_url,
        "prediction": record.prediction,
        "top3": top3,
        "confidence": record.confidence,
        "suspicious": record.suspicious,
        "estimated_value": record.estimated_value,
        "inference_time": record.inference_time,
        "queue_time": record.queue_time,
        "processing_time": record.processing_time,
        "model_version": record.model_version,
        "legacy_unverified": record.model_version != runtime_release.snapshot()[1]["version"],
        "actual_label": record.actual_label,
        "reviewed_by": record.reviewed_by,
        "reviewed_at": record.reviewed_at,
        "requested_by": record.requested_by,
        "currency_category": record.currency_category,
        "created_at": record.created_at,
    }


def serialize_user(user: User):
    role = normalize_role(user.role or ("admin" if user.is_admin else "analyst"))
    is_admin = role == "admin"

    return {
        "id": user.id,
        "username": user.username,
        "role": role,
        "is_admin": is_admin,
        "is_active": user.is_active,
        "created_at": user.created_at,
        "updated_at": user.updated_at,
    }


def parse_datetime_start(value: str) -> datetime:
    try:
        if len(value) == 10:
            return datetime.fromisoformat(value)
        return datetime.fromisoformat(value)
    except ValueError as parse_error:
        raise HTTPException(status_code=400, detail=f"Invalid date_from: {value}") from parse_error


def parse_datetime_end(value: str) -> datetime:
    try:
        if len(value) == 10:
            return datetime.fromisoformat(value) + timedelta(days=1)
        return datetime.fromisoformat(value)
    except ValueError as parse_error:
        raise HTTPException(status_code=400, detail=f"Invalid date_to: {value}") from parse_error


def apply_history_filters(
    query,
    search: Optional[str],
    suspicious: Optional[bool],
    currency: Optional[str],
    date_from: Optional[str],
    date_to: Optional[str],
):
    if search:
        keyword = search.strip().lower()
        query = query.filter(
            or_(
                func.lower(PredictionHistory.filename).contains(keyword),
                func.lower(PredictionHistory.prediction).contains(keyword),
            )
        )

    if suspicious is not None:
        query = query.filter(PredictionHistory.suspicious == suspicious)

    if currency:
        currency_keyword = currency.strip().lower()
        query = query.filter(
            or_(
                func.lower(PredictionHistory.currency_category) == currency_keyword,
                func.lower(PredictionHistory.prediction).contains(currency_keyword),
            )
        )

    if date_from:
        start_at = parse_datetime_start(date_from)
        query = query.filter(PredictionHistory.created_at >= start_at)

    if date_to:
        end_at = parse_datetime_end(date_to)
        query = query.filter(PredictionHistory.created_at < end_at)

    return query


def ensure_default_admin():
    db = SessionLocal()
    try:
        admin_username = normalize_username(DEFAULT_ADMIN_USERNAME)
        existing_admin = db.query(User).filter(User.username == admin_username).first()
        if existing_admin is None:
            db.add(
                User(
                    username=admin_username,
                    hashed_password=hash_password(DEFAULT_ADMIN_PASSWORD),
                    role="admin",
                    is_admin=True,
                    is_active=True,
                )
            )
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            if not db.query(User).filter(User.username == admin_username).first():
                raise
    finally:
        db.close()


def get_current_user_record(
    current_username: str = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> User:
    user = db.query(User).filter(User.username == normalize_username(current_username)).first()

    if user is None:
        raise HTTPException(status_code=401, detail="Invalid token")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="User account is inactive")

    return user


def get_current_admin(current_user: User = Depends(get_current_user_record)) -> User:
    role = normalize_role(current_user.role or ("admin" if current_user.is_admin else "analyst"))
    if role != "admin":
        raise HTTPException(status_code=403, detail="Admin privileges required")

    return current_user


@app.on_event("startup")
def startup_seed_admin():
    ensure_default_admin()
    if ENVIRONMENT in {"staging", "production"}:
        try:
            runtime_release.sync()
        except Exception as error:
            runtime_release.last_error = str(error)
            logger.exception("initial_approved_model_sync_failed")
            if ENVIRONMENT == "production":
                raise
        threading.Thread(target=runtime_release.poll_forever, daemon=True,
                         name="approved-model-poller").start()


# =====================================================
# ROOT + HEALTH
# =====================================================
@app.get("/")
def root():
    return {"message": "Currency AI API is running successfully"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/ready")
def ready():
    try:
        with engine.connect() as connection:
            connection.exec_driver_sql("SELECT 1")
        return {"status": "ready", "model_version": runtime_release.snapshot()[1]["version"],
                "active_run_id": runtime_release.run_id()}
    except Exception as error:
        logger.exception("readiness_failed")
        raise HTTPException(status_code=503, detail="Service not ready") from error


# =====================================================
# LOGIN
# =====================================================
@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    username = normalize_username(form_data.username)
    user = db.query(User).filter(User.username == username).first()

    if user is None:
        raise HTTPException(status_code=401, detail="Invalid username")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="User account is inactive")

    if not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid password")

    role = normalize_role(user.role or ("admin" if user.is_admin else "analyst"))
    access_token = create_access_token(data={"sub": user.username})

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "username": user.username,
        "role": role,
        "is_admin": role == "admin",
    }


@app.get("/me")
def get_me(current_user: User = Depends(get_current_user_record)):
    return {
        "username": current_user.username,
        "role": normalize_role(
            current_user.role or ("admin" if current_user.is_admin else "analyst")
        ),
        "is_admin": bool(current_user.is_admin),
        "is_active": bool(current_user.is_active),
    }


# =====================================================
# USER MANAGEMENT (ADMIN)
# =====================================================
@app.get("/users", response_model=list[UserOut])
def list_users(
    _current_admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    users = db.query(User).order_by(User.id.asc()).all()
    return [serialize_user(user) for user in users]


@app.post("/users", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def create_user(
    payload: UserCreate,
    _current_admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    username = normalize_username(payload.username)

    if db.query(User).filter(User.username == username).first():
        raise HTTPException(status_code=409, detail="Username already exists")

    role = normalize_role(payload.role) if payload.role else ("admin" if payload.is_admin else "analyst")

    user = User(
        username=username,
        hashed_password=hash_password(payload.password),
        role=role,
        is_admin=role == "admin",
        is_active=payload.is_active,
    )

    db.add(user)
    db.commit()
    db.refresh(user)
    return serialize_user(user)


@app.put("/users/{user_id}", response_model=UserOut)
def update_user(
    user_id: int,
    payload: UserUpdate,
    current_admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.id == user_id).first()

    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    if payload.username is not None:
        username = normalize_username(payload.username)
        existing_user = db.query(User).filter(User.username == username).first()
        if existing_user and existing_user.id != user.id:
            raise HTTPException(status_code=409, detail="Username already exists")
        user.username = username

    if payload.password is not None:
        if not payload.password.strip():
            raise HTTPException(status_code=400, detail="Password cannot be empty")
        user.hashed_password = hash_password(payload.password)

    current_role = normalize_role(user.role or ("admin" if user.is_admin else "analyst"))
    if payload.role is not None or payload.is_admin is not None:
        next_role = (
            normalize_role(payload.role)
            if payload.role is not None
            else ("admin" if payload.is_admin else "analyst")
        )

        if user.id == current_admin.id and next_role != "admin":
            raise HTTPException(status_code=400, detail="Cannot remove your own admin role")

        current_role = next_role
        user.role = current_role
        user.is_admin = current_role == "admin"

    if payload.is_active is not None:
        if user.id == current_admin.id and payload.is_active is False:
            raise HTTPException(status_code=400, detail="Cannot deactivate your own account")
        user.is_active = payload.is_active

    user.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(user)
    return serialize_user(user)


@app.delete("/users/{user_id}")
def delete_user(
    user_id: int,
    current_admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.id == user_id).first()

    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    if user.id == current_admin.id:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")

    db.delete(user)
    db.commit()

    return {"message": "User deleted"}


# =====================================================
# MODEL INFO (PROTECTED)
# =====================================================
@app.get("/model-info")
def model_info(_current_user: User = Depends(get_current_user_record)):
    manifest = runtime_release.snapshot()[1]
    return {
        "model_name": manifest["architecture"],
        "model_version": manifest["version"],
        "release_status": manifest.get("release_status"),
        "architecture": manifest["architecture"],
        "framework": manifest["framework"],
        "dataset_version": manifest["dataset_version"],
        "test_metrics": manifest.get("test_metrics"),
        "num_classes": len(CLASS_NAMES),
        "classes": CLASS_NAMES,
        "input_size": "x".join(str(value) for value in manifest["input_size"]),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
    }


@app.get("/mlops/runs")
def model_runs(model_version: Optional[str] = None, status_filter: Optional[str] = Query(None, alias="status"),
               limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0),
               _current_admin: User = Depends(get_current_admin), db: Session = Depends(get_db)):
    """Query run metadata; full reports remain in private object storage."""
    query = db.query(MLOpsRun).filter(MLOpsRun.environment == ENVIRONMENT)
    if model_version:
        query = query.filter(MLOpsRun.model_version == model_version)
    if status_filter:
        query = query.filter(MLOpsRun.status == status_filter)
    total = query.count()
    rows = query.order_by(MLOpsRun.recorded_at.desc()).offset(offset).limit(limit).all()
    return {"total": total, "runs": [json.loads(row.summary_json) for row in rows]}


@app.post("/mlops/runs/sync")
def sync_model_runs(_current_admin: User = Depends(get_current_admin), db: Session = Depends(get_db)):
    """Backfill the SQL registry from existing immutable S3 run objects."""
    if STORAGE_BACKEND != "s3":
        raise HTTPException(status_code=400, detail="Private object storage is not configured")
    from .config import S3_BUCKET

    prefix = f"{ENVIRONMENT}/runs/"
    registered = 0
    for page in _s3_client().get_paginator("list_objects_v2").paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            if not key.endswith("/run.json"):
                continue
            data = _s3_client().get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
            record = json.loads(data)
            run_id = record.get("run_id", "")
            if not re.fullmatch(r"[A-Za-z0-9-]{1,100}", run_id) or key != f"{prefix}{run_id}/run.json":
                raise HTTPException(status_code=409, detail="Archived run path mismatch")
            try:
                register_run(db, record, ENVIRONMENT, f"{prefix}{run_id}/", hashlib.sha256(data).hexdigest())
            except ValueError as error:
                raise HTTPException(status_code=409, detail=str(error)) from error
            registered += 1
    return {"registered": registered}


def _run_record(db: Session, run_id: str):
    row = db.get(MLOpsRun, run_id)
    if row is None or row.environment != ENVIRONMENT:
        raise HTTPException(status_code=404, detail="Run not found")
    return row


def _run_artifact(row: MLOpsRun, name: str):
    from .config import S3_BUCKET

    if name not in {"run", "manifest", "audit", "evaluation", "benchmark", "training_history", "training_config", "quality_gates", "release_gate"}:
        raise HTTPException(status_code=404, detail="Artifact not available")
    record_bytes = _s3_client().get_object(Bucket=S3_BUCKET, Key=row.object_prefix + "run.json")["Body"].read()
    if hashlib.sha256(record_bytes).hexdigest() != row.record_sha256:
        raise HTTPException(status_code=409, detail="Run record checksum mismatch")
    record = json.loads(record_bytes)
    if name == "run":
        return record_bytes, "run.json"
    item = record.get("files", {}).get(name)
    if not item or not re.fullmatch(r"[A-Za-z0-9_.-]+", item["file"]):
        raise HTTPException(status_code=404, detail="Artifact not available")
    data = _s3_client().get_object(Bucket=S3_BUCKET, Key=row.object_prefix + item["file"])["Body"].read()
    if hashlib.sha256(data).hexdigest() != item["sha256"]:
        raise HTTPException(status_code=409, detail="Artifact checksum mismatch")
    return data, item["file"]


@app.get("/mlops/runs/{run_id}")
def model_run_detail(run_id: str, _current_admin: User = Depends(get_current_admin), db: Session = Depends(get_db)):
    row = _run_record(db, run_id)
    details = {"summary": json.loads(row.summary_json), "artifacts": {}}
    for name in ("audit", "evaluation", "benchmark", "training_config", "quality_gates", "release_gate"):
        try:
            data, _ = _run_artifact(row, name)
            details["artifacts"][name] = json.loads(data)
        except HTTPException as error:
            if error.status_code != 404:
                raise
    return details


@app.get("/mlops/runs/{run_id}/artifacts/{name}")
def model_run_download(run_id: str, name: str, _current_admin: User = Depends(get_current_admin),
                       db: Session = Depends(get_db)):
    data, filename = _run_artifact(_run_record(db, run_id), name)
    media_type = "text/csv" if filename.endswith(".csv") else "application/json"
    return Response(content=data, media_type=media_type,
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/mlops/runs/{run_id}/metrics.csv")
def model_run_metrics_csv(run_id: str, _current_admin: User = Depends(get_current_admin),
                          db: Session = Depends(get_db)):
    data, _ = _run_artifact(_run_record(db, run_id), "evaluation")
    evaluation = json.loads(data)
    target = io.StringIO()
    writer = csv.writer(target)
    writer.writerow(["class", "precision", "recall", "f1", "support"])
    for label, values in sorted(evaluation.get("per_class", {}).items()):
        writer.writerow([label, values.get("precision"), values.get("recall"),
                         values.get("f1", values.get("f1-score")), values.get("support")])
    return Response(content=target.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{run_id}-metrics.csv"'})


@app.get("/mlops/benchmarks")
def list_mlops_benchmarks(model_version: Optional[str] = None, limit: int = Query(50, ge=1, le=100),
                          _current_admin: User = Depends(get_current_admin), db: Session = Depends(get_db)):
    query = db.query(MLOpsBenchmark).filter(MLOpsBenchmark.environment == ENVIRONMENT)
    if model_version:
        query = query.filter(MLOpsBenchmark.model_version == model_version)
    rows = query.order_by(MLOpsBenchmark.recorded_at.desc()).limit(limit).all()
    return {"benchmarks": [{"benchmark_id": row.benchmark_id, "recorded_at": row.recorded_at.isoformat(),
        "model_version": row.model_version, **json.loads(row.summary_json)} for row in rows]}


@app.post("/mlops/benchmarks")
def save_mlops_benchmark(payload: dict, _current_admin: User = Depends(get_current_admin),
                         db: Session = Depends(get_db)):
    from .config import S3_BUCKET

    if STORAGE_BACKEND != "s3":
        raise HTTPException(status_code=400, detail="Private object storage is required")

    benchmark_id = payload.get("benchmark_id", "")
    report = payload.get("report")
    if not re.fullmatch(r"[A-Za-z0-9-]{1,100}", benchmark_id) or not isinstance(report, dict):
        raise HTTPException(status_code=422, detail="Invalid benchmark ID or report")
    if report.get("runs", 0) < 100 or not report.get("model_versions") or report.get("success_rate") is None:
        raise HTTPException(status_code=422, detail="Benchmark requires at least 100 measured requests and model metadata")
    content = canonical_bytes(report)
    if len(content) > 1024 * 1024:
        raise HTTPException(status_code=413, detail="Benchmark report too large")
    key = f"{ENVIRONMENT}/benchmarks/{benchmark_id}.json"
    digest = hashlib.sha256(content).hexdigest()
    try:
        existing = _s3_client().get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
        if hashlib.sha256(existing).hexdigest() != digest:
            raise HTTPException(status_code=409, detail="Benchmark ID already exists with different report")
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") not in {"404", "NoSuchKey", "NotFound"}:
            raise
        _s3_client().put_object(Bucket=S3_BUCKET, Key=key, Body=content, ContentType="application/json",
                                Metadata={"sha256": digest})
    try:
        register_benchmark(db, benchmark_id, report, ENVIRONMENT, key, digest)
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return {"benchmark_id": benchmark_id, "sha256": digest}


@app.get("/mlops/benchmarks/{benchmark_id}")
def get_mlops_benchmark(benchmark_id: str, _current_admin: User = Depends(get_current_admin),
                        db: Session = Depends(get_db)):
    from .config import S3_BUCKET

    row = db.get(MLOpsBenchmark, benchmark_id)
    if not row or row.environment != ENVIRONMENT:
        raise HTTPException(status_code=404, detail="Benchmark not found")
    data = _s3_client().get_object(Bucket=S3_BUCKET, Key=row.object_key)["Body"].read()
    if hashlib.sha256(data).hexdigest() != row.report_sha256:
        raise HTTPException(status_code=409, detail="Benchmark checksum mismatch")
    return json.loads(data)


@app.delete("/mlops/benchmark-traffic/{run_id}")
def purge_mlops_benchmark_traffic(run_id: str, _current_admin: User = Depends(get_current_admin),
                                  db: Session = Depends(get_db)):
    """Remove tagged staging test traffic after its report has been archived."""
    if ENVIRONMENT != "staging":
        raise HTTPException(status_code=403, detail="Benchmark cleanup is staging only")
    try:
        canonical_id = str(uuid.UUID(run_id))
    except ValueError as error:
        raise HTTPException(status_code=422, detail="Invalid benchmark run ID") from error
    archived = (db.query(MLOpsBenchmark).filter(MLOpsBenchmark.environment == ENVIRONMENT).all())
    if not any(json.loads(item.summary_json).get("benchmark_run_id") == canonical_id for item in archived):
        raise HTTPException(status_code=409, detail="Archive the benchmark report before deleting traffic")
    rows = (db.query(PredictionHistory)
            .filter(PredictionHistory.requested_by == f"benchmark:{canonical_id}").all())
    for row in rows:
        if row.image_path:
            if not re.fullmatch(r"predictions/[0-9a-f-]{36}\.jpg", row.image_path):
                raise HTTPException(status_code=409, detail="Unexpected benchmark image key")
            delete_image(row.image_path)
        db.delete(row)
    db.commit()
    return {"run_id": canonical_id, "deleted_records": len(rows)}


@app.get("/mlops/monitoring")
def list_mlops_monitoring(limit: int = Query(30, ge=1, le=100),
                           _current_admin: User = Depends(get_current_admin), db: Session = Depends(get_db)):
    rows = (db.query(MLOpsMonitor).filter(MLOpsMonitor.environment == ENVIRONMENT)
            .order_by(MLOpsMonitor.checked_at.desc()).limit(limit).all())
    return {"snapshots": [json.loads(row.snapshot_json) for row in rows]}


@app.get("/mlops/monitoring/current")
def current_mlops_monitoring(_current_admin: User = Depends(get_current_admin),
                              db: Session = Depends(get_db)):
    """Calculate operational signals; model drift needs an approved baseline and enough samples."""
    from .config import S3_BUCKET

    targets_path = Path(__file__).resolve().parents[1] / "docs/free-staging-monitor-targets.json"
    targets = json.loads(targets_path.read_text(encoding="utf-8"))
    serving_manifest = runtime_release.snapshot()[1]
    serving_version = serving_manifest["version"]
    operational = stats(_current_user=_current_admin, db=db)
    rows = (db.query(PredictionHistory.prediction, PredictionHistory.confidence)
            .filter(PredictionHistory.model_version == serving_version,
                    PredictionHistory.requested_by.notlike("benchmark:%"))
            .order_by(PredictionHistory.id.desc()).limit(1000).all())
    predictions = [{"prediction": row.prediction, "confidence": row.confidence,
                    "threshold": CONFIDENCE_THRESHOLD} for row in rows]
    errors_24h = (db.query(func.count(ApiErrorEvent.id))
                  .filter(ApiErrorEvent.occurred_at >= datetime.utcnow() - timedelta(days=1)).scalar() or 0)
    storage_bytes = None
    if STORAGE_BACKEND == "s3":
        try:
            storage_bytes = sum(item["Size"] for page in _s3_client().get_paginator("list_objects_v2")
                                .paginate(Bucket=S3_BUCKET) for item in page.get("Contents", []))
        except Exception:
            logger.exception("storage_usage_probe_failed")
    baseline = None
    if serving_manifest.get("release_status") == "approved":
        approved = (db.query(MLOpsRun).filter(MLOpsRun.environment == ENVIRONMENT,
                    MLOpsRun.model_version == serving_version, MLOpsRun.status == "approved")
                    .order_by(MLOpsRun.recorded_at.desc()).first())
        if approved:
            audit_bytes, _ = _run_artifact(approved, "audit")
            evaluation_bytes, _ = _run_artifact(approved, "evaluation")
            audit = json.loads(audit_bytes)
            evaluation = json.loads(evaluation_bytes)
            if audit.get("dataset_manifest_verified") and audit.get("dataset_version") == evaluation.get("dataset_version"):
                baseline = {"class_counts": audit["counts"]["test"],
                            "accepted_coverage": evaluation["accepted_coverage"]}
    assessed = assess_monitoring(operational, predictions, baseline, errors_24h, storage_bytes, targets)
    release_channel = db.get(ReleaseChannel, ENVIRONMENT)
    devices = db.query(DeviceRegistration).filter(DeviceRegistration.environment == ENVIRONMENT,
        DeviceRegistration.is_active.is_(True)).all()
    active_run = release_channel.active_run_id if release_channel else None
    offline_before = datetime.utcnow() - timedelta(minutes=targets.get("device_offline_minutes", 20))
    offline_devices = sum(device.last_seen_at is None or device.last_seen_at.replace(tzinfo=None) < offline_before
                          for device in devices)
    outdated_devices = sum(device.reported_run_id != active_run for device in devices) if active_run else 0
    device_quality = {"status": "insufficient_data", "sample_size": 0,
                      "low_confidence_rate": None, "distribution_js_divergence": None,
                      "reviewed_count": 0, "reviewed_accuracy": None}
    if active_run:
        device_rows = (db.query(DevicePredictionEvent)
            .filter(DevicePredictionEvent.environment == ENVIRONMENT,
                    DevicePredictionEvent.run_id == active_run)
            .order_by(DevicePredictionEvent.received_at.desc()).limit(1000).all())
        device_quality["sample_size"] = len(device_rows)
        reviewed = [row for row in device_rows if row.actual_label is not None]
        device_quality["reviewed_count"] = len(reviewed)
        if len(reviewed) >= targets["minimum_reviewed_samples"]:
            device_quality["reviewed_accuracy"] = round(
                sum(row.prediction == row.actual_label for row in reviewed) / len(reviewed), 6)
        if device_rows:
            device_quality["low_confidence_rate"] = round(
                sum(row.confidence < CONFIDENCE_THRESHOLD for row in device_rows) / len(device_rows), 6)
        approved_run = db.get(MLOpsRun, active_run)
        if approved_run and len(device_rows) >= targets["minimum_model_samples"]:
            audit_bytes, _ = _run_artifact(approved_run, "audit")
            evaluation_bytes, _ = _run_artifact(approved_run, "evaluation")
            audit = json.loads(audit_bytes)
            evaluation = json.loads(evaluation_bytes)
            if audit.get("dataset_manifest_verified") and audit.get("human_review_complete"):
                from collections import Counter
                distribution = Counter(row.prediction for row in device_rows)
                divergence = jensen_shannon(distribution, audit["counts"]["test"])
                device_quality["distribution_js_divergence"] = divergence
                baseline_low = 1 - evaluation["accepted_coverage"]
                degraded = (device_quality["low_confidence_rate"] > baseline_low + targets["max_low_confidence_increase"]
                            or divergence is not None and divergence > targets["warning_distribution_js"])
                device_quality["status"] = "warning" if degraded else "healthy"
                if degraded:
                    assessed["reasons"].append("Device prediction distribution requires investigation")
                    if assessed["status"] != "critical":
                        assessed["status"] = "warning"
    if active_run and (offline_devices or outdated_devices):
        assessed["reasons"].append("Device fleet has offline or outdated installations")
        if assessed["status"] != "critical":
            assessed["status"] = "warning"
    if active_run and runtime_release.run_id() != active_run:
        assessed["reasons"].append("API serving model has not synchronized with release channel")
        if assessed["status"] != "critical":
            assessed["status"] = "warning"
    return {"checked_at": datetime.now(timezone.utc).isoformat(), "environment": ENVIRONMENT,
            "model_version": serving_version, "release_status": serving_manifest.get("release_status"),
            "serving_run_id": runtime_release.run_id(),
            "serving_update_error": runtime_release.last_error,
            "channel_active_run_id": active_run,
            "device_total": len(devices), "device_offline": offline_devices,
            "device_outdated": outdated_devices, "device_quality": device_quality,
            "thresholds": {key: value for key, value in targets.items() if key != "note"}, **assessed}


@app.post("/mlops/monitoring")
def save_mlops_monitoring(payload: dict, _current_admin: User = Depends(get_current_admin),
                           db: Session = Depends(get_db)):
    if payload.get("status") not in {"healthy", "warning", "critical", "insufficient_data"} or not payload.get("checked_at"):
        raise HTTPException(status_code=422, detail="Invalid monitoring snapshot")
    try:
        item = register_monitor(db, payload, ENVIRONMENT)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {"snapshot_id": item.snapshot_id}

# =====================================================
# PREDICT (PUBLIC)
# =====================================================
@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: Request,
    file: UploadFile = File(...),
    current_username: Optional[str] = Depends(get_optional_current_user),
    db: Session = Depends(get_db),
):
    request_started = time.perf_counter()
    if ENVIRONMENT == "staging":
        staging_user = db.query(User).filter(User.username == normalize_username(current_username or "")).first()
        if staging_user is None or not staging_user.is_active:
            raise HTTPException(status_code=401, detail="Staging predictions require an active login")
    requested_by = normalize_username(current_username) if current_username else "guest"
    benchmark_run = request.headers.get("X-Benchmark-Run")
    if benchmark_run and ENVIRONMENT == "staging":
        try:
            requested_by = f"benchmark:{uuid.UUID(benchmark_run)}"
        except ValueError as parse_error:
            raise HTTPException(status_code=400, detail="Invalid benchmark run ID") from parse_error
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=400, detail="Only JPEG, PNG and WebP images are supported")

    try:
        contents = await file.read(MAX_UPLOAD_SIZE_BYTES + 1)
    except Exception as read_error:
        raise HTTPException(status_code=400, detail="Unable to read upload file") from read_error

    if len(contents) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="File size exceeds 10MB limit")

    try:
        opened_image = Image.open(io.BytesIO(contents))
        if opened_image.width * opened_image.height > Image.MAX_IMAGE_PIXELS:
            raise ValueError("Image dimensions exceed limit")
        image = opened_image.convert("RGB")
    except Exception as image_error:
        raise HTTPException(status_code=400, detail="Invalid image file") from image_error

    processed = await run_in_threadpool(preprocess_image, image)

    serving_model, serving_manifest = runtime_release.snapshot()
    serving_version = serving_manifest["version"]
    queued_at = time.perf_counter()
    async with inference_slots:
        queue_time = round(time.perf_counter() - queued_at, 4)
        start_time = time.perf_counter()
        preds = (await run_in_threadpool(serving_model.predict, processed, verbose=0))[0]
        inference_time = round(time.perf_counter() - start_time, 4)

    top3_idx = preds.argsort()[-3:][::-1]
    best_idx = int(top3_idx[0])

    if len(preds) != len(CLASS_NAMES):
        raise HTTPException(status_code=500, detail="Model class index mismatch")

    predicted_class = CLASS_NAMES[best_idx]
    confidence = float(preds[best_idx])
    is_background_prediction = "background" in predicted_class.lower()
    suspicious = (confidence < CONFIDENCE_THRESHOLD) or is_background_prediction
    top3 = [
        {
            "class_name": CLASS_NAMES[int(index)],
            "probability": round(float(preds[index]), 4),
        }
        for index in top3_idx
    ]
    currency_category = infer_currency_category(predicted_class)

    result = {
        "prediction": predicted_class,
        "confidence": round(confidence, 4),
        "suspicious": suspicious,
        "estimated_value": CURRENCY_VALUE.get(predicted_class, 0),
        "inference_time": inference_time,
        "queue_time": queue_time,
        "processing_time": 0.0,
        "model_version": serving_version,
        "top3": top3,
    }

    image_filename = f"predictions/{uuid.uuid4()}.jpg"
    image_buffer = io.BytesIO()
    await run_in_threadpool(image.save, image_buffer, "JPEG", quality=88)
    save_image(image_filename, image_buffer.getvalue())

    record = PredictionHistory(
        filename=(file.filename or "upload").replace("\\", "/").rsplit("/", 1)[-1][:255],
        image_path=image_filename,
        prediction=predicted_class,
        top3_json=json.dumps(top3),
        confidence=round(confidence, 4),
        suspicious=suspicious,
        estimated_value=CURRENCY_VALUE.get(predicted_class, 0),
        inference_time=inference_time,
        queue_time=queue_time,
        model_version=serving_version,
        requested_by=requested_by,
        currency_category=currency_category,
        created_at=datetime.utcnow(),
    )

    try:
        db.add(record)
        db.flush()
        record.processing_time = round(time.perf_counter() - request_started, 4)
        db.commit()
    except Exception:
        db.rollback()
        try:
            delete_image(image_filename)
        except Exception:
            logger.exception("orphaned_image_cleanup_failed key=%s", image_filename)
        raise

    result["processing_time"] = record.processing_time

    logger.info(
        "prediction_event user=%s prediction=%s confidence=%.4f review_required=%s model=%s queue_s=%.4f inference_s=%.4f processing_s=%.4f",
        requested_by,
        predicted_class,
        confidence,
        suspicious,
        serving_version,
        queue_time,
        inference_time,
        record.processing_time,
    )

    return result


# =====================================================
# HISTORY EXPORT (PROTECTED)
# =====================================================
@app.get("/history/export")
def export_history(
    search: Optional[str] = Query(default=None),
    suspicious: Optional[bool] = Query(default=None),
    currency: Optional[str] = Query(default=None),
    date_from: Optional[str] = Query(default=None),
    date_to: Optional[str] = Query(default=None),
    _current_user: User = Depends(get_current_user_record),
    db: Session = Depends(get_db),
):
    query = db.query(PredictionHistory).filter(PredictionHistory.requested_by.notlike("benchmark:%"))
    query = apply_history_filters(query, search, suspicious, currency, date_from, date_to)
    records = query.order_by(PredictionHistory.created_at.desc()).limit(10000).all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "id",
            "filename",
            "prediction",
            "confidence",
            "suspicious",
            "estimated_value",
            "inference_time",
            "requested_by",
            "currency_category",
            "created_at",
        ]
    )

    for record in records:
        writer.writerow(
            [
                record.id,
                safe_csv_cell(record.filename),
                record.prediction,
                record.confidence,
                record.suspicious,
                record.estimated_value,
                record.inference_time,
                safe_csv_cell(record.requested_by),
                record.currency_category,
                record.created_at.isoformat() if record.created_at else "",
            ]
        )

    output.seek(0)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=prediction-history.csv"},
    )


# =====================================================
# HISTORY (PROTECTED)
# =====================================================
@app.get("/history")
def history(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    search: Optional[str] = Query(default=None),
    suspicious: Optional[bool] = Query(default=None),
    currency: Optional[str] = Query(default=None),
    date_from: Optional[str] = Query(default=None),
    date_to: Optional[str] = Query(default=None),
    _current_user: User = Depends(get_current_user_record),
    db: Session = Depends(get_db),
):
    query = db.query(PredictionHistory).filter(PredictionHistory.requested_by.notlike("benchmark:%"))
    query = apply_history_filters(query, search, suspicious, currency, date_from, date_to)

    total = query.count()
    total_pages = max(1, math.ceil(total / page_size)) if total else 1
    offset = (page - 1) * page_size

    records = (
        query.order_by(PredictionHistory.created_at.desc())
        .offset(offset)
        .limit(page_size)
        .all()
    )

    return {
        "items": [serialize_history_record(record) for record in records],
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
    }


@app.get("/history/{record_id}")
def history_detail(
    record_id: int,
    _current_user: User = Depends(get_current_user_record),
    db: Session = Depends(get_db),
):
    record = db.query(PredictionHistory).filter(PredictionHistory.id == record_id).first()

    if record is None:
        raise HTTPException(status_code=404, detail="History record not found")

    return serialize_history_record(record)


@app.post("/history/{record_id}/review")
def review_prediction(
    record_id: int,
    payload: PredictionReview,
    current_user: User = Depends(get_current_user_record),
    db: Session = Depends(get_db),
):
    label = payload.actual_label.strip()
    if label not in CLASS_NAMES and label != "unknown":
        raise HTTPException(status_code=400, detail="Actual label is not in model classes")
    record = db.query(PredictionHistory).filter(PredictionHistory.id == record_id).first()
    if record is None:
        raise HTTPException(status_code=404, detail="History record not found")
    record.actual_label = label
    record.reviewed_by = current_user.username
    record.reviewed_at = datetime.utcnow()
    db.commit()
    return serialize_history_record(record)


@app.delete("/history/{record_id}")
def delete_history_record(
    record_id: int,
    _current_admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    record = db.query(PredictionHistory).filter(PredictionHistory.id == record_id).first()
    if record is None:
        raise HTTPException(status_code=404, detail="History record not found")
    if record.image_path:
        delete_image(record.image_path)
    db.delete(record)
    db.commit()
    return {"message": "History record deleted"}


# =====================================================
# STATS (PROTECTED)
# =====================================================
@app.get("/stats")
def stats(
    _current_user: User = Depends(get_current_user_record),
    db: Session = Depends(get_db),
):
    operational_records = PredictionHistory.requested_by.notlike("benchmark:%")
    serving_version = runtime_release.snapshot()[1]["version"]
    current_records = and_(PredictionHistory.model_version == serving_version, operational_records)
    total_predictions = db.query(func.count(PredictionHistory.id)).filter(current_records).scalar() or 0
    legacy_predictions = (db.query(func.count(PredictionHistory.id)).filter(operational_records).scalar() or 0) - total_predictions
    suspicious_count = (
        db.query(func.count(PredictionHistory.id))
        .filter(current_records, PredictionHistory.suspicious.is_(True)).scalar() or 0
    )
    currency_distribution = dict(
        db.query(PredictionHistory.prediction, func.count(PredictionHistory.id))
        .filter(current_records)
        .group_by(PredictionHistory.prediction).all()
    )
    country_distribution = dict(
        db.query(PredictionHistory.currency_category, func.count(PredictionHistory.id))
        .filter(current_records)
        .group_by(PredictionHistory.currency_category).all()
    )
    daily_rows = (
        db.query(func.date(PredictionHistory.created_at), func.count(PredictionHistory.id))
        .filter(current_records, PredictionHistory.created_at >= datetime.utcnow() - timedelta(days=90))
        .group_by(func.date(PredictionHistory.created_at))
        .order_by(func.date(PredictionHistory.created_at)).all()
    )
    suspicious_daily = dict(
        db.query(func.date(PredictionHistory.created_at), func.count(PredictionHistory.id))
        .filter(current_records, PredictionHistory.created_at >= datetime.utcnow() - timedelta(days=90))
        .filter(PredictionHistory.suspicious.is_(True))
        .group_by(func.date(PredictionHistory.created_at)).all()
    )
    recent = (
        db.query(PredictionHistory.inference_time, PredictionHistory.queue_time, PredictionHistory.processing_time)
        .filter(current_records)
        .order_by(PredictionHistory.id.desc()).limit(1000).all()
    )

    def summarize_latency(values):
        ordered = sorted(round(value * 1000, 2) for value in values if value is not None)
        if not ordered:
            return {"mean_ms": None, "p50_ms": None, "p95_ms": None, "p99_ms": None}
        return {
            "mean_ms": round(sum(ordered) / len(ordered), 2),
            "p50_ms": ordered[math.ceil(0.50 * len(ordered)) - 1],
            "p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1],
            "p99_ms": ordered[math.ceil(0.99 * len(ordered)) - 1],
        }

    reviewed_count, reviewed_correct = db.query(
        func.count(PredictionHistory.id),
        func.sum(func.cast(PredictionHistory.actual_label == PredictionHistory.prediction, Integer)),
    ).filter(
        PredictionHistory.actual_label.is_not(None),
        current_records,
    ).one()
    reviewed_count = reviewed_count or 0
    reviewed_correct = reviewed_correct or 0

    return {
        "total_predictions": total_predictions,
        "legacy_predictions": legacy_predictions,
        "suspicious_count": suspicious_count,
        "detected_currencies_count": len(currency_distribution),
        "currency_distribution": currency_distribution,
        "country_distribution": country_distribution,
        "predictions_per_day": [{"date": str(day), "count": count} for day, count in daily_rows],
        "suspicious_trend": [{"date": str(day), "count": suspicious_daily.get(day, 0)} for day, _ in daily_rows],
        "model_version": serving_version,
        "latency_sample_size": len(recent),
        "inference_latency_sample_size": sum(row[1] is not None for row in recent),
        "inference_latency": summarize_latency([row[0] for row in recent if row[1] is not None]),
        "queue_latency": summarize_latency([row[1] for row in recent]),
        "processing_latency": summarize_latency([row[2] for row in recent]),
        "reviewed_count": reviewed_count,
        "reviewed_accuracy": round(reviewed_correct / reviewed_count, 4) if reviewed_count else None,
    }
