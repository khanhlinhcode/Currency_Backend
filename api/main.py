import csv
import io
import json
import logging
import math
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from PIL import Image
from sqlalchemy import func, inspect, or_, text
from sqlalchemy.orm import Session

from .auth import create_access_token
from .config import CONFIDENCE_THRESHOLD
from .database import Base, SessionLocal, engine
from .dependencies import get_current_user, get_optional_current_user
from .inference import CLASS_NAMES, CURRENCY_VALUE, preprocess_image
from .model_loader import load_model
from .models import PredictionHistory, User
from .schemas import PredictionResponse, UserCreate, UserOut, UserUpdate
from .security import hash_password, verify_password
from .users import DEFAULT_ADMIN_PASSWORD, DEFAULT_ADMIN_USERNAME


# =====================================================
# CONSTANTS
# =====================================================
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024
DEFAULT_PAGE_SIZE = 10
MAX_PAGE_SIZE = 100
ALLOWED_ROLES = {"admin", "analyst"}


# =====================================================
# LOGGER
# =====================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("currency_ai")


# =====================================================
# CREATE DATABASE TABLES + LIGHT MIGRATION
# =====================================================
Base.metadata.create_all(bind=engine)


def migrate_schema():
    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())

    with engine.begin() as connection:
        if "prediction_history" in table_names:
            columns = {col["name"] for col in inspector.get_columns("prediction_history")}
            if "image_path" not in columns:
                connection.execute(
                    text("ALTER TABLE prediction_history ADD COLUMN image_path VARCHAR")
                )
            if "top3_json" not in columns:
                connection.execute(
                    text("ALTER TABLE prediction_history ADD COLUMN top3_json VARCHAR")
                )
            if "requested_by" not in columns:
                connection.execute(
                    text(
                        "ALTER TABLE prediction_history ADD COLUMN requested_by VARCHAR DEFAULT 'guest'"
                    )
                )
            if "currency_category" not in columns:
                connection.execute(
                    text(
                        "ALTER TABLE prediction_history ADD COLUMN currency_category VARCHAR DEFAULT 'unknown'"
                    )
                )

        if "users" in table_names:
            columns = {col["name"] for col in inspector.get_columns("users")}
            if "role" not in columns:
                connection.execute(
                    text("ALTER TABLE users ADD COLUMN role VARCHAR DEFAULT 'analyst'")
                )
            connection.execute(
                text(
                    "UPDATE users "
                    "SET role = CASE WHEN is_admin = 1 THEN 'admin' ELSE 'analyst' END "
                    "WHERE role IS NULL OR role = ''"
                )
            )
            connection.execute(
                text("UPDATE users SET is_admin = CASE WHEN role = 'admin' THEN 1 ELSE 0 END")
            )


migrate_schema()


# =====================================================
# INIT FASTAPI
# =====================================================
app = FastAPI(
    title="Currency AI Recognition System",
    version="1.2.0",
    description="AI-powered currency recognition API using EfficientNetV2B0",
)


# =====================================================
# CORS (DEV MODE)
# =====================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================
# LOAD MODEL ONCE
# =====================================================
model = load_model()


# =====================================================
# UPLOAD DIRECTORY + STATIC MOUNT
# =====================================================
UPLOAD_DIR = Path("api/uploads")
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
    image_url = f"/uploads/{record.image_path}" if record.image_path else None
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
        else:
            # Dev convenience: keep default admin always usable for demo/login recovery.
            existing_admin.hashed_password = hash_password(DEFAULT_ADMIN_PASSWORD)
            existing_admin.role = "admin"
            existing_admin.is_admin = True
            existing_admin.is_active = True

        db.commit()
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


# =====================================================
# ROOT + HEALTH
# =====================================================
@app.get("/")
def root():
    return {"message": "Currency AI API is running successfully"}


@app.get("/health")
def health():
    return {"status": "healthy"}


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
    return {
        "model_name": "EfficientNetV2B0 Transfer Learning",
        "architecture": "EfficientNetV2B0 backbone + Dense classifier head",
        "framework": "TensorFlow / Keras",
        "dataset_size": 18000,
        "num_classes": len(CLASS_NAMES),
        "classes": CLASS_NAMES,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "training_accuracy": "97.4%",
        "last_training_date": "2026-02-10",
    }


# =====================================================
# PREDICT (PUBLIC)
# =====================================================
@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    current_username: Optional[str] = Depends(get_optional_current_user),
    db: Session = Depends(get_db),
):
    if not file.content_type or not file.content_type.startswith("image"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
    except Exception as read_error:
        raise HTTPException(status_code=400, detail="Unable to read upload file") from read_error

    if len(contents) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="File size exceeds 10MB limit")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as image_error:
        raise HTTPException(status_code=400, detail="Invalid image file") from image_error

    file_id = str(uuid.uuid4())
    image_filename = f"{file_id}.jpg"
    image_path = UPLOAD_DIR / image_filename
    with open(image_path, "wb") as image_file:
        image_file.write(contents)

    processed = preprocess_image(image)

    start_time = time.time()
    preds = model.predict(processed, verbose=0)[0]
    inference_time = round(time.time() - start_time, 4)

    top3_idx = preds.argsort()[-3:][::-1]
    best_idx = int(top3_idx[0])

    if best_idx >= len(CLASS_NAMES):
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
    requested_by = normalize_username(current_username) if current_username else "guest"

    result = {
        "prediction": predicted_class,
        "confidence": round(confidence, 4),
        "suspicious": suspicious,
        "estimated_value": CURRENCY_VALUE.get(predicted_class, 0),
        "inference_time": inference_time,
        "top3": top3,
    }

    record = PredictionHistory(
        filename=file.filename,
        image_path=image_filename,
        prediction=predicted_class,
        top3_json=json.dumps(top3),
        confidence=round(confidence, 4),
        suspicious=suspicious,
        estimated_value=CURRENCY_VALUE.get(predicted_class, 0),
        inference_time=inference_time,
        requested_by=requested_by,
        currency_category=currency_category,
        created_at=datetime.utcnow(),
    )

    db.add(record)
    db.commit()

    logger.info(
        "prediction_event user=%s filename=%s prediction=%s confidence=%.4f suspicious=%s",
        requested_by,
        file.filename,
        predicted_class,
        confidence,
        suspicious,
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
    query = db.query(PredictionHistory)
    query = apply_history_filters(query, search, suspicious, currency, date_from, date_to)
    records = query.order_by(PredictionHistory.created_at.desc()).all()

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
                record.filename,
                record.prediction,
                record.confidence,
                record.suspicious,
                record.estimated_value,
                record.inference_time,
                record.requested_by,
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
    query = db.query(PredictionHistory)
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


# =====================================================
# STATS (PROTECTED)
# =====================================================
@app.get("/stats")
def stats(
    _current_user: User = Depends(get_current_user_record),
    db: Session = Depends(get_db),
):
    records = db.query(PredictionHistory).order_by(PredictionHistory.created_at.asc()).all()

    if not records:
        return {
            "total_predictions": 0,
            "total_estimated_value": 0,
            "suspicious_count": 0,
            "detected_currencies_count": 0,
            "currency_distribution": {},
            "country_distribution": {},
            "predictions_per_day": [],
            "suspicious_trend": [],
        }

    total_predictions = len(records)
    total_estimated_value = sum(record.estimated_value for record in records)
    suspicious_count = sum(1 for record in records if record.suspicious)

    currency_distribution = {}
    country_distribution = {}
    predictions_per_day = {}
    suspicious_per_day = {}

    for record in records:
        currency_distribution[record.prediction] = (
            currency_distribution.get(record.prediction, 0) + 1
        )

        category = infer_currency_category(record.prediction)
        if category == "vietnam":
            country = "Vietnam"
        elif category == "usa":
            country = "USA"
        elif category == "india":
            country = "India"
        else:
            country = category.capitalize()

        country_distribution[country] = country_distribution.get(country, 0) + 1

        day_key = record.created_at.date().isoformat() if record.created_at else "unknown"
        predictions_per_day[day_key] = predictions_per_day.get(day_key, 0) + 1

        if record.suspicious:
            suspicious_per_day[day_key] = suspicious_per_day.get(day_key, 0) + 1

    prediction_series = [
        {"date": date_key, "count": predictions_per_day[date_key]}
        for date_key in sorted(predictions_per_day.keys())
    ]

    suspicious_series = [
        {"date": date_key, "count": suspicious_per_day.get(date_key, 0)}
        for date_key in sorted(predictions_per_day.keys())
    ]

    return {
        "total_predictions": total_predictions,
        "total_estimated_value": total_estimated_value,
        "suspicious_count": suspicious_count,
        "detected_currencies_count": len(currency_distribution),
        "currency_distribution": currency_distribution,
        "country_distribution": country_distribution,
        "predictions_per_day": prediction_series,
        "suspicious_trend": suspicious_series,
    }
