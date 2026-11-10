"""Approval-gated release channels and pull-based device updates."""

import hashlib
import hmac
import io
import importlib.util
import json
import math
import re
import secrets
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, File, Header, HTTPException, Response, UploadFile
from PIL import Image
from botocore.exceptions import ClientError
from sqlalchemy import update
from sqlalchemy.orm import Session

from .config import ENVIRONMENT, S3_BUCKET, STORAGE_BACKEND
from .database import SessionLocal
from .dependencies import get_current_user
from .inference import CLASS_NAMES, MODEL_MANIFEST
from .models import DevicePredictionEvent, DeviceRegistration, MLOpsRun, ReleaseChannel, ReleaseEvent, User
from .storage import _s3_client

router = APIRouter()
CHANNELS = {"staging", "production"}


def contract_matches(manifest):
    return (manifest.get("class_names") == CLASS_NAMES
        and manifest.get("input_size") == MODEL_MANIFEST["input_size"]
        and manifest.get("preprocessing") == MODEL_MANIFEST["preprocessing"]
        and manifest.get("resize_interpolation") == MODEL_MANIFEST["resize_interpolation"]
        and manifest.get("confidence_threshold") == MODEL_MANIFEST["confidence_threshold"])


def get_db():
    with SessionLocal() as db:
        yield db


def admin_user(username: str = Depends(get_current_user), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username.strip().lower()).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="Invalid account")
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin privileges required")
    return user


def channel_name(name):
    if name not in CHANNELS or name != ENVIRONMENT:
        raise HTTPException(status_code=404, detail="Channel unavailable in this environment")
    return name


def channel_state(db, name):
    row = db.get(ReleaseChannel, name)
    if row is None:
        return {"name": name, "environment": ENVIRONMENT, "active_run_id": None,
                "previous_run_id": None, "generation": 0, "updated_by": None, "updated_at": None}
    if row.environment != ENVIRONMENT:
        raise HTTPException(status_code=409, detail="Channel environment mismatch")
    return {"name": name, "environment": row.environment, "active_run_id": row.active_run_id,
            "previous_run_id": row.previous_run_id, "generation": row.generation,
            "updated_by": row.updated_by, "updated_at": row.updated_at}


def _archived_file(row, record, name):
    item = record.get("files", {}).get(name)
    if not item or not re.fullmatch(r"[A-Za-z0-9_.-]+", item.get("file", "")):
        raise HTTPException(status_code=409, detail=f"Missing archived {name}")
    data = _s3_client().get_object(Bucket=S3_BUCKET, Key=row.object_prefix + item["file"])["Body"].read()
    if hashlib.sha256(data).hexdigest() != item.get("sha256"):
        raise HTTPException(status_code=409, detail=f"Archived {name} checksum mismatch")
    return data


def verified_release(db, run_id, verify_model=True):
    """Verify the stored evidence and exact model bytes before channel activation."""
    row = db.get(MLOpsRun, run_id)
    if not row or row.environment != ENVIRONMENT or row.status != "approved":
        raise HTTPException(status_code=409, detail="Only approved runs from this environment can be activated")
    record_bytes = _s3_client().get_object(Bucket=S3_BUCKET, Key=row.object_prefix + "run.json")["Body"].read()
    if hashlib.sha256(record_bytes).hexdigest() != row.record_sha256:
        raise HTTPException(status_code=409, detail="Run record checksum mismatch")
    record = json.loads(record_bytes)
    manifest = json.loads(_archived_file(row, record, "manifest"))
    audit = json.loads(_archived_file(row, record, "audit"))
    gate = json.loads(_archived_file(row, record, "release_gate"))
    _archived_file(row, record, "evaluation")
    _archived_file(row, record, "quality_gates")
    model_item = record.get("files", {}).get("model_artifact") or {}
    model_hash = model_item.get("sha256")
    model_suffix = str(model_item.get("file", "")).rsplit(".", 1)[-1]
    if model_suffix not in {"onnx", "keras"} or (
        model_suffix == "keras" and importlib.util.find_spec("tensorflow") is None
    ):
        raise HTTPException(status_code=409, detail="Model format is unavailable on this serving environment")
    if not re.fullmatch(r"[0-9a-f]{64}", model_hash or ""):
        raise HTTPException(status_code=409, detail="Missing model artifact hash")
    if verify_model and hashlib.sha256(_archived_file(row, record, "model_artifact")).hexdigest() != model_hash:
        raise HTTPException(status_code=409, detail="Model artifact checksum mismatch")
    if (record.get("status") != "approved" or record.get("release_gate_passed") is not True
        or record.get("run_id") != run_id or record.get("model_sha256") != model_hash
        or record.get("dataset_version") != audit.get("dataset_version")
        or record.get("dataset_manifest_sha256") != audit.get("dataset_manifest_sha256")
        or not audit.get("dataset_manifest_verified") or not audit.get("human_review_complete")
        or not audit.get("group_split_verified") or audit.get("exact_cross_split_duplicates")
        or audit.get("conflicting_label_groups") or not gate.get("passed")
        or gate.get("gates_sha256") != record.get("quality_gates_sha256")
        or manifest.get("release_status") != "approved"
        or manifest.get("dataset_version") in {None, "legacy-unversioned"}
        or manifest.get("version") != record.get("model_version")
        or manifest.get("sha256") != model_hash or not contract_matches(manifest)):
        raise HTTPException(status_code=409, detail="Run evidence fails release contract")
    return {"row": row, "record": record, "manifest": manifest, "model_sha256": model_hash}


def _switch(db, name, run_id, expected_generation, actor, reason, action):
    if not isinstance(expected_generation, int) or expected_generation < 0:
        raise HTTPException(status_code=422, detail="expected_generation is required")
    if not isinstance(reason, str) or not 5 <= len(reason.strip()) <= 500:
        raise HTTPException(status_code=422, detail="A reason of 5–500 characters is required")
    if STORAGE_BACKEND != "s3":
        raise HTTPException(status_code=409, detail="Private storage required")
    verified_release(db, run_id)
    current = channel_state(db, name)
    if current["generation"] != expected_generation:
        raise HTTPException(status_code=409, detail="Channel changed; refresh before retrying")
    if current["active_run_id"] == run_id:
        return current
    now = datetime.now(timezone.utc)
    if db.get(ReleaseChannel, name) is None:
        row = ReleaseChannel(name=name, environment=ENVIRONMENT, active_run_id=run_id,
                             previous_run_id=None, generation=1, updated_by=actor, updated_at=now)
        db.add(row)
    else:
        changed = db.execute(update(ReleaseChannel).where(ReleaseChannel.name == name,
            ReleaseChannel.environment == ENVIRONMENT,
            ReleaseChannel.generation == expected_generation).values(
            active_run_id=run_id, previous_run_id=current["active_run_id"],
            generation=expected_generation + 1, updated_by=actor, updated_at=now))
        if changed.rowcount != 1:
            db.rollback()
            raise HTTPException(status_code=409, detail="Concurrent release change")
    db.add(ReleaseEvent(channel=name, environment=ENVIRONMENT, action=action,
                        from_run_id=current["active_run_id"], to_run_id=run_id,
                        generation=expected_generation + 1, actor=actor,
                        reason=reason.strip(), occurred_at=now))
    try:
        db.commit()
    except Exception:
        db.rollback()
        raise
    return channel_state(db, name)


@router.get("/mlops/channels")
def list_channels(_admin: User = Depends(admin_user), db: Session = Depends(get_db)):
    name = channel_name(ENVIRONMENT)
    events = (db.query(ReleaseEvent).filter(ReleaseEvent.channel == name,
              ReleaseEvent.environment == ENVIRONMENT).order_by(ReleaseEvent.id.desc()).limit(30).all())
    return {"channels": [channel_state(db, name)], "events": [
        {"action": item.action, "from_run_id": item.from_run_id,
         "to_run_id": item.to_run_id, "generation": item.generation,
         "actor": item.actor, "reason": item.reason, "occurred_at": item.occurred_at} for item in events]}


@router.post("/mlops/channels/{name}/activate")
def activate(name: str, payload: dict, admin: User = Depends(admin_user), db: Session = Depends(get_db)):
    channel_name(name)
    run_id = payload.get("run_id", "")
    if not re.fullmatch(r"[A-Za-z0-9-]{1,100}", run_id):
        raise HTTPException(status_code=422, detail="Invalid run_id")
    return _switch(db, name, run_id, payload.get("expected_generation"), admin.username,
                   payload.get("reason"), "activate")


@router.post("/mlops/channels/{name}/rollback")
def rollback(name: str, payload: dict, admin: User = Depends(admin_user), db: Session = Depends(get_db)):
    channel_name(name)
    current = channel_state(db, name)
    if not current["previous_run_id"]:
        raise HTTPException(status_code=409, detail="No previous approved release")
    if payload.get("expected_generation") != current["generation"]:
        raise HTTPException(status_code=409, detail="Channel changed; refresh before retrying")
    return _switch(db, name, current["previous_run_id"], current["generation"],
                   admin.username, payload.get("reason"), "rollback")


@router.get("/mlops/devices")
def list_devices(_admin: User = Depends(admin_user), db: Session = Depends(get_db)):
    rows = (db.query(DeviceRegistration).filter(DeviceRegistration.environment == ENVIRONMENT)
            .order_by(DeviceRegistration.created_at.desc()).limit(200).all())
    return {"devices": [{"id": row.id, "name": row.name, "channel": row.channel,
              "is_active": row.is_active, "created_at": row.created_at,
              "last_seen_at": row.last_seen_at, "reported_run_id": row.reported_run_id,
              "reported_generation": row.reported_generation,
              "reported_status": row.reported_status} for row in rows]}


@router.post("/mlops/devices")
def create_device(payload: dict, admin: User = Depends(admin_user), db: Session = Depends(get_db)):
    name = str(payload.get("name", "")).strip()
    channel = channel_name(payload.get("channel"))
    if not 3 <= len(name) <= 100:
        raise HTTPException(status_code=422, detail="Device name must be 3–100 characters")
    raw_token = secrets.token_urlsafe(48)
    device = DeviceRegistration(id=str(uuid.uuid4()), name=name, channel=channel,
        environment=ENVIRONMENT, token_sha256=hashlib.sha256(raw_token.encode()).hexdigest(),
        is_active=True, created_by=admin.username, created_at=datetime.now(timezone.utc))
    db.add(device)
    db.commit()
    return {"id": device.id, "name": device.name, "channel": channel,
            "token": raw_token, "note": "Token is shown once; store it on the device securely"}


@router.post("/mlops/devices/{device_id}/revoke")
def revoke_device(device_id: str, _admin: User = Depends(admin_user), db: Session = Depends(get_db)):
    row = db.get(DeviceRegistration, device_id)
    if not row or row.environment != ENVIRONMENT:
        raise HTTPException(status_code=404, detail="Device not found")
    row.is_active = False
    db.commit()
    return {"id": row.id, "is_active": False}


def device_auth(device_id: str, x_device_token: str = Header(default=""), db: Session = Depends(get_db)):
    row = db.get(DeviceRegistration, device_id)
    token_sha256 = hashlib.sha256(x_device_token.encode()).hexdigest()
    if (not row or not row.is_active or row.environment != ENVIRONMENT
        or not x_device_token or not hmac.compare_digest(row.token_sha256, token_sha256)):
        raise HTTPException(status_code=401, detail="Invalid device credential")
    return row


@router.get("/devices/{device_id}/manifest")
def device_manifest(device: DeviceRegistration = Depends(device_auth), db: Session = Depends(get_db)):
    channel = channel_state(db, device.channel)
    device.last_seen_at = datetime.now(timezone.utc)
    db.commit()
    if not channel["active_run_id"]:
        return {"status": "no_release", "channel": device.channel,
                "generation": channel["generation"]}
    release = verified_release(db, channel["active_run_id"], verify_model=False)
    record = release["record"]
    model_file = record["files"]["model_artifact"]["file"]
    return {"status": "available", "channel": device.channel,
        "generation": channel["generation"], "run_id": record["run_id"],
        "model_version": record["model_version"], "dataset_version": record["dataset_version"],
        "model_sha256": release["model_sha256"], "model_filename": model_file,
        "manifest": release["manifest"],
        "artifact_path": f"/devices/{device.id}/artifact/{record['run_id']}"}


@router.get("/devices/{device_id}/artifact/{run_id}")
def device_artifact(run_id: str, device: DeviceRegistration = Depends(device_auth),
                    db: Session = Depends(get_db)):
    channel = channel_state(db, device.channel)
    if run_id != channel["active_run_id"]:
        raise HTTPException(status_code=404, detail="Release not active")
    release = verified_release(db, run_id)
    data = _archived_file(release["row"], release["record"], "model_artifact")
    return Response(content=data, media_type="application/octet-stream",
        headers={"Cache-Control": "private, no-store", "X-Model-SHA256": release["model_sha256"]})


@router.post("/devices/{device_id}/heartbeat")
def device_heartbeat(payload: dict, device: DeviceRegistration = Depends(device_auth),
                     db: Session = Depends(get_db)):
    status = payload.get("status")
    if status not in {"ready", "updating", "failed", "rolled_back"}:
        raise HTTPException(status_code=422, detail="Invalid device status")
    run_id = payload.get("run_id")
    if run_id is not None and not re.fullmatch(r"[A-Za-z0-9-]{1,100}", run_id):
        raise HTTPException(status_code=422, detail="Invalid run ID")
    generation = payload.get("generation")
    if generation is not None and (not isinstance(generation, int) or generation < 0):
        raise HTTPException(status_code=422, detail="Invalid generation")
    device.reported_run_id = run_id
    device.reported_generation = generation
    device.reported_status = status
    device.last_seen_at = datetime.now(timezone.utc)
    db.commit()
    return {"ok": True}


@router.post("/devices/{device_id}/events")
def device_events(payload: dict, device: DeviceRegistration = Depends(device_auth),
                  db: Session = Depends(get_db)):
    """Idempotent device telemetry. Labels can only be attached by an admin review."""
    events = payload.get("events")
    if not isinstance(events, list) or not 1 <= len(events) <= 100:
        raise HTTPException(status_code=422, detail="Supply 1–100 events")
    accepted = 0
    duplicates = 0
    seen_ids = set()
    now = datetime.now(timezone.utc)
    for event in events:
        if not isinstance(event, dict):
            raise HTTPException(status_code=422, detail="Invalid event")
        try:
            event_id = str(uuid.UUID(event.get("event_id", "")))
            observed_at = datetime.fromisoformat(event.get("observed_at", "").replace("Z", "+00:00"))
        except (ValueError, TypeError) as error:
            raise HTTPException(status_code=422, detail="Invalid event ID or timestamp") from error
        if observed_at.tzinfo is None or not now - timedelta(days=7) <= observed_at <= now + timedelta(minutes=5):
            raise HTTPException(status_code=422, detail="Event timestamp outside allowed window")
        run_id = event.get("run_id", "")
        if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9-]{1,100}", run_id):
            raise HTTPException(status_code=422, detail="Invalid run ID")
        run = db.get(MLOpsRun, run_id)
        if not run or run.environment != ENVIRONMENT or run.status != "approved":
            raise HTTPException(status_code=422, detail="Telemetry requires an approved run")
        prediction = event.get("prediction", "")
        confidence = event.get("confidence")
        inference_ms = event.get("inference_ms")
        if (not isinstance(prediction, str) or not 1 <= len(prediction) <= 100
            or not isinstance(confidence, (float, int)) or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
            or (inference_ms is not None and (not isinstance(inference_ms, (float, int))
                or not math.isfinite(inference_ms) or not 0 <= inference_ms <= 120000))):
            raise HTTPException(status_code=422, detail="Invalid prediction telemetry")
        existing = db.get(DevicePredictionEvent, event_id)
        if event_id in seen_ids:
            raise HTTPException(status_code=422, detail="Repeated event ID in batch")
        seen_ids.add(event_id)
        if existing:
            stored_at = existing.observed_at
            stored_utc = (stored_at.replace(tzinfo=timezone.utc) if stored_at.tzinfo is None
                          else stored_at.astimezone(timezone.utc))
            if (existing.device_id != device.id or existing.run_id != run_id
                or existing.prediction != prediction or existing.confidence != confidence
                or existing.inference_ms != inference_ms
                or stored_utc != observed_at.astimezone(timezone.utc)):
                raise HTTPException(status_code=409, detail="Event ID reused with different data")
            duplicates += 1
            continue
        db.add(DevicePredictionEvent(event_id=event_id, device_id=device.id,
            environment=ENVIRONMENT, run_id=run_id, prediction=prediction,
            confidence=float(confidence), inference_ms=inference_ms,
            observed_at=observed_at, received_at=now))
        accepted += 1
    db.commit()
    return {"accepted": accepted, "duplicates": duplicates}


@router.get("/mlops/device-events")
def list_device_events(run_id: str = "", reviewed: str = "", limit: int = 100,
                       _admin: User = Depends(admin_user), db: Session = Depends(get_db)):
    if not 1 <= limit <= 200 or reviewed not in {"", "yes", "no"}:
        raise HTTPException(status_code=422, detail="Invalid filter")
    query = db.query(DevicePredictionEvent).filter(DevicePredictionEvent.environment == ENVIRONMENT)
    if run_id:
        query = query.filter(DevicePredictionEvent.run_id == run_id)
    if reviewed:
        query = query.filter(DevicePredictionEvent.reviewed_at.isnot(None) if reviewed == "yes"
                             else DevicePredictionEvent.reviewed_at.is_(None))
    rows = query.order_by(DevicePredictionEvent.received_at.desc()).limit(limit).all()
    return {"events": [{"event_id": row.event_id, "device_id": row.device_id,
        "run_id": row.run_id, "prediction": row.prediction, "confidence": row.confidence,
        "inference_ms": row.inference_ms, "observed_at": row.observed_at,
        "has_image": bool(row.image_key),
        "actual_label": row.actual_label, "reviewed_by": row.reviewed_by,
        "reviewed_at": row.reviewed_at} for row in rows]}


@router.post("/devices/{device_id}/events/{event_id}/image")
async def device_event_image_upload(event_id: str, file: UploadFile = File(...),
                                    device: DeviceRegistration = Depends(device_auth),
                                    db: Session = Depends(get_db)):
    row = db.get(DevicePredictionEvent, event_id)
    if not row or row.device_id != device.id or row.environment != ENVIRONMENT:
        raise HTTPException(status_code=404, detail="Device event not found")
    data = await file.read(5 * 1024 * 1024 + 1)
    if len(data) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Evidence image exceeds 5 MB")
    try:
        image = Image.open(io.BytesIO(data))
        if image.format != "JPEG" or image.width * image.height > 20_000_000:
            raise ValueError("Unsupported evidence image")
        image.verify()
    except Exception as error:
        raise HTTPException(status_code=422, detail="A valid JPEG image is required") from error
    digest = hashlib.sha256(data).hexdigest()
    if row.image_key:
        if row.image_sha256 != digest:
            raise HTTPException(status_code=409, detail="Evidence image is immutable")
        return {"event_id": event_id, "sha256": digest, "already_present": True}
    key = f"{ENVIRONMENT}/device-events/{event_id}.jpg"
    try:
        stored = _s3_client().head_object(Bucket=S3_BUCKET, Key=key)
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") not in {"404", "NoSuchKey", "NotFound"}:
            raise
        _s3_client().put_object(Bucket=S3_BUCKET, Key=key, Body=data,
            ContentType="image/jpeg", Metadata={"sha256": digest})
    else:
        if stored.get("Metadata", {}).get("sha256") != digest:
            raise HTTPException(status_code=409, detail="Evidence object already exists with different content")
    row.image_key = key
    row.image_sha256 = digest
    db.commit()
    return {"event_id": event_id, "sha256": digest, "already_present": False}


@router.get("/mlops/device-events/{event_id}/image")
def device_event_image_download(event_id: str, _admin: User = Depends(admin_user),
                                db: Session = Depends(get_db)):
    row = db.get(DevicePredictionEvent, event_id)
    if not row or row.environment != ENVIRONMENT or not row.image_key:
        raise HTTPException(status_code=404, detail="Evidence image not found")
    data = _s3_client().get_object(Bucket=S3_BUCKET, Key=row.image_key)["Body"].read()
    if hashlib.sha256(data).hexdigest() != row.image_sha256:
        raise HTTPException(status_code=409, detail="Evidence image checksum mismatch")
    return Response(content=data, media_type="image/jpeg",
                    headers={"Cache-Control": "private, no-store"})


@router.post("/mlops/device-events/{event_id}/review")
def review_device_event(event_id: str, payload: dict, admin: User = Depends(admin_user),
                        db: Session = Depends(get_db)):
    row = db.get(DevicePredictionEvent, event_id)
    if not row or row.environment != ENVIRONMENT:
        raise HTTPException(status_code=404, detail="Device event not found")
    if not row.image_key:
        raise HTTPException(status_code=409, detail="Attach private image evidence before review")
    run = db.get(MLOpsRun, row.run_id)
    record_bytes = _s3_client().get_object(Bucket=S3_BUCKET, Key=run.object_prefix + "run.json")["Body"].read()
    if hashlib.sha256(record_bytes).hexdigest() != run.record_sha256:
        raise HTTPException(status_code=409, detail="Run record checksum mismatch")
    manifest = json.loads(_archived_file(run, json.loads(record_bytes), "manifest"))
    actual = payload.get("actual_label")
    if actual not in manifest.get("class_names", []):
        raise HTTPException(status_code=422, detail="Label is not in the model's class list")
    if row.reviewed_at:
        if row.actual_label != actual:
            raise HTTPException(status_code=409, detail="Reviewed label is immutable")
        return {"event_id": event_id, "actual_label": actual, "reviewed_by": row.reviewed_by}
    row.actual_label = actual
    row.reviewed_by = admin.username
    row.reviewed_at = datetime.now(timezone.utc)
    db.commit()
    return {"event_id": event_id, "actual_label": actual, "reviewed_by": admin.username}
