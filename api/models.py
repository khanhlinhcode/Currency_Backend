from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text

from .database import Base


class PredictionHistory(Base):
    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    image_path = Column(String, nullable=True)
    prediction = Column(String)
    top3_json = Column(String, nullable=True)
    confidence = Column(Float)
    suspicious = Column(Boolean)
    estimated_value = Column(Integer)
    inference_time = Column(Float)
    queue_time = Column(Float, nullable=True)
    processing_time = Column(Float, nullable=True)
    model_version = Column(String(50), nullable=True)
    actual_label = Column(String(100), nullable=True)
    reviewed_by = Column(String(50), nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    requested_by = Column(String, default="guest", nullable=False)
    currency_category = Column(String, default="unknown", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(20), default="analyst", nullable=False)
    is_admin = Column(Boolean, default=False, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )


class MLOpsRun(Base):
    __tablename__ = "mlops_runs"

    run_id = Column(String(100), primary_key=True)
    environment = Column(String(20), nullable=False, index=True)
    status = Column(String(20), nullable=False)
    model_version = Column(String(100), nullable=False, index=True)
    dataset_version = Column(String(100), nullable=False, index=True)
    recorded_at = Column(DateTime(timezone=True), nullable=False)
    object_prefix = Column(String(300), nullable=False)
    record_sha256 = Column(String(64), nullable=False)
    summary_json = Column(Text, nullable=False)


class MLOpsBenchmark(Base):
    __tablename__ = "mlops_benchmarks"

    benchmark_id = Column(String(100), primary_key=True)
    environment = Column(String(20), nullable=False, index=True)
    model_version = Column(String(100), nullable=False, index=True)
    recorded_at = Column(DateTime(timezone=True), nullable=False)
    object_key = Column(String(300), nullable=False)
    report_sha256 = Column(String(64), nullable=False)
    summary_json = Column(Text, nullable=False)


class MLOpsMonitor(Base):
    __tablename__ = "mlops_monitor_snapshots"

    snapshot_id = Column(String(100), primary_key=True)
    environment = Column(String(20), nullable=False, index=True)
    checked_at = Column(DateTime(timezone=True), nullable=False, index=True)
    status = Column(String(30), nullable=False)
    snapshot_json = Column(Text, nullable=False)


class ApiErrorEvent(Base):
    __tablename__ = "api_error_events"

    id = Column(Integer, primary_key=True)
    request_id = Column(String(36), nullable=False)
    path = Column(String(300), nullable=False)
    status_code = Column(Integer, nullable=False)
    occurred_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)


class ReleaseChannel(Base):
    __tablename__ = "release_channels"

    name = Column(String(20), primary_key=True)
    environment = Column(String(20), nullable=False, index=True)
    active_run_id = Column(String(100), nullable=True)
    previous_run_id = Column(String(100), nullable=True)
    generation = Column(Integer, nullable=False, default=0)
    updated_by = Column(String(50), nullable=True)
    updated_at = Column(DateTime(timezone=True), nullable=True)


class ReleaseEvent(Base):
    __tablename__ = "release_events"

    id = Column(Integer, primary_key=True)
    channel = Column(String(20), nullable=False, index=True)
    environment = Column(String(20), nullable=False)
    action = Column(String(20), nullable=False)
    from_run_id = Column(String(100), nullable=True)
    to_run_id = Column(String(100), nullable=False)
    generation = Column(Integer, nullable=False)
    actor = Column(String(50), nullable=False)
    reason = Column(String(500), nullable=False)
    occurred_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)


class DeviceRegistration(Base):
    __tablename__ = "device_registrations"

    id = Column(String(36), primary_key=True)
    name = Column(String(100), nullable=False)
    channel = Column(String(20), nullable=False, index=True)
    environment = Column(String(20), nullable=False)
    token_sha256 = Column(String(64), nullable=False)
    is_active = Column(Boolean, nullable=False, default=True)
    created_by = Column(String(50), nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    last_seen_at = Column(DateTime(timezone=True), nullable=True)
    reported_run_id = Column(String(100), nullable=True)
    reported_generation = Column(Integer, nullable=True)
    reported_status = Column(String(30), nullable=True)


class DevicePredictionEvent(Base):
    __tablename__ = "device_prediction_events"

    event_id = Column(String(36), primary_key=True)
    device_id = Column(String(36), nullable=False, index=True)
    environment = Column(String(20), nullable=False, index=True)
    run_id = Column(String(100), nullable=False, index=True)
    prediction = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    inference_ms = Column(Float, nullable=True)
    image_key = Column(String(150), nullable=True)
    image_sha256 = Column(String(64), nullable=True)
    observed_at = Column(DateTime(timezone=True), nullable=False)
    received_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    actual_label = Column(String(100), nullable=True)
    reviewed_by = Column(String(50), nullable=True)
    reviewed_at = Column(DateTime(timezone=True), nullable=True)
