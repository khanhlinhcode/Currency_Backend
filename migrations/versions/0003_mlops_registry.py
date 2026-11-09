"""Add queryable run, benchmark and monitoring metadata."""

from alembic import op
import sqlalchemy as sa


revision = "0003_mlops_registry"
down_revision = "0002_queue_latency"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    existing = set(sa.inspect(bind).get_table_names())
    if "mlops_runs" not in existing:
        op.create_table("mlops_runs",
            sa.Column("run_id", sa.String(100), primary_key=True),
            sa.Column("environment", sa.String(20), nullable=False),
            sa.Column("status", sa.String(20), nullable=False),
            sa.Column("model_version", sa.String(100), nullable=False),
            sa.Column("dataset_version", sa.String(100), nullable=False),
            sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("object_prefix", sa.String(300), nullable=False),
            sa.Column("record_sha256", sa.String(64), nullable=False),
            sa.Column("summary_json", sa.Text(), nullable=False))
    if "mlops_benchmarks" not in existing:
        op.create_table("mlops_benchmarks",
            sa.Column("benchmark_id", sa.String(100), primary_key=True),
            sa.Column("environment", sa.String(20), nullable=False),
            sa.Column("model_version", sa.String(100), nullable=False),
            sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("object_key", sa.String(300), nullable=False),
            sa.Column("report_sha256", sa.String(64), nullable=False),
            sa.Column("summary_json", sa.Text(), nullable=False))
    if "mlops_monitor_snapshots" not in existing:
        op.create_table("mlops_monitor_snapshots",
            sa.Column("snapshot_id", sa.String(100), primary_key=True),
            sa.Column("environment", sa.String(20), nullable=False),
            sa.Column("checked_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("status", sa.String(30), nullable=False),
            sa.Column("snapshot_json", sa.Text(), nullable=False))
    if "api_error_events" not in existing:
        op.create_table("api_error_events",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("request_id", sa.String(36), nullable=False),
            sa.Column("path", sa.String(300), nullable=False),
            sa.Column("status_code", sa.Integer(), nullable=False),
            sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False))
    for table, columns in (("mlops_runs", ("environment", "model_version", "dataset_version")),
                           ("mlops_benchmarks", ("environment", "model_version")),
                           ("mlops_monitor_snapshots", ("environment", "checked_at")),
                           ("api_error_events", ("occurred_at",))):
        existing_indexes = {index["name"] for index in sa.inspect(bind).get_indexes(table)}
        for column in columns:
            name = f"ix_{table}_{column}"
            if name not in existing_indexes:
                op.create_index(name, table, [column])


def downgrade():
    raise RuntimeError("Restore a database backup to remove MLOps registry tables")
