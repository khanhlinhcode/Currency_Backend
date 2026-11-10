"""Add auditable release channels and pull-based device registrations."""

from alembic import op
import sqlalchemy as sa

revision = "0004_release_channels"
down_revision = "0003_mlops_registry"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    existing = set(sa.inspect(bind).get_table_names())
    if "release_channels" not in existing:
        op.create_table("release_channels",
        sa.Column("name", sa.String(20), primary_key=True),
        sa.Column("environment", sa.String(20), nullable=False),
        sa.Column("active_run_id", sa.String(100)),
        sa.Column("previous_run_id", sa.String(100)),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("updated_by", sa.String(50)),
        sa.Column("updated_at", sa.DateTime(timezone=True)))
    if "release_events" not in existing:
        op.create_table("release_events",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("channel", sa.String(20), nullable=False),
        sa.Column("environment", sa.String(20), nullable=False),
        sa.Column("action", sa.String(20), nullable=False),
        sa.Column("from_run_id", sa.String(100)),
        sa.Column("to_run_id", sa.String(100), nullable=False),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("actor", sa.String(50), nullable=False),
        sa.Column("reason", sa.String(500), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False))
    if "device_registrations" not in existing:
        op.create_table("device_registrations",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("channel", sa.String(20), nullable=False),
        sa.Column("environment", sa.String(20), nullable=False),
        sa.Column("token_sha256", sa.String(64), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_by", sa.String(50), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True)),
        sa.Column("reported_run_id", sa.String(100)),
        sa.Column("reported_generation", sa.Integer()),
        sa.Column("reported_status", sa.String(30)))
    if "device_prediction_events" not in existing:
        op.create_table("device_prediction_events",
            sa.Column("event_id", sa.String(36), primary_key=True),
            sa.Column("device_id", sa.String(36), nullable=False),
            sa.Column("environment", sa.String(20), nullable=False),
            sa.Column("run_id", sa.String(100), nullable=False),
            sa.Column("prediction", sa.String(100), nullable=False),
            sa.Column("confidence", sa.Float(), nullable=False),
            sa.Column("inference_ms", sa.Float()),
            sa.Column("image_key", sa.String(150)),
            sa.Column("image_sha256", sa.String(64)),
            sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("actual_label", sa.String(100)),
            sa.Column("reviewed_by", sa.String(50)),
            sa.Column("reviewed_at", sa.DateTime(timezone=True)))
    for table, column in (("release_channels", "environment"),
                          ("release_events", "channel"),
                          ("device_registrations", "channel"),
                          ("device_prediction_events", "device_id"),
                          ("device_prediction_events", "environment"),
                          ("device_prediction_events", "run_id")):
        index = f"ix_{table}_{column}"
        if index not in {item["name"] for item in sa.inspect(bind).get_indexes(table)}:
            op.create_index(index, table, [column])


def downgrade():
    raise RuntimeError("Restore a database backup to remove release channels")
