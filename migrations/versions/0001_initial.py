"""Initial production schema; safely adopts the existing demo SQLite tables."""

from alembic import op
import sqlalchemy as sa

from api.database import Base
from api import models  # noqa: F401


revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    Base.metadata.create_all(bind=bind, checkfirst=True)
    inspector = sa.inspect(bind)
    history_columns = {column["name"] for column in inspector.get_columns("prediction_history")}
    additions = {
        "image_path": sa.Column("image_path", sa.String(), nullable=True),
        "top3_json": sa.Column("top3_json", sa.String(), nullable=True),
        "requested_by": sa.Column("requested_by", sa.String(), nullable=False, server_default="guest"),
        "currency_category": sa.Column("currency_category", sa.String(), nullable=False, server_default="unknown"),
        "processing_time": sa.Column("processing_time", sa.Float(), nullable=True),
        "model_version": sa.Column("model_version", sa.String(50), nullable=True),
        "actual_label": sa.Column("actual_label", sa.String(100), nullable=True),
        "reviewed_by": sa.Column("reviewed_by", sa.String(50), nullable=True),
        "reviewed_at": sa.Column("reviewed_at", sa.DateTime(), nullable=True),
    }
    for name, column in additions.items():
        if name not in history_columns:
            op.add_column("prediction_history", column)

    user_columns = {column["name"] for column in sa.inspect(bind).get_columns("users")}
    if "role" not in user_columns:
        op.add_column("users", sa.Column("role", sa.String(20), nullable=False, server_default="analyst"))
    bind.execute(sa.text("UPDATE users SET role = CASE WHEN is_admin IS TRUE THEN 'admin' ELSE 'analyst' END WHERE role IS NULL OR role = ''"))


def downgrade():
    raise RuntimeError("Initial migration cannot be safely rolled back; restore a database backup")
