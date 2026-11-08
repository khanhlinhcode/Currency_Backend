"""Persist model queue wait separately from execution time."""

from alembic import op
import sqlalchemy as sa


revision = "0002_queue_latency"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    columns = {column["name"] for column in sa.inspect(bind).get_columns("prediction_history")}
    if "queue_time" not in columns:
        op.add_column("prediction_history", sa.Column("queue_time", sa.Float(), nullable=True))


def downgrade():
    op.drop_column("prediction_history", "queue_time")
