"""initial schema

Revision ID: 20260327_0001
Revises:
Create Date: 2026-03-27 00:30:00
"""

from alembic import op
import sqlalchemy as sa


revision = "20260327_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "sessions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("patient_id", sa.String(length=64), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=True),
        sa.Column("risk_level", sa.String(length=16), nullable=False),
        sa.Column("anomaly_count", sa.Integer(), nullable=True),
        sa.Column("anomaly_rate", sa.Float(), nullable=True),
        sa.Column("dominant_class", sa.String(length=8), nullable=True),
        sa.Column("recording_sec", sa.Float(), nullable=True),
        sa.Column("total_beats", sa.Integer(), nullable=True),
        sa.Column("session_data", sa.Text(), nullable=True),
    )
    op.create_index("ix_sessions_patient_time", "sessions", ["patient_id", "timestamp"], unique=False)

    op.create_table(
        "alerts",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("patient_id", sa.String(length=64), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=True),
        sa.Column("risk_level", sa.String(length=16), nullable=False),
        sa.Column("color", sa.String(length=16), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("beat_index", sa.Integer(), nullable=True),
        sa.Column("class_name", sa.String(length=64), nullable=True),
        sa.Column("review_status", sa.String(length=24), nullable=False, server_default="new"),
        sa.Column("reviewer_note", sa.Text(), nullable=True),
        sa.Column("reviewed_by", sa.String(length=64), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_alerts_patient_time", "alerts", ["patient_id", "timestamp"], unique=False)

    op.create_table(
        "agent_logs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("timestamp", sa.DateTime(), nullable=True),
        sa.Column("patient_id", sa.String(length=64), nullable=True),
        sa.Column("tool_name", sa.String(length=64), nullable=False),
        sa.Column("input_summary", sa.Text(), nullable=True),
        sa.Column("output_summary", sa.Text(), nullable=True),
    )

    op.create_table(
        "reports",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("patient_id", sa.String(length=64), nullable=False),
        sa.Column("session_id", sa.Integer(), nullable=True),
        sa.Column("timestamp", sa.DateTime(), nullable=True),
        sa.Column("file_path", sa.Text(), nullable=False),
        sa.Column("report_focus", sa.String(length=128), nullable=True),
        sa.Column("status", sa.String(length=24), nullable=False, server_default="generated"),
        sa.Column("summary", sa.Text(), nullable=True),
    )
    op.create_index("ix_reports_patient_time", "reports", ["patient_id", "timestamp"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_reports_patient_time", table_name="reports")
    op.drop_table("reports")
    op.drop_table("agent_logs")
    op.drop_index("ix_alerts_patient_time", table_name="alerts")
    op.drop_table("alerts")
    op.drop_index("ix_sessions_patient_time", table_name="sessions")
    op.drop_table("sessions")
