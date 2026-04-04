import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv
from sqlalchemy import Column, DateTime, Float, Index, Integer, String, Text, create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()

DB_PATH = os.getenv("DB_PATH", "D:/ecg_project/database/patient_history.db")
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "ecg_arrhythmia")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_ECHO = os.getenv("DB_ECHO", "False").lower() == "true"
ANONYMIZE = os.getenv("ANONYMIZE_PATIENT_IDS", "True").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)

Base = declarative_base()


def resolve_database_url(explicit_db_path: str | None = None, explicit_db_url: str | None = None) -> str:
    if explicit_db_url:
        return explicit_db_url
    if DATABASE_URL:
        return DATABASE_URL

    if all([DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD]) and os.getenv("DB_BACKEND", "sqlite").lower() == "postgresql":
        return str(
            URL.create(
                "postgresql+psycopg",
                username=DB_USER,
                password=DB_PASSWORD,
                host=DB_HOST,
                port=int(DB_PORT),
                database=DB_NAME,
            )
        )

    db_path = explicit_db_path or DB_PATH
    normalized = Path(db_path).expanduser().resolve()
    return f"sqlite:///{normalized.as_posix()}"


def database_backend_label(database_url: str) -> str:
    if database_url.startswith("postgresql"):
        return "PostgreSQL"
    if database_url.startswith("sqlite"):
        return "SQLite"
    return "SQLAlchemy"


class Session(Base):
    """One monitoring session per patient."""

    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String(64), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    risk_level = Column(String(16), nullable=False)
    anomaly_count = Column(Integer, default=0)
    anomaly_rate = Column(Float, default=0.0)
    dominant_class = Column(String(8), default="N")
    recording_sec = Column(Float, default=0.0)
    total_beats = Column(Integer, default=0)
    session_data = Column(Text, nullable=True)


class Alert(Base):
    """One alert fired per anomaly event."""

    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String(64), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    risk_level = Column(String(16), nullable=False)
    color = Column(String(16), nullable=False)
    message = Column(Text, nullable=False)
    beat_index = Column(Integer, nullable=True)
    class_name = Column(String(64), nullable=True)
    review_status = Column(String(24), nullable=False, default="new")
    reviewer_note = Column(Text, nullable=True)
    reviewed_by = Column(String(64), nullable=True)
    reviewed_at = Column(DateTime, nullable=True)


class AgentLog(Base):
    """Audit trail for agent tool calls."""

    __tablename__ = "agent_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    patient_id = Column(String(64), nullable=True)
    tool_name = Column(String(64), nullable=False)
    input_summary = Column(Text, nullable=True)
    output_summary = Column(Text, nullable=True)


class ReportRecord(Base):
    """Generated patient document metadata."""

    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String(64), nullable=False, index=True)
    session_id = Column(Integer, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    file_path = Column(Text, nullable=False)
    report_focus = Column(String(128), nullable=True)
    status = Column(String(24), nullable=False, default="generated")
    summary = Column(Text, nullable=True)


Index("ix_sessions_patient_time", Session.patient_id, Session.timestamp)
Index("ix_alerts_patient_time", Alert.patient_id, Alert.timestamp)
Index("ix_reports_patient_time", ReportRecord.patient_id, ReportRecord.timestamp)


class ECGDatabase:
    """Persistence layer for sessions, alerts, reports, and agent audit logs."""

    def __init__(self, db_path: str | None = None, db_url: str | None = None):
        self.database_url = resolve_database_url(db_path, db_url)
        self.backend = database_backend_label(self.database_url)

        if self.database_url.startswith("sqlite:///"):
            sqlite_path = Path(self.database_url.replace("sqlite:///", "", 1))
            sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            engine_kwargs = {"connect_args": {"check_same_thread": False}}
        else:
            engine_kwargs = {"pool_pre_ping": True}

        self.engine = create_engine(self.database_url, echo=DB_ECHO, future=True, **engine_kwargs)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self._ensure_alert_review_columns()
        self._ensure_report_columns()
        logger.info("[db] Connected: %s (%s)", self.database_url, self.backend)

    def _ensure_alert_review_columns(self):
        if not self.database_url.startswith("sqlite"):
            return
        with self.engine.begin() as connection:
            columns = {row[1] for row in connection.execute(text("PRAGMA table_info(alerts)")).fetchall()}

            if "review_status" not in columns:
                connection.execute(text("ALTER TABLE alerts ADD COLUMN review_status VARCHAR(24) NOT NULL DEFAULT 'new'"))
            if "reviewer_note" not in columns:
                connection.execute(text("ALTER TABLE alerts ADD COLUMN reviewer_note TEXT"))
            if "reviewed_by" not in columns:
                connection.execute(text("ALTER TABLE alerts ADD COLUMN reviewed_by VARCHAR(64)"))
            if "reviewed_at" not in columns:
                connection.execute(text("ALTER TABLE alerts ADD COLUMN reviewed_at DATETIME"))

    def _ensure_report_columns(self):
        if not self.database_url.startswith("sqlite"):
            return
        with self.engine.begin() as connection:
            columns = {row[1] for row in connection.execute(text("PRAGMA table_info(reports)")).fetchall()}

            if not columns:
                return
            if "session_id" not in columns:
                connection.execute(text("ALTER TABLE reports ADD COLUMN session_id INTEGER"))
            if "report_focus" not in columns:
                connection.execute(text("ALTER TABLE reports ADD COLUMN report_focus VARCHAR(128)"))
            if "status" not in columns:
                connection.execute(text("ALTER TABLE reports ADD COLUMN status VARCHAR(24) NOT NULL DEFAULT 'generated'"))
            if "summary" not in columns:
                connection.execute(text("ALTER TABLE reports ADD COLUMN summary TEXT"))

    @staticmethod
    def _anon(patient_id: str) -> str:
        if ANONYMIZE:
            return "PT-" + hashlib.sha256(patient_id.encode()).hexdigest()[:8].upper()
        return patient_id

    def save_session(self, patient_id: str, batch_result) -> int:
        pid = self._anon(patient_id)
        data = {
            "total_beats": batch_result.total_beats,
            "anomaly_count": batch_result.anomaly_count,
            "anomaly_rate": batch_result.anomaly_rate,
            "class_counts": batch_result.class_counts,
            "session_risk": batch_result.session_risk,
            "recording_sec": batch_result.recording_sec,
        }
        with self.Session() as session:
            row = Session(
                patient_id=pid,
                risk_level=batch_result.session_risk,
                anomaly_count=batch_result.anomaly_count,
                anomaly_rate=batch_result.anomaly_rate,
                dominant_class=batch_result.dominant_class,
                recording_sec=batch_result.recording_sec,
                total_beats=batch_result.total_beats,
                session_data=json.dumps(data),
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            logger.info("[db] Session saved: patient=%s risk=%s", pid, batch_result.session_risk)
            return row.id

    def save_report(
        self,
        patient_id: str,
        file_path: str,
        session_id: int | None = None,
        report_focus: str | None = None,
        summary: str | None = None,
        status: str = "generated",
    ) -> int:
        pid = self._anon(patient_id)
        with self.Session() as session:
            row = ReportRecord(
                patient_id=pid,
                session_id=session_id,
                file_path=file_path,
                report_focus=report_focus,
                status=status,
                summary=summary,
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            return row.id

    def save_alert(
        self,
        patient_id: str,
        risk_level: str,
        color: str,
        message: str,
        beat_index: int = None,
        class_name: str = None,
    ) -> int:
        pid = self._anon(patient_id)
        with self.Session() as session:
            row = Alert(
                patient_id=pid,
                risk_level=risk_level,
                color=color,
                message=message,
                beat_index=beat_index,
                class_name=class_name,
                review_status="new",
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            return row.id

    def log_tool_call(
        self,
        tool_name: str,
        input_summary: str,
        output_summary: str,
        patient_id: str = None,
    ):
        pid = self._anon(patient_id) if patient_id else None
        with self.Session() as session:
            session.add(
                AgentLog(
                    patient_id=pid,
                    tool_name=tool_name,
                    input_summary=input_summary[:500],
                    output_summary=output_summary[:500],
                )
            )
            session.commit()

    def get_patient_history(self, patient_id: str, limit: int = 10) -> list:
        pid = self._anon(patient_id)
        with self.Session() as session:
            rows = (
                session.query(Session)
                .filter(Session.patient_id == pid)
                .order_by(Session.timestamp.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "session_id": row.id,
                    "timestamp": row.timestamp.isoformat(),
                    "risk_level": row.risk_level,
                    "anomaly_count": row.anomaly_count,
                    "anomaly_rate": round(row.anomaly_rate * 100, 1),
                    "dominant_class": row.dominant_class,
                    "recording_sec": row.recording_sec,
                    "total_beats": row.total_beats,
                }
                for row in rows
            ]

    def get_recent_alerts(self, patient_id: str, limit: int = 20) -> list:
        pid = self._anon(patient_id)
        with self.Session() as session:
            rows = (
                session.query(Alert)
                .filter(Alert.patient_id == pid)
                .order_by(Alert.timestamp.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "alert_id": row.id,
                    "timestamp": row.timestamp.isoformat(),
                    "risk_level": row.risk_level,
                    "color": row.color,
                    "message": row.message,
                    "class_name": row.class_name,
                    "review_status": row.review_status or "new",
                    "reviewer_note": row.reviewer_note,
                    "reviewed_by": row.reviewed_by,
                    "reviewed_at": row.reviewed_at.isoformat() if row.reviewed_at else None,
                }
                for row in rows
            ]

    def get_patient_reports(self, patient_id: str, limit: int = 20) -> list:
        pid = self._anon(patient_id)
        with self.Session() as session:
            rows = (
                session.query(ReportRecord)
                .filter(ReportRecord.patient_id == pid)
                .order_by(ReportRecord.timestamp.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "report_id": row.id,
                    "session_id": row.session_id,
                    "timestamp": row.timestamp.isoformat(),
                    "file_path": row.file_path,
                    "file_name": os.path.basename(row.file_path),
                    "report_focus": row.report_focus,
                    "status": row.status,
                    "summary": row.summary,
                }
                for row in rows
            ]

    def update_alert_review(
        self,
        patient_id: str,
        alert_id: int,
        review_status: str,
        reviewer_note: str = "",
        reviewed_by: str | None = None,
    ) -> dict:
        pid = self._anon(patient_id)
        with self.Session() as session:
            row = (
                session.query(Alert)
                .filter(Alert.id == alert_id, Alert.patient_id == pid)
                .first()
            )
            if row is None:
                raise ValueError("Alert not found for this patient.")

            row.review_status = review_status
            row.reviewer_note = reviewer_note or None
            row.reviewed_by = reviewed_by
            row.reviewed_at = datetime.utcnow() if review_status != "new" else None
            session.commit()
            session.refresh(row)
            return {
                "alert_id": row.id,
                "timestamp": row.timestamp.isoformat(),
                "risk_level": row.risk_level,
                "color": row.color,
                "message": row.message,
                "class_name": row.class_name,
                "review_status": row.review_status or "new",
                "reviewer_note": row.reviewer_note,
                "reviewed_by": row.reviewed_by,
                "reviewed_at": row.reviewed_at.isoformat() if row.reviewed_at else None,
            }

    def get_anomaly_trend(self, patient_id: str, last_n_sessions: int = 5) -> dict:
        history = self.get_patient_history(patient_id, limit=last_n_sessions)
        if not history:
            return {"trend": "no_data", "sessions": 0, "avg_anomaly_rate": 0.0}

        rates = [item["anomaly_rate"] for item in history]
        avg = round(sum(rates) / len(rates), 1)
        trend = "stable"
        if len(rates) >= 2:
            trend = "worsening" if rates[0] > rates[-1] else "improving"

        return {
            "trend": trend,
            "sessions": len(history),
            "avg_anomaly_rate": avg,
            "latest_rate": rates[0] if rates else 0.0,
            "history": history,
        }
