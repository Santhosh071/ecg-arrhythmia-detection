import os
import json
import hashlib
import logging
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import (
    create_engine, Column, Integer, String, Float,
    Text, DateTime, Index, text
)
from sqlalchemy.orm import declarative_base, sessionmaker
from langchain.memory import ConversationBufferWindowMemory

load_dotenv()

DB_PATH      = os.getenv("DB_PATH", "D:/ecg_project/database/patient_history.db")
ANONYMIZE    = os.getenv("ANONYMIZE_PATIENT_IDS", "True").lower() == "true"
LOG_LEVEL    = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)
Base = declarative_base()

class Session(Base):
    """One monitoring session per patient."""
    __tablename__ = "sessions"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    patient_id     = Column(String(64),  nullable=False, index=True)
    timestamp      = Column(DateTime,    default=datetime.utcnow)
    risk_level     = Column(String(16),  nullable=False)
    anomaly_count  = Column(Integer,     default=0)
    anomaly_rate   = Column(Float,       default=0.0)
    dominant_class = Column(String(8),   default="N")
    recording_sec  = Column(Float,       default=0.0)
    total_beats    = Column(Integer,     default=0)
    session_data   = Column(Text,        nullable=True)

class Alert(Base):
    """One alert fired per anomaly event."""
    __tablename__ = "alerts"
    id         = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String(64), nullable=False, index=True)
    timestamp  = Column(DateTime,   default=datetime.utcnow)
    risk_level = Column(String(16), nullable=False)
    color      = Column(String(16), nullable=False)
    message    = Column(Text,       nullable=False)
    beat_index = Column(Integer,    nullable=True)
    class_name = Column(String(64), nullable=True)

class AgentLog(Base):
    """Every agent tool call logged here (Rule 20)."""
    __tablename__ = "agent_logs"
    id             = Column(Integer, primary_key=True, autoincrement=True)
    timestamp      = Column(DateTime, default=datetime.utcnow)
    patient_id     = Column(String(64), nullable=True)
    tool_name      = Column(String(64), nullable=False)
    input_summary  = Column(Text,       nullable=True)
    output_summary = Column(Text,       nullable=True)
Index("ix_sessions_patient_time", Session.patient_id, Session.timestamp)
Index("ix_alerts_patient_time",   Alert.patient_id,   Alert.timestamp)

class ECGDatabase:
    """
    SQLite database interface for long-term patient memory.

    Usage
    -----
    db = ECGDatabase()
    db.save_session(patient_id, batch_result)
    history = db.get_patient_history(patient_id)
    """

    def __init__(self, db_path: str = None):
        path      = db_path or DB_PATH
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.engine  = create_engine(f"sqlite:///{path}", echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        logger.info(f"[db] Connected: {path}")
    @staticmethod
    def _anon(patient_id: str) -> str:
        """Anonymise patient ID if ANONYMIZE_PATIENT_IDS=True (Rule 28)."""
        if ANONYMIZE:
            return "PT-" + hashlib.sha256(patient_id.encode()).hexdigest()[:8].upper()
        return patient_id

    def save_session(self, patient_id: str, batch_result) -> int:
        """
        Save a completed monitoring session.

        Parameters
        ----------
        patient_id   : raw patient identifier
        batch_result : BatchResult from predict.py

        Returns
        -------
        session id (int)
        """
        pid  = self._anon(patient_id)
        data = {
            "total_beats"   : batch_result.total_beats,
            "anomaly_count" : batch_result.anomaly_count,
            "anomaly_rate"  : batch_result.anomaly_rate,
            "class_counts"  : batch_result.class_counts,
            "session_risk"  : batch_result.session_risk,
            "recording_sec" : batch_result.recording_sec,
        }
        with self.Session() as s:
            row = Session(
                patient_id     = pid,
                risk_level     = batch_result.session_risk,
                anomaly_count  = batch_result.anomaly_count,
                anomaly_rate   = batch_result.anomaly_rate,
                dominant_class = batch_result.dominant_class,
                recording_sec  = batch_result.recording_sec,
                total_beats    = batch_result.total_beats,
                session_data   = json.dumps(data),
            )
            s.add(row)
            s.commit()
            s.refresh(row)
            logger.info(f"[db] Session saved: patient={pid} risk={batch_result.session_risk}")
            return row.id

    def save_alert(self, patient_id: str, risk_level: str, color: str,
                   message: str, beat_index: int = None, class_name: str = None) -> int:
        """Save one alert to the database."""
        pid = self._anon(patient_id)
        with self.Session() as s:
            row = Alert(
                patient_id = pid,
                risk_level = risk_level,
                color      = color,
                message    = message,
                beat_index = beat_index,
                class_name = class_name,
            )
            s.add(row)
            s.commit()
            s.refresh(row)
            return row.id

    def log_tool_call(self, tool_name: str, input_summary: str,
                      output_summary: str, patient_id: str = None):
        """Log every agent tool call (Rule 20)."""
        pid = self._anon(patient_id) if patient_id else None
        with self.Session() as s:
            s.add(AgentLog(
                patient_id     = pid,
                tool_name      = tool_name,
                input_summary  = input_summary[:500],
                output_summary = output_summary[:500],
            ))
            s.commit()

    def get_patient_history(self, patient_id: str, limit: int = 10) -> list:
        """
        Retrieve last N sessions for a patient.

        Returns list of dicts with session summary.
        """
        pid = self._anon(patient_id)
        with self.Session() as s:
            rows = (
                s.query(Session)
                .filter(Session.patient_id == pid)
                .order_by(Session.timestamp.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "session_id"   : r.id,
                    "timestamp"    : r.timestamp.isoformat(),
                    "risk_level"   : r.risk_level,
                    "anomaly_count": r.anomaly_count,
                    "anomaly_rate" : round(r.anomaly_rate * 100, 1),
                    "dominant_class": r.dominant_class,
                    "recording_sec": r.recording_sec,
                    "total_beats"  : r.total_beats,
                }
                for r in rows
            ]

    def get_recent_alerts(self, patient_id: str, limit: int = 20) -> list:
        """Retrieve recent alerts for a patient."""
        pid = self._anon(patient_id)
        with self.Session() as s:
            rows = (
                s.query(Alert)
                .filter(Alert.patient_id == pid)
                .order_by(Alert.timestamp.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "alert_id"  : r.id,
                    "timestamp" : r.timestamp.isoformat(),
                    "risk_level": r.risk_level,
                    "color"     : r.color,
                    "message"   : r.message,
                    "class_name": r.class_name,
                }
                for r in rows
            ]

    def get_anomaly_trend(self, patient_id: str, last_n_sessions: int = 5) -> dict:
        """
        Compute anomaly trend across last N sessions.
        Used by monitor_trends() tool.
        """
        history = self.get_patient_history(patient_id, limit=last_n_sessions)
        if not history:
            return {"trend": "no_data", "sessions": 0, "avg_anomaly_rate": 0.0}

        rates = [h["anomaly_rate"] for h in history]
        avg   = round(sum(rates) / len(rates), 1)

        if len(rates) >= 2:
            trend = "worsening" if rates[0] > rates[-1] else "improving"
        else:
            trend = "stable"

        return {
            "trend"            : trend,
            "sessions"         : len(history),
            "avg_anomaly_rate" : avg,
            "latest_rate"      : rates[0] if rates else 0.0,
            "history"          : history,
        }

def get_short_term_memory(k: int = 10) -> ConversationBufferWindowMemory:
    """
    Return a fresh short-term conversation memory window.
    k = number of message pairs to retain (default 10).
    """
    return ConversationBufferWindowMemory(
        k                   = k,
        memory_key          = "chat_history",
        return_messages     = True,
        output_key          = "output",
    )