import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.layer3_models.inference import ModelLoader
from src.layer4_agent import ECGAgent, check_llm_status
from src.layer6_storage import ECGDatabase, resolve_database_url

logger = logging.getLogger(__name__)
MODELS_DIR = Path(os.getenv("MODELS_PATH", r"D:\ecg_project\models"))
DB_PATH = os.getenv("DB_PATH", r"D:\ecg_project\database\patient_history.db")
DATABASE_URL = resolve_database_url(DB_PATH, os.getenv("DATABASE_URL"))

_loader: ModelLoader = None
_agent: ECGAgent = None
_db: ECGDatabase = None


def startup():
    global _loader, _agent, _db
    logger.info("[startup] Loading models...")
    _loader = ModelLoader(str(MODELS_DIR))
    _loader.load_all()
    logger.info("[startup] Connecting database...")
    _db = ECGDatabase(db_url=DATABASE_URL)
    logger.info("[startup] Starting agent...")
    try:
        _agent = ECGAgent(db_url=DATABASE_URL)
        _agent.db = _db
        _agent.start()
        logger.info("[startup] Agent ready.")
    except Exception as e:
        logger.warning(f"[startup] Agent unavailable: {e}")
        _agent = None
    logger.info("[startup] All resources ready.")


def shutdown():
    logger.info("[shutdown] Cleaning up.")


def get_loader() -> ModelLoader:
    if _loader is None or not _loader.is_loaded:
        raise RuntimeError("Models not loaded.")
    return _loader


def get_agent() -> ECGAgent:
    if _agent is None:
        raise RuntimeError(
            "Agent not available. Check GROQ_API_KEY in .env and internet connection."
        )
    return _agent


def get_agent_optional():
    return _agent


def get_db() -> ECGDatabase:
    if _db is None:
        raise RuntimeError("Database not connected.")
    return _db


def get_status() -> dict:
    llm = check_llm_status()
    return {
        "models_loaded": _loader is not None and _loader.is_loaded,
        "agent_ready": _agent is not None and _agent.ready,
        "db_connected": _db is not None,
        "groq_available": llm["available"],
        "db_backend": _db.backend if _db is not None else "unavailable",
    }

