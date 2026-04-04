from fastapi import APIRouter

from api.dependencies import get_status
from api.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/")
def root():
    return {
        "name": "ECG Arrhythmia Detection API",
        "version": "1.0.0",
        "docs": "http://127.0.0.1:8000/docs",
        "health": "http://127.0.0.1:8000/health",
    }


@router.get("/health", response_model=HealthResponse)
def health():
    status = get_status()
    return HealthResponse(
        status="ok" if status["models_loaded"] else "degraded",
        models_loaded=status["models_loaded"],
        agent_ready=status["agent_ready"],
        db_connected=status["db_connected"],
        groq_available=status["groq_available"],
    )
