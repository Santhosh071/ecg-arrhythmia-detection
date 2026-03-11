import os
import sys
import json
import logging
import numpy as np
from contextlib import asynccontextmanager
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from api.dependencies import startup, shutdown, get_loader, get_agent, get_db, get_status
from api.schemas import (
    HealthResponse,
    PredictRequest,    BatchResultResponse, BeatResultResponse,
    SingleBeatRequest,
    AgentQueryRequest, AgentResponse,
    ReportRequest,     ReportResponse,
    SignalQualityRequest, SignalQualityResponse,
    HistoryResponse,   AlertsResponse,
)
from src.layer3_models.inference import predict_beats, predict_beat, BEAT_LENGTH
from src.layer3_models.inference.preprocess import check_signal_quality
logging.basicConfig(
    level  = getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== ECG API starting ===")
    startup()
    logger.info("=== ECG API ready ===")
    yield
    shutdown()

app = FastAPI(
    title       = "ECG Arrhythmia Detection API",
    description = (
        "AI-powered ECG arrhythmia detection system.\n\n"
        "**DISCLAIMER:** Clinical decision-support tool only. "
        "All outputs require review by a qualified clinician."
    ),
    version  = "1.0.0",
    lifespan = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

@app.get("/", tags=["Health"])
def root():
    return {
        "name"   : "ECG Arrhythmia Detection API",
        "version": "1.0.0",
        "docs"   : "http://127.0.0.1:8000/docs",
        "health" : "http://127.0.0.1:8000/health",
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    """Check server status — models, agent, DB, Groq."""
    s = get_status()
    return HealthResponse(
        status         = "ok" if s["models_loaded"] else "degraded",
        models_loaded  = s["models_loaded"],
        agent_ready    = s["agent_ready"],
        db_connected   = s["db_connected"],
        groq_available = s["groq_available"],
    )

@app.post("/predict", response_model=BatchResultResponse, tags=["Prediction"])
def predict_batch(req: PredictRequest):
    """
    Run inference on a batch of ECG beats.

    Send beats as a list of lists — each inner list = 187 floats (one beat).
    Returns full BatchResult with per-beat predictions and session stats.
    """
    try:
        loader = get_loader()
        beats  = np.array(req.beats,      dtype=np.float32)
        times  = np.array(req.timestamps, dtype=np.float64)

        if beats.ndim != 2 or beats.shape[1] != BEAT_LENGTH:
            raise HTTPException(
                status_code=422,
                detail=f"beats must be shape (N, {BEAT_LENGTH}). Got {beats.shape}."
            )
        if len(beats) != len(times):
            raise HTTPException(
                status_code=422,
                detail="beats and timestamps must have equal length."
            )
        batch = predict_beats(beats, times, loader, batch_size=64)
        saved = False
        try:
            db = get_db()
            db.save_session(req.patient_id, batch)
            saved = True
        except Exception as e:
            logger.warning(f"Session save skipped: {e}")
        return BatchResultResponse(
            patient_id     = req.patient_id,
            total_beats    = batch.total_beats,
            anomaly_count  = batch.anomaly_count,
            anomaly_rate   = batch.anomaly_rate,
            class_counts   = {str(k): v for k, v in batch.class_counts.items()},
            dominant_class = batch.dominant_class,
            session_risk   = batch.session_risk,
            recording_sec  = batch.recording_sec,
            beats          = [BeatResultResponse(**b.to_dict()) for b in batch.beats],
            session_saved  = saved,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/predict error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/single", response_model=BeatResultResponse, tags=["Prediction"])
def predict_single(req: SingleBeatRequest):
    """
    Run inference on a single ECG beat (exactly 187 samples).
    Faster than the batch endpoint for single-beat inspection.
    """
    try:
        loader = get_loader()
        beat   = np.array(req.beat, dtype=np.float32)
        if len(beat) != BEAT_LENGTH:
            raise HTTPException(
                status_code=422,
                detail=f"beat must be {BEAT_LENGTH} samples. Got {len(beat)}."
            )
        result = predict_beat(beat, loader,
                              beat_index=req.beat_index,
                              timestamp=req.timestamp)
        return BeatResultResponse(**result.to_dict())
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/predict/single error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent/query", response_model=AgentResponse, tags=["Agent"])
def agent_query(req: AgentQueryRequest):
    """
    Send a natural language question to the ECGAgent (Groq LLaMA3).
    Returns LLM-generated answer grounded in patient context.

    DISCLAIMER: AI-generated. Requires clinician review.
    """
    try:
        agent  = get_agent()
        answer = agent.run(req.question, patient_id=req.patient_id)
        return AgentResponse(answer=answer, patient_id=req.patient_id)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"/agent/query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/report/generate", response_model=ReportResponse, tags=["Report"])
def generate_report(req: ReportRequest):
    """Generate a PDF session report. Returns saved file path."""
    try:
        agent  = get_agent()
        result = agent.run_tool_directly(
            "generate_report",
            json.dumps({"patient_id": req.patient_id, "session_data": req.session_data})
        )
        success = "saved" in result.lower()
        return ReportResponse(
            success    = success,
            file_path  = result.replace("Report saved: ", "").strip() if success else "",
            patient_id = req.patient_id,
            message    = result,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"/report/generate error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/signal/quality", response_model=SignalQualityResponse, tags=["Signal"])
def signal_quality(req: SignalQualityRequest):
    """
    Analyse raw ECG signal quality.
    Returns SNR, estimated HR, and any detected issues (flat-line, clipping etc).
    """
    try:
        signal = np.array(req.signal, dtype=np.float64)
        result = check_signal_quality(signal, fs=req.fs)
        return SignalQualityResponse(patient_id=req.patient_id, **result)
    except Exception as e:
        logger.error(f"/signal/quality error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/{patient_id}", response_model=HistoryResponse, tags=["History"])
def get_history(
    patient_id : str,
    limit      : int = Query(10, ge=1, le=100),
):
    """Retrieve past monitoring sessions for a patient. Newest first."""
    try:
        db       = get_db()
        sessions = db.get_patient_history(patient_id, limit=limit)
        return HistoryResponse(patient_id=patient_id, sessions=sessions, total=len(sessions))
    except Exception as e:
        logger.error(f"/history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/{patient_id}/alerts", response_model=AlertsResponse, tags=["History"])
def get_alerts(
    patient_id : str,
    limit      : int = Query(20, ge=1, le=200),
):
    """Retrieve recent alerts for a patient."""
    try:
        db     = get_db()
        alerts = db.get_recent_alerts(patient_id, limit=limit)
        return AlertsResponse(patient_id=patient_id, alerts=alerts, total=len(alerts))
    except Exception as e:
        logger.error(f"/history/alerts error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/{patient_id}/trend", tags=["History"])
def get_trend(
    patient_id      : str,
    last_n_sessions : int = Query(5, ge=1, le=50),
):
    """Anomaly rate trend across last N sessions — improving / worsening / stable."""
    try:
        db    = get_db()
        trend = db.get_anomaly_trend(patient_id, last_n_sessions=last_n_sessions)
        return {"patient_id": patient_id, **trend}
    except Exception as e:
        logger.error(f"/history/trend error: {e}")
        raise HTTPException(status_code=500, detail=str(e))