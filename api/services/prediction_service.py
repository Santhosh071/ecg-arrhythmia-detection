import logging

import numpy as np

from src.layer3_models.inference import BEAT_LENGTH, predict_beat, predict_beats

logger = logging.getLogger(__name__)

MAX_BEATS_PER_REQUEST = 500
RISK_COLOR_MAP = {"Medium": "yellow", "High": "orange", "Critical": "red"}


def predict_batch_result(req, loader, db):
    beats = np.array(req.beats, dtype=np.float32)
    times = np.array(req.timestamps, dtype=np.float64)

    if beats.ndim != 2 or beats.shape[1] != BEAT_LENGTH:
        raise ValueError(f"beats must be shape (N, {BEAT_LENGTH}). Got {beats.shape}.")
    if len(beats) != len(times):
        raise ValueError("beats and timestamps must have equal length.")

    truncated = len(beats) > MAX_BEATS_PER_REQUEST
    if truncated:
        beats = beats[:MAX_BEATS_PER_REQUEST]
        times = times[:MAX_BEATS_PER_REQUEST]
        logger.warning("Prediction request truncated to %s beats.", MAX_BEATS_PER_REQUEST)

    batch = predict_beats(beats, times, loader, batch_size=64)

    saved = False
    if getattr(req, "save_session", True) and db is not None:
        db.save_session(req.patient_id, batch)
        saved = True

        if batch.session_risk != "Low":
            db.save_alert(
                patient_id=req.patient_id,
                risk_level=batch.session_risk,
                color=RISK_COLOR_MAP.get(batch.session_risk, "yellow"),
                message=f"{batch.anomaly_count} anomalies - {batch.anomaly_rate*100:.1f}% anomaly rate",
                beat_index=-1,
                class_name=batch.dominant_class,
            )

    return {
        "patient_id": req.patient_id,
        "total_beats": batch.total_beats,
        "anomaly_count": batch.anomaly_count,
        "anomaly_rate": batch.anomaly_rate,
        "class_counts": {str(k): v for k, v in batch.class_counts.items()},
        "dominant_class": batch.dominant_class,
        "session_risk": batch.session_risk,
        "recording_sec": batch.recording_sec,
        "beats": [b.to_dict() for b in batch.beats],
        "session_saved": saved,
        "request_truncated": truncated,
    }


def predict_single_beat_result(req, loader):
    beat = np.array(req.beat, dtype=np.float32)
    if len(beat) != BEAT_LENGTH:
        raise ValueError(f"beat must be {BEAT_LENGTH} samples. Got {len(beat)}.")

    result = predict_beat(
        beat,
        loader,
        beat_index=req.beat_index,
        timestamp=req.timestamp,
    )
    return result.to_dict()
