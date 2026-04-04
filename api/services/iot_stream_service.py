from __future__ import annotations

from api.services.prediction_service import RISK_COLOR_MAP
from src.layer1_ingestion.iot import stream_buffer
from src.layer1_ingestion.iot.iot_models import analyze_live_stream


def ingest_iot_stream(req) -> dict:
    state = stream_buffer.ingest(
        device_id=req.device_id,
        patient_id=req.patient_id,
        samples=req.samples,
        sampling_rate=req.sampling_rate,
        start_timestamp_ms=req.start_timestamp_ms,
        lead_off=req.lead_off,
    )
    return {
        "device_id": state.device_id,
        "patient_id": state.patient_id,
        "samples_received": len(req.samples),
        "buffered_samples": len(state.samples),
        "buffered_seconds": round(len(state.samples) / state.sampling_rate, 2),
        "lead_off": state.lead_off,
        "last_seen_ms": state.last_seen_ms,
        "total_samples_received": state.total_samples_received,
    }


def get_iot_status(device_id: str) -> dict:
    return stream_buffer.get_status(device_id)


def get_iot_live_window(device_id: str, window_sec: float) -> dict:
    window = stream_buffer.get_recent_window(device_id, window_sec=window_sec)
    return {
        "device_id": window["device_id"],
        "patient_id": window["patient_id"],
        "sampling_rate": window["sampling_rate"],
        "sample_count": len(window["samples"]),
        "duration_sec": window["duration_sec"],
        "lead_off": window["lead_off"],
        "last_seen_ms": window["last_seen_ms"],
        "samples": window["samples"],
        "timestamps_ms": window["timestamps_ms"],
    }


def analyze_iot_window(req, loader, db=None) -> dict:
    window = stream_buffer.get_recent_window(req.device_id, window_sec=req.window_sec)
    patient_id = req.patient_id or window["patient_id"]
    analysis = analyze_live_stream(
        samples=window["samples"],
        sampling_rate=window["sampling_rate"],
        loader=loader,
        adc_resolution=req.adc_resolution,
        vref=req.vref,
    )
    batch = analysis["batch"]

    saved = False
    if req.save_session and db is not None:
        db.save_session(patient_id, batch)
        saved = True
        if batch.session_risk != "Low":
            db.save_alert(
                patient_id=patient_id,
                risk_level=batch.session_risk,
                color=RISK_COLOR_MAP.get(batch.session_risk, "yellow"),
                message=f"IoT live stream detected {batch.anomaly_count} anomalies",
                beat_index=-1,
                class_name=batch.dominant_class,
            )

    return {
        "device_id": req.device_id,
        "patient_id": patient_id,
        "source": "iot-live-stream",
        "raw_sampling_rate": window["sampling_rate"],
        "model_sampling_rate": analysis["model_sampling_rate"],
        "buffered_duration_sec": window["duration_sec"],
        "total_beats": batch.total_beats,
        "anomaly_count": batch.anomaly_count,
        "anomaly_rate": batch.anomaly_rate,
        "class_counts": {str(k): v for k, v in batch.class_counts.items()},
        "dominant_class": batch.dominant_class,
        "session_risk": batch.session_risk,
        "recording_sec": batch.recording_sec,
        "detected_peaks": int(len(analysis["peaks"])),
        "session_saved": saved,
        "quality": analysis["quality"],
        "beats": [beat.to_dict() for beat in batch.beats],
    }
