import torch
import numpy as np
from .model_loader  import ModelLoader, BEAT_LENGTH, DEVICE
from .result_schema import (
    BeatResult, BatchResult,
    CLASS_NAMES, SHORT_NAMES, CLASS_RISK, CLASS_ALERT_COLOR,
    compute_session_risk,
)


def _build_beat_result(
    idx         : int,
    ts          : float,
    t_error     : float,
    t_zscore    : float,
    t_flag      : bool,
    l_error     : float,
    l_flag      : bool,
    class_id    : int,
    confidence  : float,
    all_probs   : list,
) -> BeatResult:
    is_anomaly = t_flag or l_flag or (class_id != 0)
    return BeatResult(
        beat_index          = idx,
        timestamp_sec       = float(ts),
        is_anomaly          = is_anomaly,
        transformer_anomaly = t_flag,
        lstm_anomaly        = l_flag,
        transformer_score   = round(t_zscore, 4),
        transformer_error   = round(t_error,  6),
        lstm_error          = round(l_error,  6),
        cnn_class_id        = class_id,
        cnn_class_name      = CLASS_NAMES.get(class_id, "Unknown"),
        cnn_short_name      = SHORT_NAMES.get(class_id, "?"),
        cnn_confidence      = round(confidence, 4),
        cnn_all_probs       = [round(float(p), 4) for p in all_probs],
        risk_level          = CLASS_RISK.get(class_id, "Medium"),
        alert_color         = CLASS_ALERT_COLOR.get(class_id, "yellow"),
    )


def _zscore(error: float, mean: float, std: float) -> float:
    return (error - mean) / std if std > 1e-12 else 0.0


def _trans_flag(loader: ModelLoader, error, zscore) -> bool:
    if loader.trans_z_thresh is not None:
        return bool(zscore > loader.trans_z_thresh)
    return bool(error > loader.trans_threshold)


@torch.no_grad()
def predict_beat(
    beat       : np.ndarray,
    loader     : ModelLoader,
    beat_index : int   = 0,
    timestamp  : float = 0.0,
) -> BeatResult:
    if not loader.is_loaded:
        raise RuntimeError("Call ModelLoader.load_all() before predict_beat().")
    if len(beat) != BEAT_LENGTH:
        raise ValueError(f"Beat must be {BEAT_LENGTH} samples. Got {len(beat)}.")
    x     = torch.tensor(beat, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    x_cnn = x.unsqueeze(1)
    t_error  = float(loader.transformer_ae.reconstruction_error(x).item())
    t_zscore = _zscore(t_error, loader.trans_val_mean or 0.0, loader.trans_val_std or 1.0)
    t_flag   = _trans_flag(loader, t_error, t_zscore)
    l_error  = float(loader.lstm_ae.reconstruction_error(x).item())
    l_flag   = bool(l_error > loader.lstm_threshold)
    probs      = torch.softmax(loader.cnn(x_cnn), dim=1)
    class_id   = int(probs.argmax(dim=1).item())
    confidence = float(probs[0, class_id].item())
    return _build_beat_result(
        beat_index, timestamp,
        t_error, t_zscore, t_flag,
        l_error, l_flag,
        class_id, confidence, probs[0].cpu().numpy(),
    )


@torch.no_grad()
def predict_beats(
    beats      : np.ndarray,
    timestamps : np.ndarray,
    loader     : ModelLoader,
    batch_size : int = 64,
) -> BatchResult:
    if not loader.is_loaded:
        raise RuntimeError("Call ModelLoader.load_all() before predict_beats().")
    if len(beats) == 0:
        return BatchResult(
            beats=[], total_beats=0, anomaly_count=0, anomaly_rate=0.0,
            class_counts={}, dominant_class="N/A", session_risk="Low", recording_sec=0.0,
        )
    results = []
    vmean   = loader.trans_val_mean or 0.0
    vstd    = loader.trans_val_std  or 1.0
    for start in range(0, len(beats), batch_size):
        end   = min(start + batch_size, len(beats))
        x     = torch.tensor(beats[start:end], dtype=torch.float32).to(DEVICE)
        x_cnn = x.unsqueeze(1)
        ts    = timestamps[start:end]
        t_errors  = loader.transformer_ae.reconstruction_error(x).cpu().numpy()
        t_zscores = (t_errors - vmean) / vstd if vstd > 1e-12 else np.zeros(len(t_errors))
        t_flags   = (t_zscores > loader.trans_z_thresh if loader.trans_z_thresh is not None
                     else t_errors > loader.trans_threshold)
        l_errors  = loader.lstm_ae.reconstruction_error(x).cpu().numpy()
        l_flags   = l_errors > loader.lstm_threshold
        probs     = torch.softmax(loader.cnn(x_cnn), dim=1)
        class_ids = probs.argmax(dim=1).cpu().numpy()
        confs     = probs.max(dim=1).values.cpu().numpy()
        all_probs = probs.cpu().numpy()
        for i in range(len(x)):
            results.append(_build_beat_result(
                start + i, float(ts[i]),
                float(t_errors[i]), float(t_zscores[i]), bool(t_flags[i]),
                float(l_errors[i]), bool(l_flags[i]),
                int(class_ids[i]), float(confs[i]), all_probs[i],
            ))
    total        = len(results)
    anom_count   = sum(1 for r in results if r.is_anomaly)
    anom_rate    = anom_count / total if total > 0 else 0.0
    class_counts = {}
    for r in results:
        class_counts[r.cnn_class_id] = class_counts.get(r.cnn_class_id, 0) + 1
    anom_classes   = {k: v for k, v in class_counts.items() if k != 0}
    dominant_class = (SHORT_NAMES.get(max(anom_classes, key=anom_classes.get), "?")
                      if anom_classes else "N")
    return BatchResult(
        beats          = results,
        total_beats    = total,
        anomaly_count  = anom_count,
        anomaly_rate   = round(anom_rate, 4),
        class_counts   = class_counts,
        dominant_class = dominant_class,
        session_risk   = compute_session_risk(anom_rate, results),
        recording_sec  = round(float(timestamps[-1]), 2) if len(timestamps) > 0 else 0.0,
    )


def predict_ecg(
    raw_signal : np.ndarray,
    loader     : ModelLoader,
    fs         : float = 360.0,
    batch_size : int   = 64,
    verbose    : bool  = True,
) -> BatchResult:
    from .preprocess import preprocess_ecg
    beats, timestamps = preprocess_ecg(raw_signal, fs=fs, verbose=verbose)
    if len(beats) == 0:
        return BatchResult(
            beats=[], total_beats=0, anomaly_count=0, anomaly_rate=0.0,
            class_counts={}, dominant_class="N/A", session_risk="Low",
            recording_sec=round(len(raw_signal) / fs, 2),
        )
    result = predict_beats(beats, timestamps, loader, batch_size=batch_size)
    if verbose:
        print(f"\n{'='*50}\n{result.summary()}\n{'='*50}")
    return result
