from __future__ import annotations

from src.layer1_ingestion.iot.beat_segmenter import segment_live_signal
from src.layer1_ingestion.iot.signal_preprocess import prepare_stream_for_model
from src.layer3_models.inference import check_signal_quality, predict_ecg


def analyze_live_stream(
    *,
    samples: list[float],
    sampling_rate: float,
    loader,
    adc_resolution: int = 4095,
    vref: float = 3.3,
) -> dict:
    conditioned_signal, model_fs = prepare_stream_for_model(
        samples,
        sampling_rate=sampling_rate,
        adc_resolution=adc_resolution,
        vref=vref,
    )
    quality = check_signal_quality(conditioned_signal, fs=model_fs)
    beats, timestamps, peaks = segment_live_signal(conditioned_signal, fs=model_fs)
    batch = predict_ecg(conditioned_signal, loader, fs=model_fs, batch_size=64, verbose=False)
    return {
        "raw_signal": conditioned_signal,
        "quality": quality,
        "beats": beats,
        "timestamps": timestamps,
        "peaks": peaks,
        "batch": batch,
        "model_sampling_rate": model_fs,
    }
