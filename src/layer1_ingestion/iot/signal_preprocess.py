from __future__ import annotations

import numpy as np

from src.layer3_models.inference.preprocess import MIT_BIH_FS


def center_adc_samples(samples: np.ndarray, adc_resolution: int = 4095, vref: float = 3.3) -> np.ndarray:
    signal = np.asarray(samples, dtype=np.float64).ravel()
    if adc_resolution <= 0:
        return signal

    midpoint = adc_resolution / 2.0
    volts = ((signal - midpoint) / adc_resolution) * vref
    return volts.astype(np.float64)


def normalise_stream(samples: np.ndarray) -> np.ndarray:
    signal = np.asarray(samples, dtype=np.float64).ravel()
    signal = signal - np.median(signal)
    peak = np.max(np.abs(signal)) if signal.size else 0.0
    if peak < 1e-8:
        return signal
    return signal / peak


def resample_signal(signal: np.ndarray, from_fs: float, to_fs: float = MIT_BIH_FS) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64).ravel()
    if signal.size == 0 or from_fs <= 0 or abs(from_fs - to_fs) < 1e-6:
        return signal

    duration = signal.size / from_fs
    new_size = max(1, int(round(duration * to_fs)))
    old_t = np.linspace(0.0, duration, num=signal.size, endpoint=False)
    new_t = np.linspace(0.0, duration, num=new_size, endpoint=False)
    return np.interp(new_t, old_t, signal).astype(np.float64)


def prepare_stream_for_display(samples: list[float], adc_resolution: int = 4095, vref: float = 3.3) -> np.ndarray:
    centered = center_adc_samples(np.asarray(samples, dtype=np.float64), adc_resolution=adc_resolution, vref=vref)
    return normalise_stream(centered)


def prepare_stream_for_model(
    samples: list[float],
    sampling_rate: float,
    adc_resolution: int = 4095,
    vref: float = 3.3,
) -> tuple[np.ndarray, float]:
    signal = prepare_stream_for_display(samples, adc_resolution=adc_resolution, vref=vref)
    signal = resample_signal(signal, from_fs=sampling_rate, to_fs=MIT_BIH_FS)
    return signal.astype(np.float64), float(MIT_BIH_FS)
