from __future__ import annotations

import numpy as np

from src.layer3_models.inference.preprocess import preprocess_ecg


def segment_live_signal(raw_signal: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    beats, timestamps, peaks = preprocess_ecg(raw_signal, fs=fs, return_peaks=True, verbose=False)
    return beats, timestamps, peaks
