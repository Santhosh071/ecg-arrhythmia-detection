import warnings
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, medfilt
warnings.filterwarnings("ignore")

BEAT_LENGTH    = 187
MIT_BIH_FS     = 360
PRE_R_SAMPLES  = 90
POST_R_SAMPLES = BEAT_LENGTH - PRE_R_SAMPLES

def bandpass_filter(signal: np.ndarray, fs: float = MIT_BIH_FS) -> np.ndarray:
    try:
        nyq  = 0.5 * fs
        low  = np.clip(0.5  / nyq, 1e-6, 0.999)
        high = np.clip(40.0 / nyq, 1e-6, 0.999)
        b, a = butter(4, [low, high], btype="band")
        return filtfilt(b, a, signal.astype(np.float64))
    except Exception:
        return signal.astype(np.float64)

def baseline_correction(signal: np.ndarray, fs: float = MIT_BIH_FS) -> np.ndarray:
    try:
        win = int(0.2 * fs)
        win = win if win % 2 == 1 else win + 1
        return signal - medfilt(signal, kernel_size=win)
    except Exception:
        return signal

def detect_r_peaks(signal: np.ndarray, fs: float = MIT_BIH_FS) -> np.ndarray:
    try:
        squared    = np.diff(signal.astype(np.float64)) ** 2
        win        = max(1, int(0.15 * fs))
        integrated = np.convolve(squared, np.ones(win) / win, mode="same")
        threshold  = 0.30 * integrated.max()
        min_dist   = max(1, int(0.20 * fs))
        peaks, _   = find_peaks(integrated, height=threshold, distance=min_dist)
        return np.clip(peaks + 1, 0, len(signal) - 1)
    except Exception:
        return np.array([], dtype=int)

def segment_beats(signal: np.ndarray, r_peaks: np.ndarray) -> tuple:
    beats, valid = [], []
    for peak in r_peaks:
        start = int(peak) - PRE_R_SAMPLES
        end   = int(peak) + POST_R_SAMPLES
        if start < 0 or end > len(signal):
            continue
        beat = signal[start:end]
        if len(beat) == BEAT_LENGTH:
            beats.append(beat)
            valid.append(peak)
    if not beats:
        return np.empty((0, BEAT_LENGTH), dtype=np.float32), np.array([], dtype=int)
    return np.array(beats, dtype=np.float32), np.array(valid, dtype=int)

def normalise_beats(beats: np.ndarray) -> np.ndarray:
    mean = beats.mean(axis=1, keepdims=True)
    std  = np.where(beats.std(axis=1, keepdims=True) < 1e-8, 1e-8,
                    beats.std(axis=1, keepdims=True))
    return ((beats - mean) / std).astype(np.float32)

def preprocess_ecg(raw_signal: np.ndarray, fs: float = MIT_BIH_FS,
                   return_peaks: bool = False, verbose: bool = True) -> tuple:
    raw_signal = np.asarray(raw_signal, dtype=np.float64).ravel()
    if len(raw_signal) < BEAT_LENGTH:
        raise ValueError(f"Signal too short: {len(raw_signal)} samples (min {BEAT_LENGTH}).")
    corrected  = baseline_correction(bandpass_filter(raw_signal, fs), fs)
    r_peaks    = detect_r_peaks(corrected, fs)
    empty      = np.empty((0, BEAT_LENGTH), dtype=np.float32)
    if len(r_peaks) == 0:
        if verbose:
            print("[preprocess] No R-peaks detected.")
        return (empty, np.array([]), r_peaks) if return_peaks else (empty, np.array([]))
    beats, valid_peaks = segment_beats(corrected, r_peaks)
    if len(beats) == 0:
        if verbose:
            print("[preprocess] No valid beats after segmentation.")
        return (empty, np.array([]), valid_peaks) if return_peaks else (empty, np.array([]))
    beats      = normalise_beats(beats)
    timestamps = valid_peaks.astype(np.float64) / fs
    if verbose:
        print(f"[preprocess] {len(beats)} beats | {len(raw_signal)/fs:.1f}s | {fs}Hz")
    return (beats, timestamps, valid_peaks) if return_peaks else (beats, timestamps)

def preprocess_single_beat(beat: np.ndarray) -> np.ndarray:
    beat = np.asarray(beat, dtype=np.float32).ravel()
    if len(beat) != BEAT_LENGTH:
        raise ValueError(f"Beat must be {BEAT_LENGTH} samples. Got {len(beat)}.")
    std = beat.std()
    return ((beat - beat.mean()) / (std if std > 1e-8 else 1e-8)).astype(np.float32)

def check_signal_quality(raw_signal: np.ndarray, fs: float = MIT_BIH_FS) -> dict:
    raw_signal = np.asarray(raw_signal, dtype=np.float64).ravel()
    issues     = []
    peak_count = 0
    mean_hr    = 0.0
    snr_db     = 0.0
    duration   = len(raw_signal) / fs
    if duration < 2.0:
        issues.append(f"Signal too short: {duration:.1f}s")
    amp = float(raw_signal.max() - raw_signal.min())
    if amp < 0.01:
        issues.append("Flat-line — check electrode connections")
    elif amp > 10.0:
        issues.append(f"Signal clipped — range {amp:.1f}")
    if not np.isfinite(raw_signal).all():
        issues.append("NaN/Inf values in signal")
    try:
        filtered   = bandpass_filter(raw_signal, fs)
        r_peaks    = detect_r_peaks(filtered, fs)
        peak_count = len(r_peaks)
        if peak_count < 2:
            issues.append(f"Too few R-peaks: {peak_count}")
        else:
            mean_hr = float(60.0 / (np.diff(r_peaks).astype(float) / fs).mean())
            if mean_hr < 30:
                issues.append(f"Low HR: {mean_hr:.0f} bpm")
            if mean_hr > 250:
                issues.append(f"High HR: {mean_hr:.0f} bpm")
        noise_power = float(np.var(raw_signal - filtered)) + 1e-12
        snr_db      = float(10 * np.log10(np.var(filtered) / noise_power))
        if snr_db < 5.0:
            issues.append(f"Low SNR: {snr_db:.1f} dB")
    except Exception as e:
        issues.append(f"Analysis error: {e}")
    return {
        "is_good"     : len(issues) == 0,
        "issues"      : issues,
        "snr_db"      : round(snr_db,   2),
        "duration_sec": round(duration, 2),
        "peak_count"  : peak_count,
        "mean_hr_bpm" : round(mean_hr,  1),
    }