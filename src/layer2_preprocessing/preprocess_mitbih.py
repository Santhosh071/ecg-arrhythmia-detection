import sys
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import wfdb
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from config.config import (
    MIT_BIH_PATH,
    MIT_BIH_PROCESSED_PATH,
    MIT_BIH_SAMPLE_RATE,
    BEAT_SIZE,
    CLASS_NAMES
)
from config.logger import get_logger
logger = get_logger(__name__)

SAMPLE_RATE   = MIT_BIH_SAMPLE_RATE
BEAT_SAMPLES  = BEAT_SIZE
BEFORE_PEAK   = 90
AFTER_PEAK    = BEAT_SAMPLES - BEFORE_PEAK

LABEL_MAP = {
    'N': 0, 'L': 1, 'R': 2, 'A': 3,
    'V': 4, '/': 5, 'E': 6, 'F': 7
}
RECORDS = [
    100, 101, 102, 103, 104, 105, 106, 107,
    108, 109, 111, 112, 113, 114, 115, 116,
    117, 118, 119, 121, 122, 123, 124, 200,
    201, 202, 203, 205, 207, 208, 209, 210,
    212, 213, 214, 215, 217, 219, 220, 221,
    222, 223, 228, 230, 231, 232, 233, 234
]

def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=SAMPLE_RATE, order=4):
    try:
        nyq    = 0.5 * fs
        low    = lowcut  / nyq
        high   = highcut / nyq
        b, a   = butter(order, [low, high], btype='band')
        filtered = filtfilt(b, a, signal)
        return filtered
    except Exception as e:
        logger.error(f"Bandpass filter failed: {e}")
        raise

def correct_baseline(signal):
    try:
        baseline = np.median(signal)
        return signal - baseline
    except Exception as e:
        logger.error(f"Baseline correction failed: {e}")
        raise

def normalize_beat(beat):
    try:
        mean = np.mean(beat)
        std  = np.std(beat)
        if std < 1e-6:
            return beat - mean
        return (beat - mean) / std
    except Exception as e:
        logger.error(f"Normalization failed: {e}")
        raise

def extract_beats(signal, r_peaks, annotations, valid_labels):
    beats  = []
    labels = []
    ann_samples = annotations.sample
    ann_symbols = annotations.symbol
    ann_dict = dict(zip(ann_samples, ann_symbols))

    for peak in r_peaks:
        start = peak - BEFORE_PEAK
        end   = peak + AFTER_PEAK
        if start < 0 or end > len(signal):
            continue
        symbol = ann_dict.get(peak, None)
        if symbol not in valid_labels:
            continue
        beat = signal[start:end]
        if len(beat) != BEAT_SAMPLES:
            continue
        beat = normalize_beat(beat)
        beats.append(beat)
        labels.append(LABEL_MAP[symbol])
    return np.array(beats, dtype=np.float32), np.array(labels, dtype=np.int64)

def process_record(record_id):
    try:
        record_path = str(MIT_BIH_PATH / str(record_id))
        record      = wfdb.rdrecord(record_path)
        signal      = record.p_signal[:, 0]
        annotation  = wfdb.rdann(record_path, 'atr')
        signal = bandpass_filter(signal)
        signal = correct_baseline(signal)
        beats, labels = extract_beats(
            signal,
            annotation.sample,
            annotation,
            set(LABEL_MAP.keys())
        )
        logger.info(f"Record {record_id}: {len(beats)} beats extracted")
        return beats, labels
    except Exception as e:
        logger.error(f"Record {record_id} failed: {e}")
        return np.array([]), np.array([])

def preprocess_mitbih():
    try:
        logger.info("=" * 60)
        logger.info("Starting MIT-BIH Preprocessing Pipeline")
        logger.info(f"Sample Rate : {SAMPLE_RATE} Hz")
        logger.info(f"Beat Size   : {BEAT_SAMPLES} samples")
        logger.info("=" * 60)
        all_beats  = []
        all_labels = []
        failed     = []
        for record_id in RECORDS:
            beats, labels = process_record(record_id)
            if len(beats) == 0:
                failed.append(record_id)
                continue
            all_beats.append(beats)
            all_labels.append(labels)
        all_beats  = np.vstack(all_beats)
        all_labels = np.hstack(all_labels)
        logger.info(f"Total beats extracted : {len(all_beats)}")
        logger.info(f"Beat shape            : {all_beats.shape}")
        class_counts = Counter(all_labels.tolist())
        logger.info("Class distribution:")
        for label_idx, count in sorted(class_counts.items()):
            class_name = CLASS_NAMES[label_idx]
            logger.info(f"  Class {label_idx} ({class_name}): {count} beats")
        X_train, X_temp, y_train, y_temp = train_test_split(
            all_beats, all_labels,
            test_size=0.30,
            random_state=42,
            stratify=all_labels
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=0.50,
            random_state=42,
            stratify=y_temp
        )
        logger.info(f"Train : {len(X_train)} beats")
        logger.info(f"Val   : {len(X_val)}   beats")
        logger.info(f"Test  : {len(X_test)}  beats")
        save_splits(X_train, y_train, X_val, y_val, X_test, y_test)
        if failed:
            logger.warning(f"Failed records: {failed}")
        logger.info("MIT-BIH Preprocessing Complete!")
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise

def save_splits(X_train, y_train, X_val, y_val, X_test, y_test):
    try:
        train_dir = MIT_BIH_PROCESSED_PATH / "train"
        val_dir   = MIT_BIH_PROCESSED_PATH / "val"
        test_dir  = MIT_BIH_PROCESSED_PATH / "test"
        for d in [train_dir, val_dir, test_dir]:
            d.mkdir(parents=True, exist_ok=True)
        np.save(train_dir / "beats.npy",  X_train)
        np.save(train_dir / "labels.npy", y_train)
        np.save(val_dir   / "beats.npy",  X_val)
        np.save(val_dir   / "labels.npy", y_val)
        np.save(test_dir  / "beats.npy",  X_test)
        np.save(test_dir  / "labels.npy", y_test)
        logger.info(f"Saved train → {train_dir}")
        logger.info(f"Saved val   → {val_dir}")
        logger.info(f"Saved test  → {test_dir}")
        verify_saved_files(train_dir, val_dir, test_dir)
    except Exception as e:
        logger.error(f"Saving failed: {e}")
        raise

def verify_saved_files(train_dir, val_dir, test_dir):
    try:
        for split, d in [("train", train_dir), ("val", val_dir), ("test", test_dir)]:
            beats  = np.load(d / "beats.npy")
            labels = np.load(d / "labels.npy")
            logger.info(f"{split}: beats={beats.shape}, labels={labels.shape}")
        logger.info("All files verified!")
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        raise

if __name__ == "__main__":
    preprocess_mitbih()