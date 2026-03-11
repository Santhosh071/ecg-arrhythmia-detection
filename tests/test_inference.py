import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, r"C:\ecg_arrhythmia")

from src.layer3_models.inference import (
    ModelLoader,
    predict_beat,
    predict_beats,
    predict_ecg,
    check_signal_quality,
    CLASS_NAMES,
    BEAT_LENGTH,
    MIT_BIH_FS,
)

MODELS_DIR = Path(r"D:/ecg_project/models")
DATA_DIR   = Path(r"D:/ecg_project/datasets\mitbih\processed")
SEP  = "=" * 55
SEP2 = "-" * 55

def _pass(name: str):
    print(f"  PASS  {name}")

def _fail(name: str, err: Exception):
    print(f"  FAIL  {name} — {err}")

def test_model_loading() -> ModelLoader:
    print(f"\n{SEP}\nTEST 1 — Model Loading\n{SEP2}")
    loader = ModelLoader(str(MODELS_DIR))
    loader.load_all()
    assert loader.is_loaded
    assert loader.cnn            is not None
    assert loader.transformer_ae is not None
    assert loader.lstm_ae        is not None
    assert loader.trans_threshold is not None
    assert loader.lstm_threshold  is not None

    for k, v in loader.status().items():
        print(f"  {k:<25}: {v}")
    _pass("model_loading")
    return loader

def test_signal_quality():
    print(f"\n{SEP}\nTEST 2 — Signal Quality Check\n{SEP2}")
    t    = np.linspace(0, 10, 10 * MIT_BIH_FS)
    good = np.sin(2 * np.pi * 1.2 * t) * 0.5
    flat = np.zeros(MIT_BIH_FS * 5)
    q_good = check_signal_quality(good)
    q_flat = check_signal_quality(flat)
    assert not q_flat["is_good"],      "flat line should fail"
    assert len(q_flat["issues"]) > 0,  "flat line should have issues"
    print(f"  Good signal — is_good={q_good['is_good']} | snr={q_good['snr_db']} dB")
    print(f"  Flat signal — is_good={q_flat['is_good']} | issues={q_flat['issues']}")
    _pass("signal_quality")

def test_data_loading() -> tuple:
    print(f"\n{SEP}\nTEST 3 — Load Test Beats\n{SEP2}")
    test_beats_path  = DATA_DIR / "test" / "beats.npy"
    test_labels_path = DATA_DIR / "test" / "labels.npy"
    if not test_beats_path.exists():
        raise FileNotFoundError(
            f"Test beats not found at {test_beats_path}\n"
            f"Check DATA_DIR path in this file."
        )
    beats  = np.load(test_beats_path)
    labels = np.load(test_labels_path)
    assert beats.shape[1] == BEAT_LENGTH, f"Expected {BEAT_LENGTH} samples, got {beats.shape[1]}"
    assert len(beats) == len(labels)
    print(f"  Beats shape : {beats.shape}")
    print(f"  Classes     : {sorted(np.unique(labels).tolist())}")
    for cls in sorted(np.unique(labels)):
        print(f"    Class {cls} ({CLASS_NAMES[cls]:<40}): {(labels==cls).sum():>5}")
    _pass("data_loading")
    return beats, labels

def test_single_beat(loader: ModelLoader, beats: np.ndarray):
    print(f"\n{SEP}\nTEST 4 — Single Beat Prediction\n{SEP2}")
    result = predict_beat(beats[0], loader, beat_index=0, timestamp=0.0)
    assert 0 <= result.cnn_class_id <= 7
    assert 0.0 <= result.cnn_confidence <= 1.0
    assert len(result.cnn_all_probs) == 8
    assert abs(sum(result.cnn_all_probs) - 1.0) < 0.01
    assert result.risk_level   in ("Low", "Medium", "High", "Critical")
    assert result.alert_color  in ("green", "yellow", "orange", "red")
    assert isinstance(result.to_dict(), dict)
    print(f"  {result.summary_line()}")
    print(f"  All probs: {result.cnn_all_probs}")
    _pass("single_beat")

def test_batch_prediction(loader: ModelLoader, beats: np.ndarray):
    print(f"\n{SEP}\nTEST 5 — Batch Prediction (100 beats)\n{SEP2}")
    n          = min(100, len(beats))
    timestamps = np.arange(n, dtype=float) * (1.0 / MIT_BIH_FS * BEAT_LENGTH)
    result     = predict_beats(beats[:n], timestamps, loader, batch_size=32)
    assert result.total_beats   == n
    assert 0.0 <= result.anomaly_rate <= 1.0
    assert result.session_risk  in ("Low", "Medium", "High", "Critical")
    print(f"  {result.summary()}")
    print(f"\n  Class distribution:")
    for cid, cnt in sorted(result.class_counts.items()):
        print(f"    {CLASS_NAMES[cid]:<45}: {cnt}")
    _pass("batch_prediction")
    return result

def test_all_class_types(loader: ModelLoader, beats: np.ndarray, labels: np.ndarray):
    print(f"\n{SEP}\nTEST 6 — One Beat Per Class\n{SEP2}")
    print(f"  {'True Class':<45} {'CNN':>5} {'Conf':>7} {'Anomaly':>8} {'Risk':>8}")
    print(f"  {SEP2}")
    for cls in range(8):
        idx = np.where(labels == cls)[0]
        if len(idx) == 0:
            print(f"  {CLASS_NAMES[cls]:<45} {'N/A':>5}")
            continue
        r = predict_beat(beats[idx[0]], loader, beat_index=int(idx[0]))
        flag = "YES" if r.is_anomaly else "no"
        print(f"  {CLASS_NAMES[cls]:<45} {r.cnn_short_name:>5} "
              f"{r.cnn_confidence*100:>6.1f}% {flag:>8} {r.risk_level:>8}")
    _pass("all_class_types")

def test_to_dict_serialisation(loader: ModelLoader, beats: np.ndarray):
    print(f"\n{SEP}\nTEST 7 — Serialisation (to_dict)\n{SEP2}")
    import json
    n          = min(20, len(beats))
    timestamps = np.arange(n, dtype=float) * 0.8
    batch      = predict_beats(beats[:n], timestamps, loader)
    json_str = json.dumps(batch.to_dict())
    restored = json.loads(json_str)
    assert restored["total_beats"] == n
    assert len(restored["beats"])  == n
    assert "cnn_class_id" in restored["beats"][0]
    print(f"  Serialised {n} beats → {len(json_str)} bytes JSON")
    _pass("serialisation")

if __name__ == "__main__":
    print(f"\n{SEP}")
    print("ECG Inference Pipeline — Phase 5 Tests")
    print(SEP)
    passed = 0
    failed = 0
    try:
        loader = test_model_loading()
        passed += 1
    except Exception as e:
        print(f"\nCRITICAL: Model loading failed — {e}")
        print(f"Fix:\n  1. Check MODELS_DIR = {MODELS_DIR}")
        print(f"  2. cnn.py / transformer_ae.py / lstm_ae.py in C:/ecg_arrhythmia/src\\layer3_models\\")
        sys.exit(1)
    try:
        test_signal_quality()
        passed += 1
    except Exception as e:
        _fail("signal_quality", e); failed += 1

    try:
        beats, labels = test_data_loading()
        passed += 1
    except Exception as e:
        _fail("data_loading", e); failed += 1
        print(f"\nSkipping tests 4-7 (no data).")
        print(f"\n{SEP}\n{passed} passed / {passed+failed} total\n{SEP}")
        sys.exit(1)

    for name, fn, args in [
        ("single_beat",        test_single_beat,          (loader, beats)),
        ("batch_prediction",   test_batch_prediction,     (loader, beats)),
        ("all_class_types",    test_all_class_types,      (loader, beats, labels)),
        ("serialisation",      test_to_dict_serialisation,(loader, beats)),
    ]:
        try:
            fn(*args)
            passed += 1
        except Exception as e:
            _fail(name, e); failed += 1

    print(f"\n{SEP}")
    total = passed + failed
    print(f"Results : {passed}/{total} passed")
    print(SEP)