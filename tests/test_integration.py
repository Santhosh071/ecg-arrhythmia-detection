import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, r"C:\ecg_arrhythmia")
SEP  = "=" * 55
SEP2 = "-" * 55
DATA_DIR   = Path(r"D:\ecg_project\datasets\mitbih\processed")
MODELS_DIR = Path(r"D:\ecg_project\models")
DB_PATH    = r"D:\ecg_project\database\patient_history.db"
API_BASE   = "http://127.0.0.1:8000"
TEST_PID   = "INTEGRATION_TEST_001"

def _pass(name): print(f"  PASS  {name}")
def _fail(name, e): print(f"  FAIL  {name} — {e}")
def _skip(name, reason): print(f"  SKIP  {name} — {reason}")

def test_data_loading() -> tuple:
    print(f"\n{SEP}\nTEST 1 — Data Loading\n{SEP2}")
    beats_path  = DATA_DIR / "test" / "beats.npy"
    labels_path = DATA_DIR / "test" / "labels.npy"
    assert beats_path.exists(),  f"Not found: {beats_path}"
    assert labels_path.exists(), f"Not found: {labels_path}"
    beats  = np.load(str(beats_path))
    labels = np.load(str(labels_path))
    assert beats.shape[1] == 187
    assert len(beats) == len(labels)
    assert set(labels).issubset({0, 1, 2, 3, 4, 5, 6, 7})
    print(f"  Beats loaded   : {beats.shape}")
    print(f"  Label classes  : {sorted(np.unique(labels).tolist())}")
    _pass("data_loading")
    return beats, labels

def test_preprocessing() -> tuple:
    print(f"\n{SEP}\nTEST 2 — Preprocessing Pipeline\n{SEP2}")
    from src.layer3_models.inference.preprocess import (
        bandpass_filter, baseline_correction,
        detect_r_peaks, segment_beats, normalise_beats,
        preprocess_ecg, check_signal_quality, BEAT_LENGTH, MIT_BIH_FS,
    )
    t   = np.linspace(0, 10, 10 * MIT_BIH_FS)
    raw = (np.sin(2 * np.pi * 1.2 * t) * 0.6
           + np.sin(2 * np.pi * 5.0 * t) * 0.1
           + np.random.randn(len(t)) * 0.05)
    filtered  = bandpass_filter(raw)
    corrected = baseline_correction(filtered)
    r_peaks   = detect_r_peaks(corrected)
    assert len(r_peaks) > 0, "No R-peaks found"
    beats, valid = segment_beats(corrected, r_peaks)
    assert beats.shape[1] == BEAT_LENGTH
    norm = normalise_beats(beats)
    assert abs(norm[0].mean()) < 0.1
    assert 0.8 < norm[0].std() < 1.2
    out_beats, out_ts = preprocess_ecg(raw, verbose=False)
    assert len(out_beats) > 0
    assert out_beats.shape[1] == BEAT_LENGTH
    quality = check_signal_quality(raw)
    assert "is_good"      in quality
    assert "snr_db"       in quality
    assert "mean_hr_bpm"  in quality
    print(f"  R-peaks found  : {len(r_peaks)}")
    print(f"  Beats extracted: {len(out_beats)}")
    print(f"  Beat shape     : {out_beats.shape}")
    print(f"  SNR            : {quality['snr_db']} dB")
    print(f"  HR estimate    : {quality['mean_hr_bpm']} bpm")
    _pass("preprocessing")
    return out_beats, out_ts

def test_inference(beats: np.ndarray, labels: np.ndarray) -> object:
    print(f"\n{SEP}\nTEST 3 — Inference Pipeline\n{SEP2}")
    from src.layer3_models.inference import (
        ModelLoader, predict_beats, predict_beat,
        BeatResult, BatchResult, BEAT_LENGTH,
    )
    loader = ModelLoader(str(MODELS_DIR))
    loader.load_all()
    n          = min(100, len(beats))
    timestamps = np.arange(n, dtype=float) * (187 / 360)
    batch      = predict_beats(beats[:n], timestamps, loader, batch_size=32)
    assert batch.total_beats    == n
    assert 0 <= batch.anomaly_rate <= 1.0
    assert batch.session_risk   in ("Low", "Medium", "High", "Critical")
    assert len(batch.beats)     == n
    single = predict_beat(beats[0], loader)
    assert 0 <= single.cnn_class_id <= 7
    assert 0 <= single.cnn_confidence <= 1.0
    assert isinstance(single.to_dict(), dict)
    predicted = {r.cnn_class_id for r in batch.beats}
    print(f"  Beats predicted : {batch.total_beats}")
    print(f"  Anomalies       : {batch.anomaly_count} ({batch.anomaly_rate*100:.1f}%)")
    print(f"  Session risk    : {batch.session_risk}")
    print(f"  Dominant class  : {batch.dominant_class}")
    print(f"  Classes seen    : {sorted(predicted)}")
    print(f"  Critical beats  : {len(batch.critical_beats())}")
    _pass("inference")
    return batch, loader

def test_agent(batch) -> object:
    print(f"\n{SEP}\nTEST 4 — Agent Pipeline\n{SEP2}")
    from src.layer4_agent import ECGAgent, ECGDatabase, check_llm_status
    from src.layer4_agent.agent_tools import init_tools, ALL_TOOLS
    llm_status = check_llm_status()
    db         = ECGDatabase(DB_PATH)
    session_id = db.save_session(TEST_PID, batch)
    assert session_id > 0
    print(f"  Session saved   : id={session_id}")
    init_tools(db, None)
    risk_out = ALL_TOOLS[1].run(json.dumps({
        "anomaly_rate"  : batch.anomaly_rate,
        "dominant_class": batch.dominant_class,
        "critical_beats": len(batch.critical_beats()),
        "total_beats"   : batch.total_beats,
    }))
    risk = json.loads(risk_out)
    assert risk["risk_level"] in ("Low", "Medium", "High", "Critical")
    print(f"  Risk assessed   : {risk['risk_level']} — {risk['reason']}")
    alert_out = ALL_TOOLS[3].run(json.dumps({
        "patient_id": TEST_PID,
        "risk_level": risk["risk_level"],
        "message"   : f"{batch.anomaly_count} anomalies detected",
        "class_name": batch.dominant_class,
    }))
    assert "ALERT" in alert_out
    print(f"  Alert fired     : {alert_out[:60]}")
    hist_out = ALL_TOOLS[6].run(json.dumps({"patient_id": TEST_PID, "limit": 3}))
    assert isinstance(hist_out, str)
    print(f"  History checked : {hist_out[:60]}")
    if llm_status["available"]:
        agent = ECGAgent(DB_PATH)
        agent.start()
        response = agent.run(
            f"Patient has {batch.anomaly_rate*100:.1f}% anomaly rate "
            f"with dominant {batch.dominant_class} class. "
            f"Briefly assess clinical significance in 2 sentences.",
            patient_id=TEST_PID,
        )
        assert len(response) > 20
        print(f"  Agent response  : {response[:100]}...")
        agent.reset_memory()
    else:
        _skip("agent_llm", "Groq not available")
    report_out = ALL_TOOLS[4].run(json.dumps({
        "patient_id"  : TEST_PID,
        "session_data": {
            "session_risk"  : batch.session_risk,
            "total_beats"   : batch.total_beats,
            "anomaly_count" : batch.anomaly_count,
            "anomaly_rate"  : batch.anomaly_rate,
            "recording_sec" : batch.recording_sec,
            "dominant_class": batch.dominant_class,
        }
    }))
    assert "report_" in report_out.lower() or "saved" in report_out.lower()
    print(f"  PDF report      : {report_out[:60]}")
    _pass("agent_pipeline")
    return db

def test_database(db, batch) -> None:
    print(f"\n{SEP}\nTEST 5 — Database Persistence\n{SEP2}")
    history = db.get_patient_history(TEST_PID, limit=10)
    assert len(history) >= 1
    alerts  = db.get_recent_alerts(TEST_PID, limit=10)
    assert len(alerts) >= 1
    trend   = db.get_anomaly_trend(TEST_PID, last_n_sessions=5)
    assert trend["trend"]    in ("improving", "worsening", "stable", "no_data")
    assert trend["sessions"] >= 1
    print(f"  Sessions stored : {len(history)}")
    print(f"  Alerts stored   : {len(alerts)}")
    print(f"  Trend           : {trend['trend']} | avg={trend['avg_anomaly_rate']}%")
    print(f"  Latest session  : risk={history[0]['risk_level']} | "
          f"anomalies={history[0]['anomaly_count']}")
    _pass("database_persistence")

def test_api_endpoints(beats: np.ndarray, run_api: bool) -> None:
    print(f"\n{SEP}\nTEST 6 — API Endpoints\n{SEP2}")
    if not run_api:
        _skip("api_endpoints", "--no-api flag set")
        return
    import urllib.request
    import urllib.error
    try:
        urllib.request.urlopen(f"{API_BASE}/health", timeout=5)
    except Exception:
        _skip("api_endpoints", f"API not running at {API_BASE}")
        return
    import httpx
    client = httpx.Client(base_url=API_BASE, timeout=30)

    # GET /health
    r = client.get("/health")
    assert r.status_code == 200
    health = r.json()
    assert health["status"] == "ok"
    print(f"  GET /health     : {health}")

    # GET /
    r = client.get("/")
    assert r.status_code == 200
    print(f"  GET /           : {r.json()['name']}")

    # POST /predict
    n          = min(10, len(beats))
    timestamps = (np.arange(n, dtype=float) * (187 / 360)).tolist()
    payload    = {
        "patient_id"  : TEST_PID,
        "beats"       : beats[:n].tolist(),
        "timestamps"  : timestamps,
        "fs"          : 360.0,
        "save_session": True,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200, f"/predict failed: {r.text}"
    result = r.json()
    assert result["total_beats"]   == n
    assert result["session_risk"]  in ("Low", "Medium", "High", "Critical")
    print(f"  POST /predict   : {n} beats | risk={result['session_risk']} "
          f"| anomalies={result['anomaly_count']}")

    # POST /predict/single
    single_payload = {
        "patient_id": TEST_PID,
        "beat"      : beats[0].tolist(),
        "beat_index": 0,
        "timestamp" : 0.0,
    }
    r = client.post("/predict/single", json=single_payload)
    assert r.status_code == 200, f"/predict/single failed: {r.text}"
    single = r.json()
    assert 0 <= single["cnn_class_id"] <= 7
    print(f"  POST /predict/single : class={single['cnn_short_name']} "
          f"conf={single['cnn_confidence']*100:.1f}%")

    # POST /signal/quality
    t      = np.linspace(0, 5, 5 * 360)
    signal = (np.sin(2 * np.pi * 1.2 * t) * 0.5).tolist()
    r = client.post("/signal/quality", json={
        "patient_id": TEST_PID, "signal": signal, "fs": 360.0
    })
    assert r.status_code == 200
    quality = r.json()
    print(f"  POST /signal/quality : is_good={quality['is_good']} "
          f"snr={quality['snr_db']} dB")

    # GET /history/{patient_id}
    r = client.get(f"/history/{TEST_PID}")
    assert r.status_code == 200
    hist = r.json()
    assert hist["total"] >= 1
    print(f"  GET /history    : {hist['total']} session(s)")

    # GET /history/{patient_id}/alerts
    r = client.get(f"/history/{TEST_PID}/alerts")
    assert r.status_code == 200
    print(f"  GET /alerts     : {r.json()['total']} alert(s)")

    # GET /history/{patient_id}/trend
    r = client.get(f"/history/{TEST_PID}/trend")
    assert r.status_code == 200
    trend = r.json()
    print(f"  GET /trend      : {trend['trend']} | avg={trend['avg_anomaly_rate']}%")

    # POST /agent/query (only if agent ready)
    if health.get("agent_ready"):
        r = client.post("/agent/query", json={
            "question"  : "What is the clinical significance of a PVC?",
            "patient_id": TEST_PID,
        }, timeout=30)
        assert r.status_code == 200
        answer = r.json()["answer"]
        assert len(answer) > 20
        print(f"  POST /agent/query : {answer[:80]}...")
    else:
        _skip("api_agent_query", "agent not ready")

    client.close()
    _pass("api_endpoints")

def test_pipeline_timing(beats: np.ndarray, loader) -> None:
    print(f"\n{SEP}\nTEST 7 — Pipeline Performance\n{SEP2}")
    from src.layer3_models.inference import predict_beats
    for n in [10, 50, 100, 200]:
        if n > len(beats):
            continue
        timestamps = np.arange(n, dtype=float) * (187 / 360)
        t0 = time.perf_counter()
        predict_beats(beats[:n], timestamps, loader, batch_size=64)
        elapsed = (time.perf_counter() - t0) * 1000
        per_beat = elapsed / n
        print(f"  {n:4d} beats → {elapsed:7.1f} ms total | {per_beat:.2f} ms/beat")
    _pass("pipeline_timing")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-api", action="store_true",
                        help="Skip API tests (run without uvicorn)")
    args = parser.parse_args()
    print(f"\n{SEP}")
    print("ECG System — Phase 9 Integration Tests")
    print(SEP)
    passed = failed = 0
    # Test 1 — Data
    try:
        beats, labels = test_data_loading()
        passed += 1
    except Exception as e:
        _fail("data_loading", e); failed += 1
        print("CRITICAL: Cannot continue without data.")
        sys.exit(1)
    # Test 2 — Preprocessing
    try:
        test_preprocessing()
        passed += 1
    except Exception as e:
        _fail("preprocessing", e); failed += 1
    # Test 3 — Inference
    try:
        batch, loader = test_inference(beats, labels)
        passed += 1
    except Exception as e:
        _fail("inference", e); failed += 1
        print("CRITICAL: Cannot continue without inference.")
        sys.exit(1)
    # Test 4 — Agent
    try:
        db = test_agent(batch)
        passed += 1
    except Exception as e:
        _fail("agent_pipeline", e); failed += 1
        from src.layer4_agent import ECGDatabase
        db = ECGDatabase(DB_PATH)
    # Test 5 — Database
    try:
        test_database(db, batch)
        passed += 1
    except Exception as e:
        _fail("database_persistence", e); failed += 1
    # Test 6 — API
    try:
        test_api_endpoints(beats, run_api=not args.no_api)
        passed += 1
    except Exception as e:
        _fail("api_endpoints", e); failed += 1
    # Test 7 — Timing
    try:
        test_pipeline_timing(beats, loader)
        passed += 1
    except Exception as e:
        _fail("pipeline_timing", e); failed += 1
    # Summary
    print(f"\n{SEP}")
    print(f"Results : {passed}/{passed+failed} passed")
    if failed == 0:
        print("Phase 9 complete — Full pipeline integrated.")
    else:
        print(f"{failed} test(s) failed — check errors above.")
    print(SEP)