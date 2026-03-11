import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, r"C:\ecg_arrhythmia")
from src.layer4_agent.llm_config  import check_llm_status, is_internet_available
from src.layer4_agent.memory      import ECGDatabase, get_short_term_memory
from src.layer4_agent.agent_tools import init_tools, ALL_TOOLS
from src.layer4_agent.agent       import ECGAgent

SEP  = "=" * 55
SEP2 = "-" * 55
TEST_DB = "D:/ecg_project/database/test_agent.db"

def _pass(name): print(f"  PASS  {name}")
def _fail(name, e): print(f"  FAIL  {name} — {e}")

def test_llm_config():
    print(f"\n{SEP}\nTEST 1 — LLM Config & Internet Check\n{SEP2}")
    status = check_llm_status()
    for k, v in status.items():
        print(f"  {k:<18}: {v}")
    assert "available"  in status
    assert "internet"   in status
    assert "has_key"    in status
    assert status["model"] == "llama-3.3-70b-versatile"
    if not status["has_key"]:
        print("  WARNING: GROQ_API_KEY not set — LLM tests will be skipped")
    if not status["internet"]:
        print("  WARNING: No internet — LLM tests will be skipped")
    _pass("llm_config")
    return status

def test_database():
    print(f"\n{SEP}\nTEST 2 — SQLite Database\n{SEP2}")
    db = ECGDatabase(TEST_DB)
    class MockBatch:
        total_beats    = 200
        anomaly_count  = 30
        anomaly_rate   = 0.15
        dominant_class = "V"
        session_risk   = "High"
        recording_sec  = 160.0
        class_counts   = {0: 170, 4: 30}
    session_id = db.save_session("patient_test_001", MockBatch())
    assert session_id > 0
    print(f"  Session saved: id={session_id}")
    alert_id = db.save_alert("patient_test_001", "High", "orange", "3 PVCs detected", 42, "V")
    assert alert_id > 0
    print(f"  Alert saved: id={alert_id}")
    db.log_tool_call("test_tool", "test_input", "test_output", "patient_test_001")
    print(f"  Tool call logged")
    history = db.get_patient_history("patient_test_001", limit=5)
    assert len(history) >= 1
    assert history[0]["risk_level"] == "High"
    print(f"  History retrieved: {len(history)} session(s)")
    print(f"  Latest: {history[0]}")
    alerts = db.get_recent_alerts("patient_test_001", limit=5)
    assert len(alerts) >= 1
    print(f"  Alerts retrieved: {len(alerts)}")
    trend = db.get_anomaly_trend("patient_test_001", last_n_sessions=5)
    assert "trend" in trend
    print(f"  Trend: {trend['trend']} | avg={trend['avg_anomaly_rate']}%")
    _pass("database")
    return db

def test_memory():
    print(f"\n{SEP}\nTEST 3 — Short-term Memory\n{SEP2}")
    mem = get_short_term_memory(k=10)
    assert mem is not None
    assert mem.k == 10
    print(f"  Memory window k={mem.k}")
    mem.clear()
    print(f"  Memory cleared OK")
    _pass("memory")

def test_tools_no_llm(db):
    print(f"\n{SEP}\nTEST 4 — Tools (no LLM required)\n{SEP2}")
    init_tools(db, None)
    out = ALL_TOOLS[0].run(json.dumps({"patient_id": "patient_test_001", "last_n_sessions": 3}))
    print(f"  monitor_trends    : {out[:80]}")
    assert isinstance(out, str)
    out = ALL_TOOLS[1].run(json.dumps({
        "anomaly_rate": 0.15, "dominant_class": "V",
        "critical_beats": 0, "total_beats": 200
    }))
    result = json.loads(out)
    assert result["risk_level"] in ("Low", "Medium", "High", "Critical")
    print(f"  assess_risk       : risk={result['risk_level']} | color={result['color']}")
    out = ALL_TOOLS[3].run(json.dumps({
        "patient_id": "patient_test_001",
        "risk_level": "High",
        "message": "3 PVCs in last 60 seconds",
        "beat_index": 142, "class_name": "V"
    }))
    print(f"  send_alert        : {out[:80]}")
    assert "ALERT" in out
    out = ALL_TOOLS[6].run(json.dumps({"patient_id": "patient_test_001", "limit": 3}))
    print(f"  check_history     : {out[:80]}")
    assert isinstance(out, str)
    good_summary = {"is_good": True, "issues": [], "snr_db": 18.5, "mean_hr_bpm": 72.0, "duration_sec": 10.0}
    out = ALL_TOOLS[7].run(json.dumps({"patient_id": "patient_test_001", "raw_signal_summary": good_summary}))
    print(f"  detect_sensor (OK): {out[:80]}")
    assert "OK" in out
    bad_summary = {"is_good": False, "issues": ["Flat-line detected"], "snr_db": -10.0, "mean_hr_bpm": 0.0, "duration_sec": 5.0}
    out = ALL_TOOLS[7].run(json.dumps({"patient_id": "patient_test_001", "raw_signal_summary": bad_summary}))
    print(f"  detect_sensor(BAD): {out[:80]}")
    assert "ISSUES" in out
    _pass("tools_no_llm")

def test_tools_with_llm(db, llm_status):
    print(f"\n{SEP}\nTEST 5 — Tools with LLM (Groq)\n{SEP2}")
    if not llm_status["available"]:
        print("  SKIPPED — Groq API not available")
        return
    from src.layer4_agent.llm_config import get_llm
    llm = get_llm(temperature=0.2)
    init_tools(db, llm)
    out = ALL_TOOLS[2].run(json.dumps({
        "class_name"      : "Premature Ventricular Contraction (V)",
        "confidence"      : 0.91,
        "patient_context" : "65 year old, hypertension"
    }))
    print(f"  explain_arrhythmia:\n    {out[:150]}...")
    assert len(out) > 50
    assert "DISCLAIMER" in out
    out = ALL_TOOLS[5].run(json.dumps({
        "question": "What does frequent PVC indicate?",
        "context" : "Patient has 15% PVC rate, age 65"
    }))
    print(f"  answer_query:\n    {out[:150]}...")
    assert len(out) > 50
    _pass("tools_with_llm")

def test_generate_report(db):
    print(f"\n{SEP}\nTEST 6 — PDF Report Generation\n{SEP2}")
    init_tools(db, None)
    session_data = {
        "session_risk"  : "High",
        "total_beats"   : 200,
        "anomaly_count" : 30,
        "anomaly_rate"  : 0.15,
        "recording_sec" : 160.0,
        "dominant_class": "V",
    }
    out = ALL_TOOLS[4].run(json.dumps({
        "patient_id"   : "patient_test_001",
        "session_data" : session_data
    }))
    print(f"  {out}")
    assert "report_" in out.lower() or "Error" not in out
    _pass("generate_report")

def test_agent_run(llm_status):
    print(f"\n{SEP}\nTEST 7 — Full Agent Run\n{SEP2}")
    if not llm_status["available"]:
        print("  SKIPPED — Groq API not available")
        return
    agent = ECGAgent(TEST_DB)
    agent.start()
    status = agent.status()
    print(f"  Agent status: {status}")
    assert status["agent_ready"]
    assert status["tools_count"] == 8
    response = agent.run(
        "Briefly explain what a Premature Ventricular Contraction is.",
        patient_id="patient_test_001"
    )
    print(f"  Agent response:\n    {response[:200]}...")
    assert len(response) > 30
    direct = agent.run_tool_directly(
        "assess_risk",
        json.dumps({"anomaly_rate": 0.25, "dominant_class": "V",
                    "critical_beats": 1, "total_beats": 300})
    )
    result = json.loads(direct)
    print(f"  Direct tool call: risk={result['risk_level']}")
    assert result["risk_level"] in ("High", "Critical")
    agent.reset_memory()
    _pass("agent_run")

def test_inference_integration(db):
    print(f"\n{SEP}\nTEST 8 — Inference + Agent Integration\n{SEP2}")
    sys.path.insert(0, r"C:\ecg_arrhythmia")
    from src.layer3_models.inference import ModelLoader, predict_beats, BEAT_LENGTH
    DATA_DIR = Path(r"D:\ecg_project\datasets\mitbih\processed")
    if not (DATA_DIR / "test" / "beats.npy").exists():
        print("  SKIPPED — test beats not found")
        return
    import numpy as np
    beats  = np.load(DATA_DIR / "test" / "beats.npy")[:50]
    labels = np.load(DATA_DIR / "test" / "labels.npy")[:50]
    timestamps = np.arange(50, dtype=float) * 0.8
    loader = ModelLoader().load_all()
    batch  = predict_beats(beats, timestamps, loader)
    session_id = db.save_session("integration_test_patient", batch)
    print(f"  Session saved: id={session_id} | risk={batch.session_risk}")
    init_tools(db, None)
    risk_out = ALL_TOOLS[1].run(json.dumps({
        "anomaly_rate"  : batch.anomaly_rate,
        "dominant_class": batch.dominant_class,
        "critical_beats": len(batch.critical_beats()),
        "total_beats"   : batch.total_beats,
    }))
    risk_result = json.loads(risk_out)
    print(f"  Risk assessment: {risk_result['risk_level']} — {risk_result['reason']}")
    if risk_result["risk_level"] != "Low" and batch.anomaly_beats():
        first_anomaly = batch.anomaly_beats()[0]
        alert_out = ALL_TOOLS[3].run(json.dumps({
            "patient_id": "integration_test_patient",
            "risk_level": risk_result["risk_level"],
            "message"   : f"{batch.anomaly_count} anomalies in {batch.recording_sec}s",
            "beat_index": first_anomaly.beat_index,
            "class_name": first_anomaly.cnn_short_name,
        }))
        print(f"  Alert: {alert_out[:80]}")
    _pass("inference_integration")

if __name__ == "__main__":
    print(f"\n{SEP}")
    print("ECG Agent — Phase 6 Tests")
    print(SEP)
    passed = failed = 0
    try:
        llm_status = test_llm_config()
        passed += 1
    except Exception as e:
        _fail("llm_config", e); failed += 1
        llm_status = {"available": False, "has_key": False, "internet": False}
    try:
        db = test_database()
        passed += 1
    except Exception as e:
        _fail("database", e); failed += 1
        print("CRITICAL: DB failed — skipping remaining tests")
        sys.exit(1)
    for name, fn, args in [
        ("memory",               test_memory,               ()),
        ("tools_no_llm",         test_tools_no_llm,         (db,)),
        ("tools_with_llm",       test_tools_with_llm,       (db, llm_status)),
        ("generate_report",      test_generate_report,      (db,)),
        ("agent_run",            test_agent_run,            (llm_status,)),
        ("inference_integration",test_inference_integration,(db,)),
    ]:
        try:
            fn(*args)
            passed += 1
        except Exception as e:
            _fail(name, e); failed += 1
    try:
        Path(TEST_DB).unlink(missing_ok=True)
    except Exception:
        pass
    print(f"\n{SEP}")
    print(f"Results : {passed}/{passed+failed} passed")
    if failed == 0:
        print("Phase 6 complete — AI Agent ready.")
        print("Next: Phase 7 — Streamlit Dashboard")
    else:
        print(f"{failed} test(s) failed — check errors above.")
    print(SEP)