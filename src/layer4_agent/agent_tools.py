import os
import json
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv()

REPORTS_PATH = Path(os.getenv("REPORTS_PATH", "C:/ecg_arrhythmia/outputs/reports"))
REPORTS_PATH.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)
_db  = None
_llm = None

def init_tools(db, llm):
    """Called once by agent.py to inject DB and LLM into tool scope."""
    global _db, _llm
    _db  = db
    _llm = llm

@tool
def monitor_trends(input_json: str) -> str:
    """
    Analyse anomaly frequency trend for a patient across recent sessions.

    Input JSON:
        {"patient_id": "P001", "last_n_sessions": 5}

    Returns trend summary: improving / worsening / stable + rates.

    DISCLAIMER: For monitoring support only. Not a clinical diagnosis.
    """
    try:
        data       = json.loads(input_json)
        patient_id = data.get("patient_id", "unknown")
        n          = int(data.get("last_n_sessions", 5))

        if _db is None:
            return "Database not initialised. Call init_tools() first."

        trend = _db.get_anomaly_trend(patient_id, last_n_sessions=n)
        _db.log_tool_call("monitor_trends", input_json, json.dumps(trend), patient_id)

        if trend["trend"] == "no_data":
            return f"No session history found for patient {patient_id}."

        return (
            f"Trend for {patient_id} over {trend['sessions']} sessions: "
            f"{trend['trend'].upper()} | "
            f"Avg anomaly rate: {trend['avg_anomaly_rate']}% | "
            f"Latest rate: {trend['latest_rate']}%"
        )
    except Exception as e:
        logger.error(f"monitor_trends error: {e}")
        return f"Error in monitor_trends: {e}"

@tool
def assess_risk(input_json: str) -> str:
    """
    Compute risk level from a session summary.

    Input JSON:
        {"anomaly_rate": 0.15, "dominant_class": "V",
         "critical_beats": 2, "total_beats": 200}

    Returns: Low / Medium / High / Critical with reasoning.

    DISCLAIMER: Risk scores are decision-support only. Clinician review required.
    """
    try:
        data           = json.loads(input_json)
        anomaly_rate   = float(data.get("anomaly_rate", 0.0))
        dominant_class = data.get("dominant_class", "N")
        critical_beats = int(data.get("critical_beats", 0))
        total_beats    = int(data.get("total_beats", 1))
        if critical_beats > 0 or dominant_class in ("E", "V") and anomaly_rate >= 0.20:
            risk  = "Critical"
            color = "red"
            reason = f"{critical_beats} critical beats detected with {dominant_class} class dominance."
        elif anomaly_rate >= 0.30 or dominant_class in ("V", "E", "F"):
            risk  = "High"
            color = "orange"
            reason = f"High anomaly rate ({anomaly_rate*100:.1f}%) with dominant class {dominant_class}."
        elif anomaly_rate >= 0.10 or dominant_class in ("L", "R", "A"):
            risk  = "Medium"
            color = "yellow"
            reason = f"Moderate anomaly rate ({anomaly_rate*100:.1f}%)."
        else:
            risk  = "Low"
            color = "green"
            reason = f"Low anomaly rate ({anomaly_rate*100:.1f}%). No significant arrhythmia detected."

        result = {
            "risk_level"   : risk,
            "color"        : color,
            "reason"       : reason,
            "anomaly_rate" : f"{anomaly_rate*100:.1f}%",
            "disclaimer"   : "Decision-support only. Not a clinical diagnosis.",
        }
        if _db:
            _db.log_tool_call("assess_risk", input_json, json.dumps(result))
        return json.dumps(result)
    except Exception as e:
        logger.error(f"assess_risk error: {e}")
        return f"Error in assess_risk: {e}"

@tool
def explain_arrhythmia(input_json: str) -> str:
    """
    Generate a plain-English clinical explanation of a detected arrhythmia class.

    Input JSON:
        {"class_name": "Premature Ventricular Contraction (V)",
         "confidence": 0.91, "patient_context": "elderly, hypertension"}

    Returns LLM-generated explanation in simple clinical language.

    DISCLAIMER: AI-generated explanation. Must be reviewed by a clinician.
    """
    try:
        data            = json.loads(input_json)
        class_name      = data.get("class_name", "Unknown")
        confidence      = float(data.get("confidence", 0.0))
        patient_context = data.get("patient_context", "no additional context")
        if _llm is None:
            return "LLM not initialised. Check GROQ_API_KEY in .env."
        prompt = (
            f"You are a clinical cardiology assistant.\n"
            f"Explain the following ECG finding to a clinician in 3-4 sentences:\n\n"
            f"Detected arrhythmia : {class_name}\n"
            f"Model confidence    : {confidence*100:.1f}%\n"
            f"Patient context     : {patient_context}\n\n"
            f"Include: what it is, clinical significance, and when to be concerned.\n"
            f"Be concise and factual. Do not recommend specific treatments."
        )
        response = _llm.invoke(prompt)
        explanation = response.content if hasattr(response, "content") else str(response)
        disclaimer = "\n\nDISCLAIMER: AI-generated. Final interpretation requires clinician review."
        result     = explanation + disclaimer
        if _db:
            _db.log_tool_call("explain_arrhythmia", class_name, explanation[:300])
        return result
    except Exception as e:
        logger.error(f"explain_arrhythmia error: {e}")
        return f"Error in explain_arrhythmia: {e}"

@tool
def _send_alert(input_str: str) -> str:
    """
    Send a tiered alert.
    Saves to SQLite DB AND writes .txt file to outputs/alerts/

    Input JSON keys:
        patient_id : str
        risk_level : Low | Medium | High | Critical
        message    : str
        class_name : str (optional)
        beat_index : int (optional)
    """
    try:
        import os
        from pathlib import Path
        from datetime import datetime

        data       = json.loads(input_str)
        patient_id = data.get("patient_id", "UNKNOWN")
        risk_level = data.get("risk_level", "Low")
        message    = data.get("message",    "No details provided.")
        class_name = data.get("class_name", "")
        beat_index = data.get("beat_index", -1)

        color_map = {
            "Low"     : "green",
            "Medium"  : "yellow",
            "High"    : "orange",
            "Critical": "red",
        }
        color = color_map.get(risk_level, "yellow")
        ts    = datetime.utcnow().isoformat()

        # ── Save to SQLite ────────────────────────────────────────────────
        if _db is not None:
            try:
                _db.save_alert(
                    patient_id = patient_id,
                    risk_level = risk_level,
                    color      = color,
                    message    = message,
                    beat_index = beat_index,
                    class_name = class_name,
                )
            except Exception:
                pass

        # ── Save .txt file to outputs/alerts/ ─────────────────────────────
        alerts_dir = Path(os.getenv("PROJECT_ROOT", r"C:\ecg_arrhythmia")) / "outputs" / "alerts"
        alerts_dir.mkdir(parents=True, exist_ok=True)

        date_str  = datetime.utcnow().strftime("%Y-%m-%d")
        safe_pid  = patient_id.replace("/", "-").replace("\\", "-")
        file_path = alerts_dir / f"alerts_{safe_pid}_{date_str}.txt"

        cls_tag   = f" | Class: {class_name}" if class_name else ""
        beat_tag  = f" | Beat #{beat_index}"  if beat_index >= 0 else ""

        with open(str(file_path), "a", encoding="utf-8") as f:
            f.write(
                f"[{ts}] ALERT\n"
                f"  Patient   : {patient_id}\n"
                f"  Risk      : {risk_level} ({color})\n"
                f"  Message   : {message}{cls_tag}{beat_tag}\n"
                f"  Disclaimer: AI-generated. Requires clinician review.\n"
                f"{'─'*55}\n"
            )

        return (
            f"ALERT [{risk_level.upper()}] fired for {patient_id}. "
            f"File: {file_path.name}"
        )

    except Exception as e:
        return f"Alert error: {e}"

@tool
def generate_report(input_json: str) -> str:
    """
    Generate a PDF session summary report.

    Input JSON:
        {"patient_id": "P001", "session_data": {...}}

    Saves PDF to REPORTS_PATH and returns the file path.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        data         = json.loads(input_json)
        patient_id   = data.get("patient_id", "unknown")
        session_data = data.get("session_data", {})
        timestamp    = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename     = REPORTS_PATH / f"report_{patient_id}_{timestamp}.pdf"
        doc    = SimpleDocTemplate(str(filename), pagesize=A4,
                                   topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()
        story  = []
        title_style = ParagraphStyle("title", parent=styles["Title"],
                                     fontSize=18, spaceAfter=12)
        story.append(Paragraph("ECG Arrhythmia Detection Report", title_style))
        story.append(Spacer(1, 0.5*cm))
        meta = [
            ["Patient ID",    patient_id],
            ["Generated",     datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")],
            ["Risk Level",    session_data.get("session_risk", "N/A")],
            ["Total Beats",   str(session_data.get("total_beats", "N/A"))],
            ["Anomaly Count", str(session_data.get("anomaly_count", "N/A"))],
            ["Anomaly Rate",  f"{float(session_data.get('anomaly_rate', 0))*100:.1f}%"],
            ["Duration",      f"{session_data.get('recording_sec', 'N/A')}s"],
            ["Dominant Class",session_data.get("dominant_class", "N/A")],
        ]
        t = Table(meta, colWidths=[5*cm, 10*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (0, -1), colors.lightgrey),
            ("FONTNAME",    (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE",    (0, 0), (-1, -1), 10),
            ("GRID",        (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.whitesmoke, colors.white]),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.5*cm))
        disclaimer = (
            "<b>DISCLAIMER:</b> This report is generated by an AI-assisted system "
            "for clinical decision support only. It does not constitute a medical diagnosis. "
            "All findings must be reviewed and interpreted by a qualified clinician."
        )
        story.append(Paragraph(disclaimer, styles["Normal"]))

        doc.build(story)

        if _db:
            _db.log_tool_call("generate_report", patient_id, str(filename))
        return f"Report saved: {filename}"
    except Exception as e:
        logger.error(f"generate_report error: {e}")
        return f"Error generating report: {e}"

@tool
def answer_query(input_json: str) -> str:
    """
    Answer a clinician's natural language question about ECG findings.

    Input JSON:
        {"question": "What does a high PVC rate indicate?",
         "context": "Patient has 15% PVC rate, age 65, hypertension"}

    Returns LLM answer grounded in the provided context.

    DISCLAIMER: AI-generated answer. Not a substitute for clinical expertise.
    """
    try:
        data     = json.loads(input_json)
        question = data.get("question", "")
        context  = data.get("context", "No context provided.")
        if not question:
            return "No question provided."
        if _llm is None:
            return "LLM not initialised. Check GROQ_API_KEY in .env."
        prompt = (
            f"You are a clinical cardiology decision-support assistant.\n"
            f"Answer the following question using the provided patient context.\n\n"
            f"Context : {context}\n"
            f"Question: {question}\n\n"
            f"Be factual, concise, and evidence-based. "
            f"Do not recommend specific medications or treatments."
        )
        response = _llm.invoke(prompt)
        answer   = response.content if hasattr(response, "content") else str(response)
        if _db:
            _db.log_tool_call("answer_query", question[:200], answer[:300])
        return answer + "\n\nDISCLAIMER: AI-generated. Verify with clinical guidelines."
    except Exception as e:
        logger.error(f"answer_query error: {e}")
        return f"Error in answer_query: {e}"

@tool
def check_history(input_json: str) -> str:
    """
    Retrieve past session history for a patient from the database.

    Input JSON:
        {"patient_id": "P001", "limit": 5}

    Returns list of past sessions with risk, anomaly rates, timestamps.
    """
    try:
        data       = json.loads(input_json)
        patient_id = data.get("patient_id", "unknown")
        limit      = int(data.get("limit", 5))
        if _db is None:
            return "Database not initialised."
        history = _db.get_patient_history(patient_id, limit=limit)
        _db.log_tool_call("check_history", patient_id, f"{len(history)} records", patient_id)
        if not history:
            return f"No history found for patient {patient_id}."
        lines = [f"History for {patient_id} (last {len(history)} sessions):"]
        for h in history:
            lines.append(
                f"  [{h['timestamp'][:10]}] Risk={h['risk_level']} | "
                f"Anomalies={h['anomaly_count']} ({h['anomaly_rate']}%) | "
                f"Class={h['dominant_class']} | Duration={h['recording_sec']}s"
            )
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"check_history error: {e}")
        return f"Error in check_history: {e}"

@tool
def detect_sensor_issue(input_json: str) -> str:
    """
    Analyse signal quality and flag sensor or electrode issues.

    Input JSON:
        {"patient_id": "P001", "raw_signal_summary":
            {"is_good": false, "issues": ["Flat-line detected"],
             "snr_db": -10.5, "mean_hr_bpm": 0}}

    OR pass raw signal array as base64 for full analysis.

    Returns: quality verdict, detected issues, recommended action.
    """
    try:
        data    = json.loads(input_json)
        pid     = data.get("patient_id", "unknown")
        summary = data.get("raw_signal_summary", {})

        is_good   = summary.get("is_good", True)
        issues    = summary.get("issues", [])
        snr_db    = summary.get("snr_db", 0.0)
        mean_hr   = summary.get("mean_hr_bpm", 0.0)
        dur       = summary.get("duration_sec", 0.0)
        if is_good:
            result = f"Signal quality OK | SNR={snr_db} dB | HR={mean_hr} bpm | Duration={dur}s"
            action = "No action needed."
        else:
            result = f"SIGNAL ISSUES DETECTED for {pid}:\n"
            result += "\n".join(f"  - {i}" for i in issues)
            if snr_db < 5.0:
                action = "Check electrode contact and reduce patient movement."
            elif mean_hr == 0:
                action = "No heartbeat detected — check lead placement immediately."
            else:
                action = "Review signal source. Consider repositioning electrodes."
            result += f"\nRecommended action: {action}"
        if _db:
            _db.log_tool_call("detect_sensor_issue", input_json[:200], result[:300], pid)
        return result
    except Exception as e:
        logger.error(f"detect_sensor_issue error: {e}")
        return f"Error in detect_sensor_issue: {e}"

ALL_TOOLS = [
    monitor_trends,
    assess_risk,
    explain_arrhythmia,
    _send_alert,
    generate_report,
    answer_query,
    check_history,
    detect_sensor_issue,
]