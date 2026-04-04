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
    global _db, _llm
    _db  = db
    _llm = llm

@tool
def monitor_trends(input_json: str) -> str:
    """Summarise recent anomaly-rate trends for a patient from stored session history."""
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
            f"{trend['trend'].upper()} | Avg anomaly rate: {trend['avg_anomaly_rate']}% | "
            f"Latest rate: {trend['latest_rate']}%"
        )
    except Exception as e:
        logger.error(f"monitor_trends error: {e}")
        return f"Error in monitor_trends: {e}"

@tool
def assess_risk(input_json: str) -> str:
    """Assign a LOW/MEDIUM/HIGH/CRITICAL risk label from anomaly statistics."""
    try:
        data           = json.loads(input_json)
        anomaly_rate   = float(data.get("anomaly_rate", 0.0))
        dominant_class = data.get("dominant_class", "N")
        critical_beats = int(data.get("critical_beats", 0))
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
    """Explain a detected arrhythmia class and its clinical significance."""
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
    """Persist and write an alert record for a risky ECG finding."""
    try:
        from pathlib import Path
        data       = json.loads(input_str)
        patient_id = data.get("patient_id", "UNKNOWN")
        risk_level = data.get("risk_level", "Low")
        message    = data.get("message",    "No details provided.")
        class_name = data.get("class_name", "")
        beat_index = data.get("beat_index", -1)
        color_map = {"Low": "green", "Medium": "yellow", "High": "orange", "Critical": "red"}
        color = color_map.get(risk_level, "yellow")
        ts    = datetime.utcnow().isoformat()
        if _db is not None:
            try:
                _db.save_alert(patient_id=patient_id, risk_level=risk_level, color=color, message=message, beat_index=beat_index, class_name=class_name)
            except Exception:
                pass
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
                f"{'-'*55}\n"
            )
        return f"ALERT [{risk_level.upper()}] fired for {patient_id}. File: {file_path.name}"
    except Exception as e:
        return f"Alert error: {e}"

@tool
def generate_report(input_json: str) -> str:
    """Generate a detailed PDF report from structured ECG session data."""
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
        doc    = SimpleDocTemplate(str(filename), pagesize=A4, topMargin=1.6*cm, bottomMargin=1.6*cm)
        styles = getSampleStyleSheet()
        story  = []

        title_style = ParagraphStyle("title", parent=styles["Title"], fontSize=19, spaceAfter=12)
        section_style = ParagraphStyle("section", parent=styles["Heading2"], fontSize=13, textColor=colors.HexColor("#115e59"), spaceBefore=10, spaceAfter=8)
        body_style = styles["BodyText"]

        story.append(Paragraph("ECG Arrhythmia Detection Report", title_style))
        story.append(Paragraph(f"Patient: <b>{patient_id}</b>", body_style))
        story.append(Paragraph(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", body_style))
        story.append(Spacer(1, 0.4*cm))

        summary_text = session_data.get("summary_text", "No summary provided.")
        report_focus = session_data.get("report_focus", "Clinical review summary")
        clinical_notes = session_data.get("clinical_notes", "")
        story.append(Paragraph("Executive Summary", section_style))
        story.append(Paragraph(summary_text, body_style))
        story.append(Paragraph(f"Report focus: <b>{report_focus}</b>", body_style))

        meta = [
            ["Risk Level", session_data.get("session_risk", "N/A")],
            ["Total Beats", str(session_data.get("total_beats", "N/A"))],
            ["Anomaly Count", str(session_data.get("anomaly_count", "N/A"))],
            ["Anomaly Rate", f"{float(session_data.get('anomaly_rate', 0))*100:.1f}%"],
            ["Duration", f"{session_data.get('recording_sec', 'N/A')}s"],
            ["Dominant Class", session_data.get("dominant_class", "N/A")],
        ]
        meta_table = Table(meta, colWidths=[5.2*cm, 9.8*cm])
        meta_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#dbeeea")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.whitesmoke, colors.white]),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
        ]))
        story.append(Paragraph("Session Metrics", section_style))
        story.append(meta_table)

        class_counts = session_data.get("class_counts", {}) or {}
        if class_counts:
            story.append(Paragraph("Class Distribution", section_style))
            class_rows = [["Class", "Count"]] + [[str(k), str(v)] for k, v in class_counts.items()]
            class_table = Table(class_rows, colWidths=[6*cm, 4*cm])
            class_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f4f8")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
            ]))
            story.append(class_table)

        top_anomalies = session_data.get("top_anomalies", []) or []
        if top_anomalies:
            story.append(Paragraph("Top Anomaly Highlights", section_style))
            anomaly_rows = [["Beat", "Time (s)", "Class", "Confidence", "Risk"]]
            for item in top_anomalies:
                anomaly_rows.append([
                    str(item.get("beat_index", "-")),
                    f"{float(item.get('timestamp_sec', 0.0)):.2f}",
                    str(item.get("class_short", item.get("class_name", "-"))),
                    f"{float(item.get('confidence', 0.0))*100:.1f}%",
                    str(item.get("risk_level", "-")),
                ])
            anomaly_table = Table(anomaly_rows, colWidths=[2*cm, 2.2*cm, 3*cm, 3*cm, 2.5*cm])
            anomaly_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#fdecea")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
            ]))
            story.append(anomaly_table)

        review_flags = session_data.get("review_flags", []) or []
        if review_flags:
            story.append(Paragraph("Review Flags", section_style))
            for item in review_flags:
                story.append(Paragraph(f"• {item}", body_style))

        if clinical_notes:
            story.append(Paragraph("Clinician Notes", section_style))
            story.append(Paragraph(str(clinical_notes), body_style))

        recommendations = session_data.get("recommendations", []) or []
        if recommendations:
            story.append(Paragraph("Review Guidance", section_style))
            for item in recommendations:
                story.append(Paragraph(f"• {item}", body_style))

        disclaimer = (
            "<b>DISCLAIMER:</b> This report is generated by an AI-assisted system for clinical decision support only. "
            "It does not constitute a medical diagnosis. All findings must be reviewed and interpreted by a qualified clinician."
        )
        story.append(Spacer(1, 0.4*cm))
        story.append(Paragraph(disclaimer, body_style))

        doc.build(story)
        if _db:
            _db.log_tool_call("generate_report", patient_id, str(filename))
        return f"Report saved: {filename}"
    except Exception as e:
        logger.error(f"generate_report error: {e}")
        return f"Error generating report: {e}"

@tool
def answer_query(input_json: str) -> str:
    """Answer a clinician or disease-related question using provided ECG context."""
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
            f"Be factual, concise, and evidence-based. Explain the likely disease meaning of the ECG pattern when relevant, "
            f"state what the finding may suggest clinically, include what to review next, and clearly state what should not be assumed "
            f"without clinician confirmation. Do not recommend specific medications or treatments."
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
    """Return recent patient session history from the database."""
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
                f"  [{h['timestamp'][:10]}] Risk={h['risk_level']} | Anomalies={h['anomaly_count']} ({h['anomaly_rate']}%) | Class={h['dominant_class']} | Duration={h['recording_sec']}s"
            )
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"check_history error: {e}")
        return f"Error in check_history: {e}"

@tool
def detect_sensor_issue(input_json: str) -> str:
    """Summarise ECG signal quality and likely sensor issues from raw-signal metadata."""
    try:
        data    = json.loads(input_json)
        pid     = data.get("patient_id", "unknown")
        summary = data.get("raw_signal_summary", {})
        is_good = summary.get("is_good", True)
        issues  = summary.get("issues", [])
        snr_db  = summary.get("snr_db", 0.0)
        mean_hr = summary.get("mean_hr_bpm", 0.0)
        dur     = summary.get("duration_sec", 0.0)
        if is_good:
            result = f"Signal quality OK | SNR={snr_db} dB | HR={mean_hr} bpm | Duration={dur}s"
        else:
            result = f"SIGNAL ISSUES DETECTED for {pid}:\n" + "\n".join(f"  - {i}" for i in issues)
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

