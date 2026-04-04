from pathlib import Path
import json

REPORTS_ROOT = Path(Path(__file__).resolve().parents[1] / "outputs" / "reports")


def build_agent_context(patient_id: str | None, session_context: dict | None) -> str:
    if not session_context:
        return f"Patient ID: {patient_id or 'unknown'}"

    anomaly_rate = session_context.get("anomaly_rate")
    if isinstance(anomaly_rate, (int, float)):
        anomaly_rate_text = f"{anomaly_rate * 100:.1f}%"
    else:
        anomaly_rate_text = "unknown"

    parts = [
        f"Patient ID: {patient_id or 'unknown'}",
        f"Session risk: {session_context.get('session_risk', 'unknown')}",
        f"Dominant class: {session_context.get('dominant_class', 'unknown')}",
        f"Anomaly rate: {anomaly_rate_text}",
        f"Anomaly count: {session_context.get('anomaly_count', 'unknown')}",
        f"Total beats: {session_context.get('total_beats', 'unknown')}",
    ]

    class_counts = session_context.get("class_counts")
    if isinstance(class_counts, dict) and class_counts:
        distribution = ", ".join(f"{label}: {count}" for label, count in class_counts.items())
        parts.append(f"Class distribution: {distribution}")

    review_flags = session_context.get("review_flags")
    if isinstance(review_flags, list) and review_flags:
        parts.append(f"Review flags: {', '.join(str(flag) for flag in review_flags)}")

    notes = session_context.get("clinical_notes")
    if notes:
        parts.append(f"Clinician notes: {notes}")

    return " | ".join(parts)


def build_tool_payload(question: str, context: str) -> str:
    return json.dumps({"question": question, "context": context})
