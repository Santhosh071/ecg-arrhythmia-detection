import streamlit as st
from datetime import datetime

ALERT_STYLES = {
    "green" : {"bg": "#1B4332", "border": "#4CAF50", "icon": "✅"},
    "yellow": {"bg": "#3D2E00", "border": "#FFC107", "icon": "⚠️"},
    "orange": {"bg": "#3E1A00", "border": "#FF9800", "icon": "🔶"},
    "red"   : {"bg": "#3B0A0A", "border": "#F44336", "icon": "🚨"},
}

RISK_BADGE = {
    "Low"     : ("#4CAF50", "⬤ Low"),
    "Medium"  : ("#FFC107", "⬤ Medium"),
    "High"    : ("#FF9800", "⬤ High"),
    "Critical": ("#F44336", "⬤ Critical"),
}


def render_risk_badge(risk_level: str):
    """Display a coloured risk badge in sidebar or main area."""
    color, label = RISK_BADGE.get(risk_level, ("#9E9E9E", "⬤ Unknown"))
    st.markdown(
        f'<div style="background:#1A1A1A;border-left:5px solid {color};'
        f'padding:10px 16px;border-radius:6px;margin-bottom:8px;">'
        f'<span style="color:{color};font-size:20px;font-weight:700;">{label}</span>'
        f'<br><span style="color:#999;font-size:12px;">Session Risk Level</span></div>',
        unsafe_allow_html=True,
    )

def render_stats_row(batch_result):
    """Top metrics row — 4 columns with key session stats."""
    if batch_result is None:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Beats",   "—")
        col2.metric("Anomalies",     "—")
        col3.metric("Anomaly Rate",  "—")
        col4.metric("Session Risk",  "—")
        return
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Beats",  batch_result.total_beats)
    col2.metric("Anomalies",    batch_result.anomaly_count)
    col3.metric("Anomaly Rate", f"{batch_result.anomaly_rate*100:.1f}%")
    col4.metric("Session Risk", batch_result.session_risk)


def render_alert_card(alert: dict):
    color  = alert.get("color", "yellow")
    style  = ALERT_STYLES.get(color, ALERT_STYLES["yellow"])
    icon   = style["icon"]
    ts_raw = alert.get("timestamp", "")
    try:
        ts = datetime.fromisoformat(ts_raw).strftime("%H:%M:%S")
    except Exception:
        ts = ts_raw[:19] if ts_raw else "—"

    cls_tag = f" | Class: {alert['class_name']}" if alert.get("class_name") else ""
    st.markdown(
        f'<div style="background:{style["bg"]};border-left:4px solid {style["border"]};'
        f'padding:8px 12px;border-radius:4px;margin-bottom:6px;">'
        f'<span style="color:{style["border"]};font-weight:600;">'
        f'{icon} {alert["risk_level"]}{cls_tag}</span>'
        f'<span style="color:#AAA;font-size:11px;float:right;">{ts}</span>'
        f'<br><span style="color:#DDD;font-size:13px;">{alert["message"]}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

def render_alert_panel(batch_result, max_alerts: int = 15):
    """
    Render scrollable alert feed from a BatchResult.

    Shows up to max_alerts anomaly beats, newest first.
    Each card shows class, confidence, risk, timestamp.
    """
    st.subheader("🔔 Alert Feed")

    if batch_result is None or batch_result.anomaly_count == 0:
        st.markdown(
            '<div style="color:#4CAF50;padding:10px;background:#1B4332;'
            'border-radius:4px;text-align:center;">✅ No anomalies detected</div>',
            unsafe_allow_html=True,
        )
        return
    anomalies = batch_result.anomaly_beats()
    st.caption(f"{len(anomalies)} anomalous beat(s) detected")
    for beat_r in reversed(anomalies[-max_alerts:]):
        alert = {
            "risk_level": beat_r.risk_level,
            "color"     : beat_r.alert_color,
            "message"   : (f"Beat #{beat_r.beat_index} at {beat_r.timestamp_sec:.2f}s — "
                           f"{beat_r.cnn_class_name} "
                           f"({beat_r.cnn_confidence*100:.1f}% confidence) | "
                           f"Z-score: {beat_r.transformer_score:.2f}"),
            "timestamp" : datetime.utcnow().isoformat(),
            "class_name": beat_r.cnn_short_name,
        }
        render_alert_card(alert)

def render_db_alerts(alerts: list, max_alerts: int = 20):
    """Render alerts loaded from SQLite database (patient history page)."""
    st.subheader("🔔 Recent Alerts")
    if not alerts:
        st.info("No alerts on record.")
        return
    for alert in alerts[:max_alerts]:
        render_alert_card(alert)

def render_disclaimer():
    """Always-visible disclaimer banner (Rule 18 & 19)."""
    st.markdown(
        '<div style="background:#1A1A2E;border:1px solid #3D3D6B;padding:8px 14px;'
        'border-radius:4px;margin-top:10px;">'
        '⚕️ <b>Clinical Disclaimer:</b> This system is a decision-support tool only. '
        'All findings must be reviewed and confirmed by a qualified clinician. '
        'Do not act on AI-generated alerts without clinical judgement.</div>',
        unsafe_allow_html=True,
    )