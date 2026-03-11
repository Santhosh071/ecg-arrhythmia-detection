
import streamlit as st
import pandas as pd
from .ecg_plot import plot_anomaly_trend
from .alert_panel import render_db_alerts, RISK_BADGE

def render_history_page(db, patient_id: str):
    """
    Full patient history page.

    Parameters
    ----------
    db         : ECGDatabase
    patient_id : patient identifier string
    """
    st.header("📊 Patient History")

    if not patient_id or patient_id.strip() == "":
        st.warning("Enter a Patient ID in the sidebar to load history.")
        return
    history = db.get_patient_history(patient_id, limit=20)
    if not history:
        st.info(f"No session history found for patient: {patient_id}")
        return
    st.plotly_chart(plot_anomaly_trend(history), use_container_width=True)
    trend_data = db.get_anomaly_trend(patient_id, last_n_sessions=5)
    col1, col2, col3 = st.columns(3)
    col1.metric("Trend",            trend_data["trend"].capitalize())
    col2.metric("Avg Anomaly Rate", f"{trend_data['avg_anomaly_rate']:.1f}%")
    col3.metric("Sessions Reviewed",trend_data["sessions"])
    st.markdown("---")
    st.subheader("📋 Session Records")
    df = pd.DataFrame(history)
    df["timestamp"]    = df["timestamp"].str[:19].str.replace("T", " ")
    df["anomaly_rate"] = df["anomaly_rate"].map(lambda x: f"{x:.1f}%")
    df["recording_sec"]= df["recording_sec"].map(lambda x: f"{x:.0f}s")
    df = df.rename(columns={
        "session_id"    : "ID",
        "timestamp"     : "Date & Time",
        "risk_level"    : "Risk",
        "anomaly_count" : "Anomalies",
        "anomaly_rate"  : "Rate",
        "dominant_class": "Class",
        "recording_sec" : "Duration",
        "total_beats"   : "Beats",
    })
    def color_risk(val):
        color_map = {"Low": "color: #4CAF50", "Medium": "color: #FFC107",
                     "High": "color: #FF9800", "Critical": "color: #F44336"}
        return color_map.get(val, "")
    styled = df.style.applymap(color_risk, subset=["Risk"])
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.markdown("---")
    alerts = db.get_recent_alerts(patient_id, limit=20)
    render_db_alerts(alerts)