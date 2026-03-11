import sys
import os
import numpy as np
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
st.set_page_config(
    page_title = "ECG Arrhythmia Monitor",
    page_icon  = "🫀",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)
from src.layer3_models.inference import (
    ModelLoader, predict_ecg, predict_beats,
    check_signal_quality, BEAT_LENGTH, MIT_BIH_FS,
)
from src.layer4_agent            import ECGAgent, ECGDatabase, check_llm_status
from src.layer5_dashboard.ecg_plot      import (
    plot_ecg_waveform, plot_single_beat,
    plot_class_distribution, plot_confidence_bars,
)
from src.layer5_dashboard.alert_panel   import (
    render_alert_panel, render_risk_badge,
    render_stats_row, render_disclaimer,
)
from src.layer5_dashboard.chat_interface import render_chat, render_quick_actions
from src.layer5_dashboard.history_charts import render_history_page

MODELS_DIR = Path(os.getenv("MODELS_PATH",  r"D:\ecg_project\models"))
DATA_DIR   = Path(os.getenv("PROCESSED_DATA_PATH", r"D:\ecg_project\datasets\mitbih\processed"))
DB_PATH    = os.getenv("DB_PATH", r"D:\ecg_project\database\patient_history.db")

def _init_state():
    defaults = {
        "loader"        : None,
        "agent"         : None,
        "db"            : None,
        "batch_result"  : None,
        "beats"         : None,
        "timestamps"    : None,
        "selected_beat" : 0,
        "patient_id"    : "P001",
        "models_loaded" : False,
        "agent_ready"   : False,
        "chat_messages" : [],
        "demo_mode"     : False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

@st.cache_resource(show_spinner="Loading models into memory...")
def load_models():
    loader = ModelLoader(str(MODELS_DIR))
    loader.load_all()
    return loader


@st.cache_resource(show_spinner="Starting AI agent...")
def load_agent():
    try:
        agent = ECGAgent(DB_PATH)
        agent.start()
        return agent
    except Exception as e:
        st.warning(f"Agent unavailable: {e}")
        return None


@st.cache_resource
def load_db():
    return ECGDatabase(DB_PATH)

def render_sidebar():
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/heart-with-pulse.png", width=60)
        st.title("ECG Monitor")
        st.caption("AI Arrhythmia Detection System")
        st.markdown("---")
        st.subheader("Patient")
        st.session_state.patient_id = st.text_input(
            "Patient ID", value=st.session_state.patient_id
        )
        st.markdown("---")
        st.subheader("Data Source")
        data_source = st.radio(
            "Select input",
            ["Demo (test data)", "Upload .npy file"],
            index=0,
        )
        uploaded = None
        if data_source == "Upload .npy file":
            uploaded = st.file_uploader("Upload ECG beats (.npy)", type=["npy"])

        st.markdown("---")
        run_pressed = st.button("▶️ Run Analysis", type="primary", use_container_width=True)
        st.markdown("---")
        st.subheader("System Status")
        _status_indicator("Models",  st.session_state.models_loaded)
        _status_indicator("Agent",   st.session_state.agent_ready)
        llm = check_llm_status()
        _status_indicator("Groq LLM", llm["available"])
        _status_indicator("Internet", llm["internet"])
        st.markdown("---")
        render_disclaimer()
    return data_source, uploaded, run_pressed

def _status_indicator(label: str, ok: bool):
    icon  = "🟢" if ok else "🔴"
    color = "#4CAF50" if ok else "#F44336"
    st.markdown(
        f'<div style="display:flex;justify-content:space-between;padding:2px 0;">'
        f'<span style="color:#CCC;">{label}</span>'
        f'<span style="color:{color};">{icon}</span></div>',
        unsafe_allow_html=True,
    )

def ensure_resources():
    if not st.session_state.models_loaded:
        try:
            st.session_state.loader       = load_models()
            st.session_state.models_loaded = True
        except Exception as e:
            st.error(f"Model loading failed: {e}")
    if not st.session_state.agent_ready:
        try:
            st.session_state.agent       = load_agent()
            st.session_state.db          = load_db()
            st.session_state.agent_ready = st.session_state.agent is not None
        except Exception as e:
            st.warning(f"Agent load warning: {e}")
            st.session_state.db = load_db()

def run_analysis(data_source: str, uploaded):
    """Load beats, run inference, store results in session state."""
    loader = st.session_state.loader
    if loader is None:
        st.error("Models not loaded. Check D:/ecg_project/models/ path.")
        return
    beats, timestamps = None, None
    if data_source == "Demo (test data)":
        test_path = DATA_DIR / "test" / "beats.npy"
        if not test_path.exists():
            st.error(f"Demo data not found at {test_path}")
            return
        all_beats = np.load(str(test_path))
        n         = min(200, len(all_beats))
        beats     = all_beats[:n]
        timestamps= np.arange(n) * (BEAT_LENGTH / MIT_BIH_FS)
        st.session_state.demo_mode = True
    elif uploaded is not None:
        try:
            raw = np.load(uploaded)
            if raw.ndim == 1:
                # Raw 1D signal — run full pipeline
                with st.spinner("Preprocessing ECG signal..."):
                    from src.layer3_models.inference.preprocess import preprocess_ecg
                    beats, timestamps = preprocess_ecg(raw, verbose=False)
            elif raw.ndim == 2 and raw.shape[1] == BEAT_LENGTH:
                beats      = raw
                timestamps = np.arange(len(raw)) * (BEAT_LENGTH / MIT_BIH_FS)
            else:
                st.error(f"Unexpected shape {raw.shape}. Expected (N, 187) or (N,).")
                return
        except Exception as e:
            st.error(f"File load error: {e}")
            return
    else:
        st.warning("Select a data source and click Run Analysis.")
        return

    if beats is None or len(beats) == 0:
        st.error("No beats extracted. Check signal quality.")
        return
    with st.spinner(f"Running inference on {len(beats)} beats..."):
        try:
            from src.layer3_models.inference import predict_beats
            batch = predict_beats(beats, timestamps, loader, batch_size=64)
        except Exception as e:
            st.error(f"Inference error: {e}")
            return
    st.session_state.batch_result  = batch
    st.session_state.beats         = beats
    st.session_state.timestamps    = timestamps
    st.session_state.selected_beat = 0

    if st.session_state.db:
        try:
            st.session_state.db.save_session(
                st.session_state.patient_id, batch
            )
        except Exception:
            pass
    # Auto-fire alert if risk is Medium or above
    if batch.session_risk != "Low":
        try:
            from pathlib import Path
            from datetime import datetime
            alerts_dir = Path(r"C:\ecg_arrhythmia\outputs\alerts")
            alerts_dir.mkdir(parents=True, exist_ok=True)
            ts        = datetime.utcnow().isoformat()
            date_str  = datetime.utcnow().strftime("%Y-%m-%d")
            safe_pid  = str(st.session_state.patient_id).replace("/", "-")
            file_path = alerts_dir / f"alerts_{safe_pid}_{date_str}.txt"
            with open(str(file_path), "a", encoding="utf-8") as f:
                f.write(
                    f"[{ts}] ALERT\n"
                    f"  Patient   : {st.session_state.patient_id}\n"
                    f"  Risk      : {batch.session_risk}\n"
                    f"  Message   : {batch.anomaly_count} anomalies — "
                    f"{batch.anomaly_rate*100:.1f}% anomaly rate\n"
                    f"  Class     : {batch.dominant_class}\n"
                    f"  Disclaimer: AI-generated. Requires clinician review.\n"
                    f"{'─'*55}\n"
                )
        except Exception as e:
            pass
    st.success(f"Analysis complete — {batch.total_beats} beats, "
               f"{batch.anomaly_count} anomalies, risk: {batch.session_risk}")

def page_live_monitor():
    st.header("🫀 Live ECG Monitor")

    batch  = st.session_state.batch_result
    beats  = st.session_state.beats
    times  = st.session_state.timestamps
    render_stats_row(batch)
    st.markdown("---")
    if batch is None:
        st.info("Click **Run Analysis** in the sidebar to start.")
        return
    render_risk_badge(batch.session_risk)
    col_ecg, col_alert = st.columns([2, 1])
    with col_ecg:
        max_b = st.slider("Beats to display", 20, min(200, batch.total_beats), 80, step=10)
        fig   = plot_ecg_waveform(beats, times, batch, max_beats=max_b)
        st.plotly_chart(fig, use_container_width=True)
    with col_alert:
        render_alert_panel(batch)
    st.markdown("---")
    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.subheader("🔍 Beat Inspector")
        beat_idx = st.slider(
            "Select beat", 0, batch.total_beats - 1,
            st.session_state.selected_beat
        )
        st.session_state.selected_beat = beat_idx
        if beat_idx < len(beats):
            beat_r = batch.beats[beat_idx]
            st.plotly_chart(
                plot_single_beat(beats[beat_idx], beat_r),
                use_container_width=True,
            )
            st.markdown(f"""
| Field | Value |
|---|---|
| Beat Index | {beat_r.beat_index} |
| Timestamp | {beat_r.timestamp_sec:.2f}s |
| CNN Class | {beat_r.cnn_class_name} |
| Confidence | {beat_r.cnn_confidence*100:.1f}% |
| Transformer Z | {beat_r.transformer_score:.3f} |
| LSTM Error | {beat_r.lstm_error:.6f} |
| Is Anomaly | {'✅ Yes' if beat_r.is_anomaly else '❌ No'} |
| Risk | {beat_r.risk_level} |
""")

    with col_right:
        st.subheader("📊 Class Distribution")
        st.plotly_chart(plot_class_distribution(batch), use_container_width=True)

        st.subheader("📈 Class Probabilities")
        if beat_idx < len(batch.beats):
            st.plotly_chart(
                plot_confidence_bars(batch.beats[beat_idx]),
                use_container_width=True,
            )

def page_chat():
    st.header("💬 Clinician Chat")
    if not st.session_state.agent_ready:
        st.warning(
            "AI Agent not available. "
            "Check your GROQ_API_KEY in .env and internet connection."
        )
    if st.session_state.batch_result:
        render_quick_actions(
            st.session_state.agent,
            st.session_state.batch_result,
            st.session_state.patient_id,
        )
        st.markdown("---")
    render_chat(st.session_state.agent, st.session_state.patient_id)


def page_history():
    if st.session_state.db is None:
        st.error("Database not initialised.")
        return
    render_history_page(st.session_state.db, st.session_state.patient_id)

def page_report():
    st.header("📄 Session Report")
    batch = st.session_state.batch_result
    if batch is None:
        st.info("Run an analysis first to generate a report.")
        return

    st.subheader("Session Summary")
    st.markdown(f"""
| Metric | Value |
|---|---|
| Patient ID | {st.session_state.patient_id} |
| Total Beats | {batch.total_beats} |
| Anomaly Count | {batch.anomaly_count} |
| Anomaly Rate | {batch.anomaly_rate*100:.1f}% |
| Session Risk | {batch.session_risk} |
| Dominant Class | {batch.dominant_class} |
| Duration | {batch.recording_sec:.1f}s |
| Critical Beats | {len(batch.critical_beats())} |
""")

    if st.button("📄 Generate PDF Report", type="primary"):
        if st.session_state.agent:
            import json
            session_data = {
                "session_risk"  : batch.session_risk,
                "total_beats"   : batch.total_beats,
                "anomaly_count" : batch.anomaly_count,
                "anomaly_rate"  : batch.anomaly_rate,
                "recording_sec" : batch.recording_sec,
                "dominant_class": batch.dominant_class,
            }
            result = st.session_state.agent.run_tool_directly(
                "generate_report",
                json.dumps({"patient_id": st.session_state.patient_id,
                            "session_data": session_data})
            )
            if "saved" in result.lower():
                st.success(result)
                # Offer download
                report_path = result.replace("Report saved: ", "").strip()
                try:
                    with open(report_path, "rb") as f:
                        st.download_button(
                            label     = "⬇️ Download PDF",
                            data      = f.read(),
                            file_name = Path(report_path).name,
                            mime      = "application/pdf",
                        )
                except Exception:
                    pass
            else:
                st.error(result)
        else:
            st.warning("Agent not available — PDF generation requires the agent.")

def main():
    ensure_resources()

    data_source, uploaded, run_pressed = render_sidebar()
    if run_pressed:
        run_analysis(data_source, uploaded)

    page = st.tabs(["🫀 Live Monitor", "💬 Clinician Chat", "📊 Patient History", "📄 Report"])
    with page[0]:
        page_live_monitor()
    with page[1]:
        page_chat()
    with page[2]:
        page_history()
    with page[3]:
        page_report()

if __name__ == "__main__":
    main()