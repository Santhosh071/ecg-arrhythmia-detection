import streamlit as st

def init_chat_state():
    """Initialise session state keys for chat."""
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "agent" not in st.session_state:
        st.session_state.agent = None

def render_chat(agent, patient_id: str = None):
    """
    Full chat interface — message history + input box.

    Parameters
    ----------
    agent      : ECGAgent (already started)
    patient_id : optional patient ID for context
    """
    init_chat_state()
    st.subheader("💬 Clinician Chat")
    st.caption("Ask questions about ECG findings, arrhythmia classes, or patient history.")
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    user_input = st.chat_input("Ask about ECG findings, risk, patient history...")
    if user_input:
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    if agent is None:
                        response = (
                            "Agent not initialised. "
                            "Check GROQ_API_KEY in your .env file and internet connection."
                        )
                    else:
                        response = agent.run(user_input, patient_id=patient_id)
                except Exception as e:
                    response = f"Agent error: {e}"
            st.markdown(response)
            st.session_state.chat_messages.append(
                {"role": "assistant", "content": response}
            )
    with st.sidebar:
        st.markdown("---")
        st.subheader("Chat Controls")
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_messages = []
            if agent:
                agent.reset_memory()
            st.rerun()
        st.caption("Chat memory: last 10 messages")

def render_quick_actions(agent, batch_result, patient_id: str = None):
    """
    One-click quick action buttons that trigger common agent queries.
    Placed below the chat interface.
    """
    if batch_result is None:
        return
    st.markdown("**Quick Actions**")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📋 Explain Finding"):
            if batch_result.anomaly_beats():
                top = batch_result.anomaly_beats()[0]
                query = (
                    f"Explain this ECG finding: {top.cnn_class_name} "
                    f"detected with {top.cnn_confidence*100:.1f}% confidence. "
                    f"Risk level is {top.risk_level}."
                )
                _fire_quick_query(agent, query, patient_id)
    with col2:
        if st.button("📊 Assess Risk"):
            query = (
                f"Assess the risk for a session with "
                f"{batch_result.anomaly_rate*100:.1f}% anomaly rate, "
                f"dominant class {batch_result.dominant_class}, "
                f"{len(batch_result.critical_beats())} critical beats."
            )
            _fire_quick_query(agent, query, patient_id)
    with col3:
        if st.button("📜 Patient History"):
            query = f"Show me the history for patient {patient_id or 'current patient'}."
            _fire_quick_query(agent, query, patient_id)

def _fire_quick_query(agent, query: str, patient_id: str):
    """Send a quick action query to the agent and append to chat."""
    init_chat_state()
    st.session_state.chat_messages.append({"role": "user", "content": query})
    try:
        response = agent.run(query, patient_id=patient_id) if agent else "Agent not ready."
    except Exception as e:
        response = f"Error: {e}"
    st.session_state.chat_messages.append({"role": "assistant", "content": response})
    st.rerun()