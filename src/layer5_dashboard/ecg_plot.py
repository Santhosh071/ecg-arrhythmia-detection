import numpy as np
import plotly.graph_objects as go

COLOR_NORMAL  = "#2196F3"
COLOR_ANOMALY = "#F44336"
COLOR_BG      = "#0E1117"
COLOR_GRID    = "#2D2D2D"
COLOR_TEXT    = "#FAFAFA"
RISK_COLORS = {
    "Low"     : "#4CAF50",
    "Medium"  : "#FFC107",
    "High"    : "#FF9800",
    "Critical": "#F44336",
}

def _base_layout(title: str, height: int = 300, xlab: str = "", ylab: str = "") -> dict:
    return dict(
        title         = dict(text=title, font=dict(size=13, color=COLOR_TEXT)),
        paper_bgcolor = COLOR_BG,
        plot_bgcolor  = COLOR_BG,
        font          = dict(color=COLOR_TEXT),
        xaxis         = dict(title=xlab, gridcolor=COLOR_GRID, showgrid=True, zeroline=False),
        yaxis         = dict(title=ylab, gridcolor=COLOR_GRID, showgrid=True, zeroline=False),
        margin        = dict(l=50, r=20, t=45, b=45),
        height        = height,
    )

def _empty_figure(msg: str, height: int = 250) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(size=13, color=COLOR_TEXT))
    fig.update_layout(
        paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG, height=height,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig

def plot_ecg_waveform(
    beats       : np.ndarray,
    timestamps  : np.ndarray,
    batch_result,
    fs          : float = 360.0,
    title       : str   = "ECG Recording",
    max_beats   : int   = 80,
) -> go.Figure:
    """
    Full ECG waveform. Normal beats = blue, Anomaly beats = red.
    Anomaly markers shown as triangles above R-peaks.

    Parameters
    ----------
    beats        : (N, 187) normalised beat segments
    timestamps   : (N,) R-peak times in seconds
    batch_result : BatchResult
    max_beats    : render limit for performance (80 default)
    """
    if beats is None or len(beats) == 0:
        return _empty_figure("No ECG data loaded.")

    n       = min(len(beats), max_beats)
    results = batch_result.beats[:n] if batch_result else []
    fig     = go.Figure()

    for i in range(n):
        beat    = beats[i]
        t_start = float(timestamps[i]) if i < len(timestamps) else i * 187 / fs
        t_axis  = t_start + np.arange(187) / fs
        is_anom = results[i].is_anomaly if i < len(results) else False

        fig.add_trace(go.Scatter(
            x          = t_axis,
            y          = beat,
            mode       = "lines",
            line       = dict(color=COLOR_ANOMALY if is_anom else COLOR_NORMAL,
                              width=2.0 if is_anom else 1.2),
            showlegend = False,
            hoverinfo  = "skip",
        ))
    for color, name in [(COLOR_NORMAL, "Normal"), (COLOR_ANOMALY, "Anomaly")]:
        fig.add_trace(go.Scatter(x=[], y=[], mode="lines",
                                 line=dict(color=color, width=2), name=name))
    anom_x = [float(timestamps[i]) for i, r in enumerate(results)
               if r.is_anomaly and i < len(timestamps)]
    anom_t = [f"{r.cnn_short_name} {r.cnn_confidence*100:.0f}%"
               for i, r in enumerate(results) if r.is_anomaly and i < len(timestamps)]
    if anom_x:
        fig.add_trace(go.Scatter(
            x=anom_x, y=[1.3] * len(anom_x),
            mode="markers+text",
            marker=dict(symbol="triangle-down", size=9, color=COLOR_ANOMALY),
            text=anom_t, textposition="top center",
            textfont=dict(size=8, color=COLOR_ANOMALY),
            showlegend=False, hoverinfo="skip",
        ))
    risk    = batch_result.session_risk if batch_result else "N/A"
    anom_pc = batch_result.anomaly_rate * 100 if batch_result else 0
    layout  = _base_layout(
        f"{title}  |  Risk: {risk}  |  Anomalies: {anom_pc:.1f}%",
        height=360, xlab="Time (s)", ylab="Amplitude (norm.)"
    )
    layout["legend"] = dict(bgcolor="#1A1A1A", bordercolor=COLOR_GRID, borderwidth=1)
    fig.update_layout(**layout)
    return fig

def plot_single_beat(beat: np.ndarray, beat_result, fs: float = 360.0) -> go.Figure:
    """
    Single 187-sample beat plot with class label + confidence.

    Parameters
    ----------
    beat        : (187,) array
    beat_result : BeatResult
    """
    if beat is None or len(beat) != 187:
        return _empty_figure("Select a beat to inspect.", 200)

    t     = np.arange(187) / fs * 1000
    color = COLOR_ANOMALY if beat_result.is_anomaly else COLOR_NORMAL
    lbl   = (f"{beat_result.cnn_class_name}  |  "
             f"Conf: {beat_result.cnn_confidence*100:.1f}%  |  "
             f"Risk: {beat_result.risk_level}")
    fig = go.Figure(go.Scatter(x=t, y=beat, mode="lines",
                               line=dict(color=color, width=2)))
    fig.update_layout(**_base_layout(lbl, height=220, xlab="Time (ms)", ylab="Amplitude"))
    return fig

def plot_class_distribution(batch_result) -> go.Figure:
    """Horizontal bar chart of CNN class distribution."""
    if not batch_result or not batch_result.class_counts:
        return _empty_figure("No class data.")
    from src.layer3_models.inference import SHORT_NAMES
    keys    = sorted(batch_result.class_counts)
    labels  = [SHORT_NAMES.get(k, str(k)) for k in keys]
    counts  = [batch_result.class_counts[k] for k in keys]
    bcolors = [COLOR_ANOMALY if k != 0 else COLOR_NORMAL for k in keys]
    fig = go.Figure(go.Bar(
        x=labels, y=counts,
        marker_color=bcolors,
        text=counts, textposition="outside",
    ))
    fig.update_layout(**_base_layout("Beat Class Distribution",
                                     height=250, xlab="Class", ylab="Count"))
    return fig

def plot_confidence_bars(beat_result) -> go.Figure:
    """Horizontal bar showing softmax probabilities for all 8 classes."""
    if not beat_result:
        return _empty_figure("No beat selected.", 200)
    from src.layer3_models.inference import SHORT_NAMES
    labels = [SHORT_NAMES.get(i, str(i)) for i in range(8)]
    probs  = [p * 100 for p in beat_result.cnn_all_probs]
    bcolors= [COLOR_ANOMALY if i != 0 else COLOR_NORMAL for i in range(8)]
    fig = go.Figure(go.Bar(
        x=labels, y=probs,
        marker_color=bcolors,
        text=[f"{p:.1f}%" for p in probs],
        textposition="outside",
    ))
    fig.update_layout(**_base_layout(
        f"Class Probabilities — Predicted: {beat_result.cnn_short_name}",
        height=220, xlab="Class", ylab="Probability (%)"
    ))
    fig.update_yaxes(range=[0, 110])
    return fig

def plot_anomaly_trend(history: list) -> go.Figure:
    """
    Line chart of anomaly rate across past sessions.
    Points colour-coded by risk level.
    """
    if not history:
        return _empty_figure("No history available.")
    history_rev  = list(reversed(history))
    dates        = [h["timestamp"][:10] for h in history_rev]
    rates        = [h["anomaly_rate"]    for h in history_rev]
    point_colors = [RISK_COLORS.get(h["risk_level"], COLOR_NORMAL) for h in history_rev]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=rates,
        mode="lines+markers",
        line=dict(color="#7C4DFF", width=2),
        marker=dict(color=point_colors, size=10,
                    line=dict(width=1, color="#fff")),
        hovertemplate="%{x}<br>Rate: %{y:.1f}%<extra></extra>",
        name="Anomaly %",
    ))
    fig.add_hline(y=10, line_dash="dot", line_color="#FFC107",
                  annotation_text="Medium (10%)", annotation_font_color="#FFC107")
    fig.add_hline(y=30, line_dash="dot", line_color=COLOR_ANOMALY,
                  annotation_text="High (30%)", annotation_font_color=COLOR_ANOMALY)

    fig.update_layout(**_base_layout("Anomaly Rate — Session History",
                                     height=280, xlab="Date", ylab="Anomaly Rate (%)"))
    return fig