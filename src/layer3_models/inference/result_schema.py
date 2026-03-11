import time
from dataclasses import dataclass, field, asdict

CLASS_NAMES = {
    0: "Normal (N)",
    1: "Left Bundle Branch Block (L)",
    2: "Right Bundle Branch Block (R)",
    3: "Atrial Premature Beat (A)",
    4: "Premature Ventricular Contraction (V)",
    5: "Paced Beat (/)",
    6: "Ventricular Escape Beat (E)",
    7: "Fusion Beat (F)",
}
SHORT_NAMES = {0: "N", 1: "L", 2: "R", 3: "A", 4: "V", 5: "/", 6: "E", 7: "F"}
CLASS_RISK = {0: "Low", 1: "Medium", 2: "Medium", 3: "Medium", 4: "High", 5: "Low", 6: "High", 7: "High",}
CLASS_ALERT_COLOR = {0: "green", 1: "yellow", 2: "yellow", 3: "yellow", 4: "orange", 5: "green", 6: "red", 7: "orange",}

@dataclass
class BeatResult:
    beat_index          : int
    timestamp_sec       : float
    is_anomaly          : bool
    transformer_anomaly : bool
    lstm_anomaly        : bool
    transformer_score   : float
    lstm_score          : float
    transformer_error   : float
    lstm_error          : float
    cnn_class_id        : int
    cnn_class_name      : str
    cnn_short_name      : str
    cnn_confidence      : float
    cnn_all_probs       : list
    risk_level          : str
    alert_color         : str
    timestamp_created   : float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    def is_critical(self) -> bool:
        return (
            self.transformer_anomaly
            and self.lstm_anomaly
            and self.cnn_class_id != 0
            and self.risk_level in ("High", "Critical")
        )

    def summary_line(self) -> str:
        flag = "ANOMALY" if self.is_anomaly else "Normal "
        return (
            f"[Beat {self.beat_index:05d} | {self.timestamp_sec:7.2f}s] "
            f"{flag} | CNN: {self.cnn_short_name} ({self.cnn_confidence*100:.1f}%) "
            f"| Z={self.transformer_score:.2f} | Risk={self.risk_level}"
        )

@dataclass
class BatchResult:
    beats          : list
    total_beats    : int
    anomaly_count  : int
    anomaly_rate   : float
    class_counts   : dict
    dominant_class : str
    session_risk   : str
    recording_sec  : float

    def to_dict(self) -> dict:
        d = asdict(self)
        d["beats"] = [b.to_dict() for b in self.beats]
        return d

    def anomaly_beats(self) -> list:
        return [b for b in self.beats if b.is_anomaly]

    def critical_beats(self) -> list:
        return [b for b in self.beats if b.is_critical()]

    def summary(self) -> str:
        return (
            f"Total beats    : {self.total_beats}\n"
            f"Anomalies      : {self.anomaly_count} ({self.anomaly_rate*100:.1f}%)\n"
            f"Critical beats : {len(self.critical_beats())}\n"
            f"Session risk   : {self.session_risk}\n"
            f"Dominant class : {self.dominant_class}\n"
            f"Duration       : {self.recording_sec:.1f}s"
        )

def compute_session_risk(anomaly_rate: float, beat_results: list) -> str:
    if any(b.is_critical() for b in beat_results):
        return "Critical"
    if anomaly_rate >= 0.30:
        return "High"
    if anomaly_rate >= 0.10:
        return "Medium"
    return "Low"