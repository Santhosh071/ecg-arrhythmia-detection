from typing import Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    patient_id   : str               = Field(...,   description="Patient identifier")
    beats        : list[list[float]] = Field(...,   description="List of 187-sample beats")
    timestamps   : list[float]       = Field(...,   description="R-peak timestamps in seconds")
    fs           : float             = Field(360.0, description="Sampling rate Hz")
    save_session : bool              = Field(True,  description="Save result to DB")


class LoginRequest(BaseModel):
    username: str = Field(..., description="Clinician username")
    password: str = Field(..., description="Clinician password")


class UserResponse(BaseModel):
    username: str
    full_name: str
    role: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class SingleBeatRequest(BaseModel):
    patient_id : str         = Field(...,  description="Patient identifier")
    beat       : list[float] = Field(...,  description="Single 187-sample beat")
    beat_index : int         = Field(0,    description="Beat index in recording")
    timestamp  : float       = Field(0.0, description="R-peak time in seconds")


class AgentQueryRequest(BaseModel):
    question        : str            = Field(...,  description="Clinician question")
    patient_id      : Optional[str]  = Field(None, description="Patient ID for context")
    session_context : Optional[dict] = Field(
        None,
        description="Optional structured ECG session summary used to ground the response",
    )


class ReportRequest(BaseModel):
    patient_id   : str  = Field(..., description="Patient identifier")
    session_data : dict = Field(..., description="BatchResult summary dict")


class SignalQualityRequest(BaseModel):
    patient_id : str         = Field(...,   description="Patient identifier")
    signal     : list[float] = Field(...,   description="Raw 1D ECG signal")
    fs         : float       = Field(360.0, description="Sampling rate Hz")


class HealthResponse(BaseModel):
    status         : str
    models_loaded  : bool
    agent_ready    : bool
    db_connected   : bool
    groq_available : bool


class BeatResultResponse(BaseModel):
    beat_index          : int
    timestamp_sec       : float
    is_anomaly          : bool
    transformer_anomaly : bool
    lstm_anomaly        : bool
    transformer_score   : float
    lstm_error          : float
    cnn_class_id        : int
    cnn_class_name      : str
    cnn_short_name      : str
    cnn_confidence      : float
    cnn_all_probs       : list[float]
    risk_level          : str
    alert_color         : str


class BatchResultResponse(BaseModel):
    patient_id     : str
    total_beats    : int
    anomaly_count  : int
    anomaly_rate   : float
    class_counts   : dict
    dominant_class : str
    session_risk   : str
    recording_sec  : float
    beats          : list[BeatResultResponse]
    session_saved  : bool = False
    request_truncated : bool = False
    disclaimer     : str  = (
        "AI-generated output for clinical decision support only. "
        "Requires clinician review before any action."
    )


class AgentResponse(BaseModel):
    answer     : str
    patient_id : Optional[str] = None
    disclaimer : str = "AI-generated response. Not a substitute for clinical expertise."


class ReportResponse(BaseModel):
    success    : bool
    file_path  : str
    patient_id : str
    message    : str


class ReportRecordResponse(BaseModel):
    report_id: int
    session_id: Optional[int] = None
    timestamp: str
    file_path: str
    file_name: str
    report_focus: Optional[str] = None
    status: str
    summary: Optional[str] = None


class ReportsHistoryResponse(BaseModel):
    patient_id: str
    reports: list[ReportRecordResponse]
    total: int


class SignalQualityResponse(BaseModel):
    patient_id   : str
    is_good      : bool
    issues       : list[str]
    snr_db       : float
    duration_sec : float
    peak_count   : int
    mean_hr_bpm  : float


class HistoryResponse(BaseModel):
    patient_id : str
    sessions   : list[dict]
    total      : int


class AlertsResponse(BaseModel):
    patient_id : str
    alerts     : list[dict]
    total      : int


class AlertReviewRequest(BaseModel):
    patient_id: str = Field(..., description="Patient identifier")
    review_status: str = Field(..., description="new | reviewed | acknowledged | dismissed")
    reviewer_note: str = Field("", description="Optional clinician review note")


class AlertReviewResponse(BaseModel):
    patient_id: str
    alert: dict
