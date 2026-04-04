from pydantic import BaseModel, Field

from api.schemas import BeatResultResponse


class IoTStreamRequest(BaseModel):
    device_id: str = Field(..., description="ESP32 device identifier")
    patient_id: str = Field(..., description="Patient identifier associated with the stream")
    sampling_rate: float = Field(250.0, description="Raw device sampling rate in Hz")
    samples: list[float] = Field(..., description="Sequential raw ECG samples")
    start_timestamp_ms: int | None = Field(None, description="Unix epoch timestamp in milliseconds for the first sample")
    lead_off: bool = Field(False, description="True when the AD8232 lead-off indicator is active")
    adc_resolution: int = Field(4095, description="ADC resolution used by the ESP32 sampling firmware")
    vref: float = Field(3.3, description="Reference voltage used for ADC conversion")


class IoTStreamResponse(BaseModel):
    device_id: str
    patient_id: str
    samples_received: int
    buffered_samples: int
    buffered_seconds: float
    lead_off: bool
    last_seen_ms: int
    total_samples_received: int


class IoTStatusResponse(BaseModel):
    device_id: str
    patient_id: str
    sampling_rate: float
    buffered_samples: int
    buffered_seconds: float
    lead_off: bool
    last_seen_ms: int
    last_seen_delta_ms: int | None
    total_samples_received: int
    is_streaming: bool


class IoTLiveWindowResponse(BaseModel):
    device_id: str
    patient_id: str
    sampling_rate: float
    sample_count: int
    duration_sec: float
    lead_off: bool
    last_seen_ms: int
    samples: list[float]
    timestamps_ms: list[int]


class IoTAnalyzeRequest(BaseModel):
    device_id: str = Field(..., description="ESP32 device identifier")
    patient_id: str | None = Field(None, description="Optional patient id override")
    window_sec: float = Field(12.0, description="Buffered duration to analyze")
    save_session: bool = Field(True, description="Persist results to the patient history database")
    adc_resolution: int = Field(4095, description="ADC resolution used by the ESP32 sampling firmware")
    vref: float = Field(3.3, description="Reference voltage used for ADC conversion")


class IoTAnalyzeResponse(BaseModel):
    device_id: str
    patient_id: str
    source: str
    raw_sampling_rate: float
    model_sampling_rate: float
    buffered_duration_sec: float
    total_beats: int
    anomaly_count: int
    anomaly_rate: float
    class_counts: dict
    dominant_class: str
    session_risk: str
    recording_sec: float
    detected_peaks: int
    session_saved: bool = False
    quality: dict
    beats: list[BeatResultResponse]
    disclaimer: str = (
        "AI-generated output for clinical decision support only. "
        "Requires clinician review before any action."
    )
