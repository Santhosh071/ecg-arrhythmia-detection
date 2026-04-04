"use client";

import { useEffect, useMemo, useState } from "react";

import { getStoredToken } from "@/lib/auth";

type IoTStatusResponse = {
  device_id: string;
  patient_id: string;
  sampling_rate: number;
  buffered_samples: number;
  buffered_seconds: number;
  lead_off: boolean;
  last_seen_ms: number;
  last_seen_delta_ms: number | null;
  total_samples_received: number;
  is_streaming: boolean;
};

type IoTLiveWindowResponse = {
  device_id: string;
  patient_id: string;
  sampling_rate: number;
  sample_count: number;
  duration_sec: number;
  lead_off: boolean;
  last_seen_ms: number;
  samples: number[];
  timestamps_ms: number[];
};

type IoTAnalyzeResponse = {
  device_id: string;
  patient_id: string;
  source: string;
  raw_sampling_rate: number;
  model_sampling_rate: number;
  buffered_duration_sec: number;
  total_beats: number;
  anomaly_count: number;
  anomaly_rate: number;
  class_counts: Record<string, number>;
  dominant_class: string;
  session_risk: string;
  recording_sec: number;
  detected_peaks: number;
  session_saved: boolean;
  quality: {
    is_good: boolean;
    issues: string[];
    snr_db: number;
    duration_sec: number;
    peak_count: number;
    mean_hr_bpm: number;
  };
};

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ?? "http://127.0.0.1:8000";

function buildHeaders() {
  const token = getStoredToken();
  return {
    "Content-Type": "application/json",
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
}

function riskClassName(riskLevel: string) {
  const value = riskLevel.toLowerCase();
  if (value === "low") return "pill pill-low";
  if (value === "medium") return "pill pill-medium";
  return "pill pill-high";
}

function buildPath(samples: number[]) {
  if (!samples.length) return "";
  const width = 760;
  const height = 220;
  const points = samples.map((sample, index) => {
    const x = (index / Math.max(samples.length - 1, 1)) * width;
    const y = height / 2 - sample * (height * 0.38);
    return `${index === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
  });
  return points.join(" ");
}

export function LiveEcgStream() {
  const [deviceId, setDeviceId] = useState("esp32-001");
  const [patientId, setPatientId] = useState("P001");
  const [status, setStatus] = useState<IoTStatusResponse | null>(null);
  const [windowData, setWindowData] = useState<IoTLiveWindowResponse | null>(null);
  const [analysis, setAnalysis] = useState<IoTAnalyzeResponse | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [polling, setPolling] = useState(true);

  async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
    const response = await fetch(`${API_BASE_URL}${path}`, {
      cache: "no-store",
      ...init,
      headers: {
        ...buildHeaders(),
        ...(init?.headers ?? {}),
      },
    });

    if (!response.ok) {
      let detail = `Request failed: ${response.status}`;
      try {
        const payload = (await response.json()) as { detail?: string };
        if (payload.detail) detail = payload.detail;
      } catch {
      }
      throw new Error(detail);
    }

    return (await response.json()) as T;
  }

  async function hydrate() {
    setLoading(true);
    try {
      setError("");
      const [statusResponse, liveResponse] = await Promise.all([
        fetchJson<IoTStatusResponse>(`/iot/status/${encodeURIComponent(deviceId)}`),
        fetchJson<IoTLiveWindowResponse>(`/iot/live/${encodeURIComponent(deviceId)}?window_sec=8`),
      ]);
      setStatus(statusResponse);
      setWindowData(liveResponse);
      setPatientId(statusResponse.patient_id);
    } catch (fetchError) {
      setError(fetchError instanceof Error ? fetchError.message : "Failed to load IoT monitor.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void hydrate();
  }, [deviceId]);

  useEffect(() => {
    if (!polling) return;
    const interval = window.setInterval(() => {
      void hydrate();
    }, 1500);
    return () => window.clearInterval(interval);
  }, [deviceId, polling]);

  const waveformPath = useMemo(() => buildPath(windowData?.samples ?? []), [windowData?.samples]);

  async function runAnalysis() {
    try {
      setError("");
      const response = await fetchJson<IoTAnalyzeResponse>("/iot/analyze", {
        method: "POST",
        body: JSON.stringify({
          device_id: deviceId,
          patient_id: patientId,
          window_sec: 12,
          save_session: true,
        }),
      });
      setAnalysis(response);
    } catch (analysisError) {
      setError(analysisError instanceof Error ? analysisError.message : "Failed to analyze stream.");
    }
  }

  return (
    <section className="grid dashboard-grid">
      <div className="stack">
        <div className="panel section-card">
          <div className="header-row">
            <div>
              <h2>Device Stream</h2>
              <div className="subtle">Phase 1 live ingestion for ESP32 + AD8232.</div>
            </div>
            <button type="button" onClick={() => setPolling((value) => !value)}>
              {polling ? "Pause Polling" : "Resume Polling"}
            </button>
          </div>
          <div className="input-row">
            <input value={deviceId} onChange={(event) => setDeviceId(event.target.value)} />
            <button type="button" onClick={() => void hydrate()}>Refresh</button>
            <button type="button" onClick={() => void runAnalysis()}>Analyze</button>
          </div>
          <div className="metric-list">
            <div className="metric"><span>Device</span><strong>{status?.device_id ?? "--"}</strong></div>
            <div className="metric"><span>Patient</span><strong>{status?.patient_id ?? "--"}</strong></div>
            <div className="metric"><span>Sampling Rate</span><strong>{status ? `${status.sampling_rate} Hz` : "--"}</strong></div>
            <div className="metric"><span>Buffered</span><strong>{status ? `${status.buffered_seconds.toFixed(1)} s` : "--"}</strong></div>
            <div className="metric"><span>Lead Status</span><strong>{status?.lead_off ? "Lead Off" : "Connected"}</strong></div>
            <div className="metric"><span>Streaming</span><strong>{status?.is_streaming ? "Live" : "Idle"}</strong></div>
          </div>
        </div>

        <div className="panel section-card">
          <div className="header-row">
            <div>
              <h3>Live Waveform</h3>
              <div className="subtle">Latest 8-second ECG window from the device buffer.</div>
            </div>
            <span className="pill pill-low">{windowData ? `${windowData.sample_count} samples` : "No samples"}</span>
          </div>
          <svg viewBox="0 0 760 220" role="img" aria-label="Live ECG waveform" style={{ width: "100%", height: "220px" }}>
            <rect x="0" y="0" width="760" height="220" rx="18" fill="rgba(8, 18, 33, 0.06)" />
            <path d={waveformPath} stroke="#0f766e" strokeWidth="2.5" fill="none" />
          </svg>
          {!waveformPath && <div className="chart-empty">No live samples received yet. Start the ESP32 stream and refresh this page.</div>}
        </div>
      </div>

      <div className="stack">
        <div className="panel section-card">
          <div className="header-row">
            <div>
              <h3>Phase 2 Analysis</h3>
              <div className="subtle">Runs preprocessing, beat segmentation, and the existing AI models on the live device buffer.</div>
            </div>
            <span className={riskClassName(analysis?.session_risk ?? "high")}>
              {analysis?.session_risk ?? "Pending"}
            </span>
          </div>
          <div className="metric-list">
            <div className="metric"><span>Total Beats</span><strong>{analysis?.total_beats ?? "--"}</strong></div>
            <div className="metric"><span>Anomalies</span><strong>{analysis?.anomaly_count ?? "--"}</strong></div>
            <div className="metric"><span>Anomaly Rate</span><strong>{typeof analysis?.anomaly_rate === "number" ? `${(analysis.anomaly_rate * 100).toFixed(1)}%` : "--"}</strong></div>
            <div className="metric"><span>Dominant Class</span><strong>{analysis?.dominant_class ?? "--"}</strong></div>
            <div className="metric"><span>Detected Peaks</span><strong>{analysis?.detected_peaks ?? "--"}</strong></div>
            <div className="metric"><span>Session Saved</span><strong>{analysis?.session_saved ? "Yes" : "No"}</strong></div>
          </div>
        </div>

        <div className="panel section-card">
          <h3>Signal Quality</h3>
          <div className="metric-list">
            <div className="metric"><span>Quality</span><strong>{analysis?.quality?.is_good ? "Good" : "Needs Review"}</strong></div>
            <div className="metric"><span>SNR</span><strong>{typeof analysis?.quality?.snr_db === "number" ? `${analysis.quality.snr_db} dB` : "--"}</strong></div>
            <div className="metric"><span>Estimated HR</span><strong>{typeof analysis?.quality?.mean_hr_bpm === "number" ? `${analysis.quality.mean_hr_bpm} bpm` : "--"}</strong></div>
          </div>
          {!!analysis?.quality?.issues?.length && (
            <div className="history-list">
              {analysis.quality.issues.map((issue) => (
                <div className="history-item" key={issue}>{issue}</div>
              ))}
            </div>
          )}
        </div>
      </div>

      {error && <div className="error-box">{error}</div>}
      {loading && <div className="subtle">Refreshing IoT monitor...</div>}
    </section>
  );
}
