"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

import { AIExplanationPanel } from "@/components/ai-explanation-panel";
import { DistributionChart } from "@/components/charts/distribution-chart";
import { WaveformChart } from "@/components/charts/waveform-chart";
import { PredictionWorkbench } from "@/components/prediction-workbench";
import { ReportPanel } from "@/components/report-panel";
import { SessionControls } from "@/components/session-controls";
import {
  fetchAlerts,
  fetchHealth,
  fetchHistory,
  fetchTrend,
  type AlertsResponse,
  type HealthResponse,
  type HistoryResponse,
  type PredictResponse,
  type TrendResponse,
} from "@/lib/api";

function riskClassName(riskLevel: string) {
  const value = riskLevel.toLowerCase();
  if (value === "low") return "pill pill-low";
  if (value === "medium") return "pill pill-medium";
  return "pill pill-high";
}

export function DashboardShell() {
  const [activePatientId, setActivePatientId] = useState("P001");
  const [patientInput, setPatientInput] = useState("P001");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [history, setHistory] = useState<HistoryResponse | null>(null);
  const [alerts, setAlerts] = useState<AlertsResponse | null>(null);
  const [trend, setTrend] = useState<TrendResponse | null>(null);
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [uploadedBeats, setUploadedBeats] = useState<number[][]>([]);
  const [uploadedFileName, setUploadedFileName] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  async function hydrate(patientId: string) {
    setLoading(true);
    setError("");
    try {
      const [healthResponse, historyResponse, alertsResponse, trendResponse] = await Promise.all([
        fetchHealth(),
        fetchHistory(patientId),
        fetchAlerts(patientId),
        fetchTrend(patientId),
      ]);
      setHealth(healthResponse);
      setHistory(historyResponse);
      setAlerts(alertsResponse);
      setTrend(trendResponse);
    } catch (fetchError) {
      setError(fetchError instanceof Error ? fetchError.message : "Failed to load dashboard.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void hydrate(activePatientId);
  }, [activePatientId]);

  const latestSession = useMemo(() => {
    if (prediction) {
      return {
        risk_level: prediction.session_risk,
        anomaly_count: prediction.anomaly_count,
        anomaly_rate: prediction.anomaly_rate * 100,
        dominant_class: prediction.dominant_class,
      };
    }
    return history?.sessions?.[0] ?? null;
  }, [history?.sessions, prediction]);

  return (
    <main className="shell">
      <section className="toolbar-row">
        <div>
          <h1 className="page-title">ECG Arrhythmia Monitor</h1>
          <div className="subtle">Protected clinician workspace for ECG review, prediction, history, alerts, reports, and AI support.</div>
        </div>
        <div className="stack toolbar-actions">
          <div className="nav-row">
            <Link href="/" className="nav-chip">Dashboard</Link>
            <Link href="/history" className="nav-chip">History</Link>
            <Link href="/alerts" className="nav-chip">Alerts</Link>
            <Link href={`/patients/${activePatientId}`} className="nav-chip">Patient</Link>
          </div>
          <SessionControls />
        </div>
      </section>

      {error && <div className="error-box">{error}</div>}

      <section className="grid dashboard-grid">
        <div className="stack">
          <div className="panel section-card">
            <div className="header-row">
              <div>
                <h2>Patient Overview</h2>
                <div className="subtle">Current patient context and latest session summary.</div>
              </div>
              <span className={riskClassName(latestSession?.risk_level ?? "high")}>
                {latestSession?.risk_level ?? "Pending"}
              </span>
            </div>

            <div className="metric-list">
              <div className="metric"><span>Patient ID</span><strong>{activePatientId}</strong></div>
              <div className="metric"><span>Latest Anomalies</span><strong>{latestSession?.anomaly_count ?? "--"}</strong></div>
              <div className="metric"><span>Anomaly Rate</span><strong>{latestSession ? `${Number(latestSession.anomaly_rate).toFixed(1)}%` : "--"}</strong></div>
              <div className="metric"><span>Dominant Class</span><strong>{latestSession?.dominant_class ?? "--"}</strong></div>
            </div>
          </div>

          <PredictionWorkbench
            patientId={activePatientId}
            onPatientIdCommit={(patientId) => {
              setActivePatientId(patientId);
              setPatientInput(patientId);
            }}
            onPredictionComplete={async ({ result, beats, fileName, patientId }) => {
              setPrediction(result);
              setUploadedBeats(beats);
              setUploadedFileName(fileName);
              setActivePatientId(patientId);
              setPatientInput(patientId);
              await hydrate(patientId);
            }}
          />

          <div className="panel section-card">
            <div className="header-row">
              <div>
                <h3>Waveform Preview</h3>
                <div className="subtle">Live input strip from the uploaded beat array, with anomaly markers and beat summaries.</div>
              </div>
              <span className="pill pill-low">{uploadedFileName || "No file"}</span>
            </div>
            <WaveformChart beats={uploadedBeats} results={prediction?.beats ?? []} />
          </div>

          <div className="panel section-card">
            <div className="header-row">
              <div>
                <h3>Class / Anomaly Distribution</h3>
                <div className="subtle">Simple distribution bars from the current prediction.</div>
              </div>
            </div>
            <DistributionChart result={prediction} />
          </div>
        </div>

        <div className="stack">
          <div className="panel section-card">
            <div className="header-row">
              <div>
                <h3>Patient Search</h3>
                <div className="subtle">Load another patient history and alerts.</div>
              </div>
              {loading && <span className="pill pill-low">Loading</span>}
            </div>
            <div className="input-row">
              <input value={patientInput} onChange={(event) => setPatientInput(event.target.value)} aria-label="Patient ID" />
              <button type="button" onClick={() => setActivePatientId(patientInput || "P001")}>Open</button>
            </div>
          </div>

          <div className="status-grid compact-grid">
            <div className="status-card">
              <h3>API</h3>
              <div className="status-value">{health?.status ?? "--"}</div>
            </div>
            <div className="status-card">
              <h3>Models</h3>
              <div className="status-value">{health?.models_loaded ? "Ready" : "Offline"}</div>
            </div>
            <div className="status-card">
              <h3>Agent</h3>
              <div className="status-value">{health?.agent_ready ? "Available" : "Offline"}</div>
            </div>
            <div className="status-card">
              <h3>Trend</h3>
              <div className="status-value">{trend?.trend ?? "--"}</div>
            </div>
          </div>

          <AIExplanationPanel patientId={activePatientId} result={prediction} />
          <ReportPanel patientId={activePatientId} result={prediction} />

          <div className="panel section-card">
            <div className="header-row">
              <div>
                <h3>Recent History</h3>
                <div className="subtle">Auto-refreshes after a successful prediction.</div>
              </div>
            </div>
            <div className="history-list">
              {(history?.sessions ?? []).slice(0, 5).map((session) => (
                <div className="history-item" key={session.session_id}>
                  <div className="header-row">
                    <strong>{session.timestamp.slice(0, 10)}</strong>
                    <span className={riskClassName(session.risk_level)}>{session.risk_level}</span>
                  </div>
                  <div className="subtle">
                    {session.anomaly_count} anomalies - {session.anomaly_rate.toFixed(1)}% - {session.recording_sec}s
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="panel section-card">
            <div className="header-row">
              <div>
                <h3>Alert Center</h3>
                <div className="subtle">Recent alerts for the active patient.</div>
              </div>
            </div>
            <div className="alert-list">
              {(alerts?.alerts ?? []).slice(0, 4).map((alert) => (
                <div className="alert-item" key={alert.alert_id}>
                  <div className="header-row">
                    <strong>{alert.timestamp.slice(0, 16).replace("T", " ")}</strong>
                    <span className={riskClassName(alert.risk_level)}>{alert.risk_level}</span>
                  </div>
                  <div className="subtle">{alert.message}</div>
                </div>
              ))}
              {!alerts?.alerts?.length && <div className="chart-empty">No alerts for this patient yet.</div>}
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}


