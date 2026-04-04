"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

import { AIExplanationPanel } from "@/components/ai-explanation-panel";
import { DistributionChart } from "@/components/charts/distribution-chart";
import { WaveformChart } from "@/components/charts/waveform-chart";
import { PredictionWorkbench } from "@/components/prediction-workbench";
import { ProtectedRoute } from "@/components/protected-route";
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

type PatientWorkspaceProps = {
  patientId: string;
};

const TABS = ["overview", "waveform", "history", "alerts", "reports", "ai"] as const;
type WorkspaceTab = (typeof TABS)[number];

function riskClassName(riskLevel: string) {
  const value = riskLevel.toLowerCase();
  if (value === "low") return "pill pill-low";
  if (value === "medium") return "pill pill-medium";
  return "pill pill-high";
}

function titleCase(value: string) {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

export function PatientWorkspace({ patientId }: PatientWorkspaceProps) {
  const [activeTab, setActiveTab] = useState<WorkspaceTab>("overview");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [history, setHistory] = useState<HistoryResponse | null>(null);
  const [alerts, setAlerts] = useState<AlertsResponse | null>(null);
  const [trend, setTrend] = useState<TrendResponse | null>(null);
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [uploadedBeats, setUploadedBeats] = useState<number[][]>([]);
  const [uploadedFileName, setUploadedFileName] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  async function hydrate() {
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
      setError(fetchError instanceof Error ? fetchError.message : "Failed to load patient workspace.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void hydrate();
  }, [patientId]);

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
    <ProtectedRoute>
      <main className="shell">
        <section className="toolbar-row">
          <div>
            <h1 className="page-title">Patient Workspace</h1>
            <div className="subtle">Dedicated patient review workspace for {patientId}.</div>
          </div>
          <div className="stack toolbar-actions">
            <div className="nav-row">
              <Link href="/" className="nav-chip">Dashboard</Link>
              <Link href="/history" className="nav-chip">History</Link>
              <Link href="/alerts" className="nav-chip">Alerts</Link>
              <Link href={`/patients/${patientId}`} className="nav-chip">Patient</Link>
            </div>
            <SessionControls />
          </div>
        </section>

        {error && <div className="error-box">{error}</div>}

        <section className="panel section-card">
          <div className="header-row">
            <div>
              <h2>{patientId}</h2>
              <div className="subtle">Workspace tabs organize overview, waveform, history, alerts, reports, and AI support for a single patient.</div>
            </div>
            <span className={riskClassName(latestSession?.risk_level ?? "high")}>
              {loading ? "Loading" : latestSession?.risk_level ?? "Pending"}
            </span>
          </div>
          <div className="chip-row top-gap-small">
            {TABS.map((tab) => (
              <button key={tab} type="button" className="secondary-chip" onClick={() => setActiveTab(tab)}>
                {titleCase(tab)}
              </button>
            ))}
          </div>
        </section>

        {activeTab === "overview" && (
          <section className="grid dashboard-grid top-gap">
            <div className="panel section-card">
              <h3>Current Summary</h3>
              <div className="metric-list">
                <div className="metric"><span>Patient ID</span><strong>{patientId}</strong></div>
                <div className="metric"><span>Latest Anomalies</span><strong>{latestSession?.anomaly_count ?? "--"}</strong></div>
                <div className="metric"><span>Anomaly Rate</span><strong>{latestSession ? `${Number(latestSession.anomaly_rate).toFixed(1)}%` : "--"}</strong></div>
                <div className="metric"><span>Dominant Class</span><strong>{latestSession?.dominant_class ?? "--"}</strong></div>
              </div>
            </div>

            <div className="panel section-card">
              <h3>System Readiness</h3>
              <div className="metric-list">
                <div className="metric"><span>API</span><strong>{health?.status ?? "--"}</strong></div>
                <div className="metric"><span>Models</span><strong>{health?.models_loaded ? "Ready" : "Offline"}</strong></div>
                <div className="metric"><span>Agent</span><strong>{health?.agent_ready ? "Available" : "Offline"}</strong></div>
                <div className="metric"><span>Trend</span><strong>{trend?.trend ?? "--"}</strong></div>
              </div>
            </div>
          </section>
        )}

        {activeTab === "waveform" && (
          <section className="stack top-gap">
            <PredictionWorkbench
              patientId={patientId}
              onPatientIdCommit={() => undefined}
              onPredictionComplete={async ({ result, beats, fileName }) => {
                setPrediction(result);
                setUploadedBeats(beats);
                setUploadedFileName(fileName);
                await hydrate();
              }}
            />
            <div className="panel section-card">
              <div className="header-row">
                <div>
                  <h3>Waveform Preview</h3>
                  <div className="subtle">Live beat strip and summaries for the current uploaded session.</div>
                </div>
                <span className="pill pill-low">{uploadedFileName || "No file"}</span>
              </div>
              <WaveformChart beats={uploadedBeats} results={prediction?.beats ?? []} />
            </div>
            <div className="panel section-card">
              <h3>Class / Anomaly Distribution</h3>
              <DistributionChart result={prediction} />
            </div>
          </section>
        )}

        {activeTab === "history" && (
          <section className="grid dashboard-grid top-gap">
            <div className="panel section-card">
              <h3>Trend Summary</h3>
              <div className="metric-list">
                <div className="metric"><span>Trend</span><strong>{trend?.trend ?? "--"}</strong></div>
                <div className="metric"><span>Sessions</span><strong>{trend?.sessions ?? "--"}</strong></div>
                <div className="metric"><span>Average Anomaly Rate</span><strong>{typeof trend?.avg_anomaly_rate === "number" ? `${trend.avg_anomaly_rate.toFixed(1)}%` : "--"}</strong></div>
                <div className="metric"><span>Latest Rate</span><strong>{typeof trend?.latest_rate === "number" ? `${trend.latest_rate.toFixed(1)}%` : "--"}</strong></div>
              </div>
            </div>
            <div className="panel section-card">
              <h3>Session Timeline</h3>
              <div className="history-list">
                {(history?.sessions ?? []).map((session) => (
                  <div className="history-item" key={session.session_id}>
                    <div className="header-row">
                      <strong>{session.timestamp.replace("T", " ").slice(0, 16)}</strong>
                      <span className={riskClassName(session.risk_level)}>{session.risk_level}</span>
                    </div>
                    <div className="subtle">
                      Beats: {session.total_beats} - Anomalies: {session.anomaly_count} - Rate: {session.anomaly_rate.toFixed(1)}% - Class: {session.dominant_class}
                    </div>
                  </div>
                ))}
                {!history?.sessions?.length && <div className="chart-empty">No sessions found for this patient.</div>}
              </div>
            </div>
          </section>
        )}

        {activeTab === "alerts" && (
          <section className="panel section-card top-gap">
            <h3>Recent Alerts</h3>
            <div className="alert-list top-gap-small">
              {(alerts?.alerts ?? []).map((alert) => (
                <div className="alert-item" key={alert.alert_id}>
                  <div className="header-row">
                    <strong>{alert.timestamp.replace("T", " ").slice(0, 16)}</strong>
                    <span className={riskClassName(alert.risk_level)}>{alert.review_status ? titleCase(alert.review_status) : alert.risk_level}</span>
                  </div>
                  <div className="subtle">{alert.message}</div>
                  <div className="subtle top-gap-small">Reviewer: {alert.reviewed_by ?? "Pending"} • Reviewed at: {alert.reviewed_at ? alert.reviewed_at.replace("T", " ").slice(0, 16) : "Pending"}</div>
                </div>
              ))}
              {!alerts?.alerts?.length && <div className="chart-empty">No alerts for this patient yet.</div>}
            </div>
          </section>
        )}

        {activeTab === "reports" && (
          <section className="top-gap">
            <ReportPanel patientId={patientId} result={prediction} />
          </section>
        )}

        {activeTab === "ai" && (
          <section className="top-gap">
            <AIExplanationPanel patientId={patientId} result={prediction} />
          </section>
        )}
      </main>
    </ProtectedRoute>
  );
}
