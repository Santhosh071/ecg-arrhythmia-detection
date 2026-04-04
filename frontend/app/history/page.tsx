"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { ProtectedRoute } from "@/components/protected-route";
import { SessionControls } from "@/components/session-controls";
import { fetchHistory, fetchTrend, type HistoryResponse, type TrendResponse } from "@/lib/api";

function HistoryWorkspace() {
  const [patientId, setPatientId] = useState("P001");
  const [patientInput, setPatientInput] = useState("P001");
  const [history, setHistory] = useState<HistoryResponse | null>(null);
  const [trend, setTrend] = useState<TrendResponse | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    async function load() {
      try {
        setError("");
        const [historyResponse, trendResponse] = await Promise.all([
          fetchHistory(patientId),
          fetchTrend(patientId),
        ]);
        setHistory(historyResponse);
        setTrend(trendResponse);
      } catch (fetchError) {
        setError(fetchError instanceof Error ? fetchError.message : "Failed to load history.");
      }
    }

    void load();
  }, [patientId]);

  return (
    <main className="shell">
      <section className="toolbar-row">
        <div>
          <h1 className="page-title">Patient History</h1>
          <div className="subtle">Trend and session history for a selected patient.</div>
        </div>
        <div className="stack toolbar-actions">
          <div className="nav-row">
            <Link href="/" className="nav-chip">Dashboard</Link>
            <Link href="/history" className="nav-chip">History</Link>
            <Link href="/alerts" className="nav-chip">Alerts</Link>
          </div>
          <SessionControls />
        </div>
      </section>

      {error && <div className="error-box">{error}</div>}

      <section className="grid dashboard-grid">
        <div className="stack">
          <div className="panel section-card">
            <h3>Patient Search</h3>
            <div className="input-row">
              <input value={patientInput} onChange={(event) => setPatientInput(event.target.value)} />
              <button type="button" onClick={() => setPatientId(patientInput.trim() || "P001")}>Open</button>
            </div>
          </div>

          <div className="panel section-card">
            <h3>Trend Summary</h3>
            <div className="metric-list">
              <div className="metric"><span>Trend</span><strong>{trend?.trend ?? "--"}</strong></div>
              <div className="metric"><span>Sessions</span><strong>{trend?.sessions ?? "--"}</strong></div>
              <div className="metric"><span>Average Anomaly Rate</span><strong>{typeof trend?.avg_anomaly_rate === "number" ? `${trend.avg_anomaly_rate.toFixed(1)}%` : "--"}</strong></div>
              <div className="metric"><span>Latest Rate</span><strong>{typeof trend?.latest_rate === "number" ? `${trend.latest_rate.toFixed(1)}%` : "--"}</strong></div>
            </div>
          </div>
        </div>

        <div className="panel section-card">
          <h3>Session Timeline</h3>
          <div className="history-list">
            {(history?.sessions ?? []).map((session) => (
              <div className="history-item" key={session.session_id}>
                <div className="header-row">
                  <strong>{session.timestamp.replace("T", " ").slice(0, 16)}</strong>
                  <span className="pill pill-medium">{session.risk_level}</span>
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
    </main>
  );
}

export default function HistoryPage() {
  return (
    <ProtectedRoute>
      <HistoryWorkspace />
    </ProtectedRoute>
  );
}
