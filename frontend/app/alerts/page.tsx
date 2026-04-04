"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { ProtectedRoute } from "@/components/protected-route";
import { SessionControls } from "@/components/session-controls";
import { fetchAlerts, reviewAlert, type AlertRecord, type AlertsResponse } from "@/lib/api";
import { getStoredSession } from "@/lib/auth";

const REVIEW_OPTIONS = [
  { value: "reviewed", label: "Reviewed" },
  { value: "acknowledged", label: "Acknowledged" },
  { value: "dismissed", label: "Dismissed" },
];

function reviewPillClass(status?: string, riskLevel?: string) {
  const value = (status ?? "new").toLowerCase();
  if (value === "reviewed") return "pill pill-low";
  if (value === "acknowledged") return "pill pill-medium";
  if (value === "dismissed") return "pill pill-low";
  if ((riskLevel ?? "").toLowerCase() === "medium") return "pill pill-medium";
  return "pill pill-high";
}

function reviewLabel(alert: AlertRecord) {
  const status = (alert.review_status ?? "new").toLowerCase();
  if (status === "new") return alert.risk_level;
  return status.charAt(0).toUpperCase() + status.slice(1);
}

function AlertsWorkspace() {
  const [patientId, setPatientId] = useState("P001");
  const [patientInput, setPatientInput] = useState("P001");
  const [alerts, setAlerts] = useState<AlertsResponse | null>(null);
  const [reviewNotes, setReviewNotes] = useState<Record<number, string>>({});
  const [savingId, setSavingId] = useState<number | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    async function load() {
      try {
        setError("");
        const response = await fetchAlerts(patientId);
        setAlerts(response);
        setReviewNotes(
          Object.fromEntries(
            response.alerts.map((alert) => [alert.alert_id, alert.reviewer_note ?? ""]),
          ),
        );
      } catch (fetchError) {
        setError(fetchError instanceof Error ? fetchError.message : "Failed to load alerts.");
      }
    }

    void load();
  }, [patientId]);

  async function onReview(alert: AlertRecord, reviewStatus: string) {
    setError("");
    setSavingId(alert.alert_id);
    try {
      const response = await reviewAlert(alert.alert_id, {
        patient_id: patientId,
        review_status: reviewStatus,
        reviewer_note: reviewNotes[alert.alert_id] ?? "",
      });
      setAlerts((current) => {
        if (!current) return current;
        return {
          ...current,
          alerts: current.alerts.map((item) =>
            item.alert_id === alert.alert_id ? response.alert : item,
          ),
        };
      });
    } catch (reviewError) {
      setError(reviewError instanceof Error ? reviewError.message : "Failed to save alert review.");
    } finally {
      setSavingId(null);
    }
  }

  return (
    <main className="shell">
      <section className="toolbar-row">
        <div>
          <h1 className="page-title">Alert Center</h1>
          <div className="subtle">Review, acknowledge, or dismiss recent patient alerts with persistent notes.</div>
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

      <section className="panel section-card">
        <h3>Patient Search</h3>
        <div className="input-row">
          <input value={patientInput} onChange={(event) => setPatientInput(event.target.value)} />
          <button type="button" onClick={() => setPatientId(patientInput.trim() || "P001")}>Open</button>
        </div>
      </section>

      <section className="panel section-card top-gap">
        <h3>Active Alerts</h3>
        <div className="alert-list">
          {(alerts?.alerts ?? []).map((alert) => (
            <div className="alert-item" key={alert.alert_id}>
              <div className="header-row">
                <strong>{alert.timestamp.replace("T", " ").slice(0, 16)}</strong>
                <span className={reviewPillClass(alert.review_status, alert.risk_level)}>
                  {reviewLabel(alert)}
                </span>
              </div>
              <div className="subtle">{alert.message}</div>
              <div className="subtle top-gap-small">
                Reviewer: {alert.reviewed_by ?? getStoredSession()?.user.username ?? "-"} • Reviewed at: {alert.reviewed_at ? alert.reviewed_at.replace("T", " ").slice(0, 16) : "Pending"}
              </div>
              <div className="question-box top-gap-small">
                <label className="field">
                  <span>Review note</span>
                  <textarea
                    rows={3}
                    value={reviewNotes[alert.alert_id] ?? ""}
                    onChange={(event) =>
                      setReviewNotes((current) => ({
                        ...current,
                        [alert.alert_id]: event.target.value,
                      }))
                    }
                    placeholder="Add clinician review notes, follow-up context, or dismissal reason."
                  />
                </label>
              </div>
              <div className="chip-row top-gap-small">
                {REVIEW_OPTIONS.map((option) => (
                  <button
                    key={option.value}
                    type="button"
                    className="secondary-chip"
                    onClick={() => onReview(alert, option.value)}
                    disabled={savingId === alert.alert_id}
                  >
                    {savingId === alert.alert_id ? "Saving..." : option.label}
                  </button>
                ))}
              </div>
            </div>
          ))}
          {!alerts?.alerts?.length && <div className="chart-empty">No alerts found for this patient.</div>}
        </div>
      </section>
    </main>
  );
}

export default function AlertsPage() {
  return (
    <ProtectedRoute>
      <AlertsWorkspace />
    </ProtectedRoute>
  );
}
