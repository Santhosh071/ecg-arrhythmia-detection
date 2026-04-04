"use client";

import { useEffect, useMemo, useState, useTransition } from "react";

import {
  downloadReport,
  fetchReports,
  generateReport,
  type PredictResponse,
  type ReportRecord,
  type ReportResponse,
} from "@/lib/api";

type ReportPanelProps = {
  patientId: string;
  result: PredictResponse | null;
};

export function ReportPanel({ patientId, result }: ReportPanelProps) {
  const [report, setReport] = useState<ReportResponse | null>(null);
  const [reportHistory, setReportHistory] = useState<ReportRecord[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [error, setError] = useState("");
  const [clinicalNotes, setClinicalNotes] = useState("");
  const [reportFocus, setReportFocus] = useState("Clinical review summary");
  const [isPending, startTransition] = useTransition();
  const [isDownloading, setIsDownloading] = useState(false);
  const [openingPath, setOpeningPath] = useState<string | null>(null);

  async function loadReportHistory(targetPatientId: string) {
    setHistoryLoading(true);
    try {
      const response = await fetchReports(targetPatientId);
      setReportHistory(response.reports);
    } catch (historyError) {
      setError(historyError instanceof Error ? historyError.message : "Failed to load report history.");
    } finally {
      setHistoryLoading(false);
    }
  }

  useEffect(() => {
    void loadReportHistory(patientId);
  }, [patientId]);

  const derivedSummary = useMemo(() => {
    if (!result) return null;

    const anomalousBeats = result.beats.filter((beat) => beat.is_anomaly);
    const topAnomalies = anomalousBeats.slice(0, 8).map((beat) => ({
      beat_index: beat.beat_index,
      timestamp_sec: beat.timestamp_sec,
      class_name: beat.cnn_class_name,
      class_short: beat.cnn_short_name,
      confidence: beat.cnn_confidence,
      transformer_score: beat.transformer_score,
      lstm_error: beat.lstm_error,
      risk_level: beat.risk_level,
    }));

    const reviewFlags = [
      result.request_truncated ? "Prediction request was truncated to backend safety limit." : "Prediction request completed without backend truncation.",
      anomalousBeats.length ? `${anomalousBeats.length} beats were flagged as anomalous.` : "No beats were flagged as anomalous.",
      `Dominant class observed: ${result.dominant_class}.`,
    ];

    const recommendations = [
      "Review the dominant anomaly class against symptoms, lead quality, and clinician context.",
      "Check whether repeated anomalous beats cluster in time or appear as isolated events.",
      "Correlate ECG findings with prior session history before escalating interpretation.",
      "Do not treat the AI risk level or class label as a final diagnosis without clinician review.",
    ];

    return {
      session_risk: result.session_risk,
      total_beats: result.total_beats,
      anomaly_count: result.anomaly_count,
      anomaly_rate: result.anomaly_rate,
      dominant_class: result.dominant_class,
      recording_sec: result.recording_sec,
      class_counts: result.class_counts,
      top_anomalies: topAnomalies,
      review_flags: reviewFlags,
      recommendations,
      report_focus: reportFocus,
      clinical_notes: clinicalNotes.trim(),
      summary_text: `Detected ${result.anomaly_count} anomalous beats out of ${result.total_beats}, with dominant class ${result.dominant_class}, session risk ${result.session_risk}, and report focus ${reportFocus.toLowerCase()}.`,
    };
  }, [clinicalNotes, reportFocus, result]);

  function onGenerate() {
    if (!result || !derivedSummary) {
      setError("Run a prediction first to generate a report.");
      return;
    }

    setError("");
    startTransition(async () => {
      try {
        const response = await generateReport({
          patient_id: patientId,
          session_data: derivedSummary,
        });
        setReport(response);
        await loadReportHistory(patientId);
      } catch (reportError) {
        setError(reportError instanceof Error ? reportError.message : "Report generation failed.");
      }
    });
  }

  async function onOpen(filePath: string) {
    setError("");
    setIsDownloading(true);
    setOpeningPath(filePath);
    try {
      await downloadReport(filePath);
    } catch (downloadError) {
      setError(downloadError instanceof Error ? downloadError.message : "Report download failed.");
    } finally {
      setIsDownloading(false);
      setOpeningPath(null);
    }
  }

  return (
    <div className="panel section-card">
      <div className="header-row">
        <div>
          <h3>Session Report</h3>
          <div className="subtle">Generate a richer PDF with findings, anomaly highlights, clinician notes, and review guidance.</div>
        </div>
        <button type="button" className="action-button" onClick={onGenerate}>
          {isPending ? "Generating..." : "Generate PDF"}
        </button>
      </div>

      {derivedSummary && (
        <div className="metric-list compact-metrics">
          <div className="metric"><span>Risk</span><strong>{derivedSummary.session_risk}</strong></div>
          <div className="metric"><span>Dominant Class</span><strong>{derivedSummary.dominant_class}</strong></div>
          <div className="metric"><span>Top Anomalies</span><strong>{derivedSummary.top_anomalies.length}</strong></div>
        </div>
      )}

      <div className="top-gap-small">
        <div className="subtle">Report focus</div>
        <div className="chip-row top-gap-small">
          {[
            "Clinical review summary",
            "Patient-friendly explanation",
            "Escalation handoff",
          ].map((focus) => (
            <button
              key={focus}
              type="button"
              className="secondary-chip"
              onClick={() => setReportFocus(focus)}
            >
              {focus}
            </button>
          ))}
        </div>
      </div>

      <div className="question-box top-gap-small">
        <label className="field">
          <span>Clinician Notes To Include</span>
          <textarea
            value={clinicalNotes}
            onChange={(event) => setClinicalNotes(event.target.value)}
            rows={4}
            placeholder="Add symptoms, review priorities, signal quality concerns, or follow-up context for the report."
          />
        </label>
      </div>

      {error && <div className="error-box">{error}</div>}

      {!report && !error && (
        <div className="chart-empty">No report generated yet for this session.</div>
      )}

      {report && (
        <div className="upload-status">
          <strong>{report.success ? "Report ready" : "Report failed"}</strong>
          <div className="subtle">{report.message}</div>
          {report.success && report.file_path && (
            <button type="button" className="download-link button-link" onClick={() => onOpen(report.file_path)}>
              {isDownloading && openingPath === report.file_path ? "Opening PDF..." : "Open generated PDF"}
            </button>
          )}
        </div>
      )}

      <div className="top-gap">
        <div className="header-row">
          <div>
            <h3>Patient Documents</h3>
            <div className="subtle">Stored PDF reports for this patient can be reopened here.</div>
          </div>
          <button type="button" className="secondary-chip" onClick={() => void loadReportHistory(patientId)}>
            {historyLoading ? "Refreshing..." : "Refresh list"}
          </button>
        </div>

        <div className="alert-list top-gap-small">
          {reportHistory.map((item) => (
            <div className="alert-item" key={item.report_id}>
              <div className="header-row">
                <strong>{item.file_name}</strong>
                <span className="pill pill-low">{item.status}</span>
              </div>
              <div className="subtle">Generated: {item.timestamp.replace("T", " ").slice(0, 16)}</div>
              <div className="subtle">Focus: {item.report_focus ?? "Clinical review summary"}</div>
              {item.summary && <div className="subtle top-gap-small">{item.summary}</div>}
              <div className="chip-row top-gap-small">
                <button type="button" className="secondary-chip" onClick={() => onOpen(item.file_path)}>
                  {isDownloading && openingPath === item.file_path ? "Opening..." : "Open PDF"}
                </button>
              </div>
            </div>
          ))}
          {!reportHistory.length && !historyLoading && (
            <div className="chart-empty">No previous reports found for this patient.</div>
          )}
          {historyLoading && <div className="chart-empty">Loading report history...</div>}
        </div>
      </div>
    </div>
  );
}
