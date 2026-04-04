"use client";

import { useEffect, useState, useTransition } from "react";

import { predictBatch, type PredictResponse } from "@/lib/api";
import { parseNpyFile, toPredictPayload } from "@/lib/npy";

type PredictionWorkbenchProps = {
  patientId: string;
  onPatientIdCommit: (patientId: string) => void;
  onPredictionComplete: (payload: {
    result: PredictResponse;
    beats: number[][];
    fileName: string;
    patientId: string;
  }) => void;
};

function riskClassName(riskLevel: string) {
  const value = riskLevel.toLowerCase();
  if (value === "low") return "pill pill-low";
  if (value === "medium") return "pill pill-medium";
  return "pill pill-high";
}

export function PredictionWorkbench({
  patientId,
  onPatientIdCommit,
  onPredictionComplete,
}: PredictionWorkbenchProps) {
  const [draftPatientId, setDraftPatientId] = useState(patientId);
  const [selectedFileName, setSelectedFileName] = useState("");
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState("");
  const [isPending, startTransition] = useTransition();

  useEffect(() => {
    setDraftPatientId(patientId);
  }, [patientId]);

  function onFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    setError("");
    setResult(null);

    if (!file) {
      setSelectedFileName("");
      return;
    }

    setSelectedFileName(file.name);

    startTransition(async () => {
      try {
        const parsed = await parseNpyFile(file);
        const resolvedPatientId = draftPatientId.trim() || "P001";
        onPatientIdCommit(resolvedPatientId);
        const payload = toPredictPayload(resolvedPatientId, parsed);
        const response = await predictBatch(payload);
        setResult(response);
        onPredictionComplete({
          result: response,
          beats: payload.beats,
          fileName: file.name,
          patientId: resolvedPatientId,
        });
      } catch (uploadError) {
        setError(uploadError instanceof Error ? uploadError.message : "Prediction failed.");
      }
    });
  }

  return (
    <div className="panel section-card">
      <div className="header-row">
        <div>
          <h2>Prediction Workbench</h2>
          <div className="subtle">
            Upload a `.npy` beat array with shape `(N, 187)` to test FastAPI `/predict`.
          </div>
        </div>
        <span className="pill pill-low">Upload Ready</span>
      </div>

      <div className="form-grid">
        <label className="field">
          <span>Patient ID</span>
          <input
            value={draftPatientId}
            onChange={(event) => setDraftPatientId(event.target.value)}
          />
        </label>

        <label className="field field-upload">
          <span>Beat File</span>
          <input type="file" accept=".npy" onChange={onFileChange} />
        </label>
      </div>

      <div className="upload-status">
        <strong>{selectedFileName || "No beat file selected yet."}</strong>
        <div className="subtle">
          {isPending
            ? "Parsing file and running inference..."
            : "Raw 1D ECG upload is a separate next step. This flow expects pre-segmented beats."}
        </div>
      </div>

      {error && <div className="error-box">{error}</div>}

      {result && (
        <div className="metric-list">
          <div className="metric">
            <span>Session Risk</span>
            <strong>
              <span className={riskClassName(result.session_risk)}>{result.session_risk}</span>
            </strong>
          </div>
          <div className="metric">
            <span>Total Beats</span>
            <strong>{result.total_beats}</strong>
          </div>
          <div className="metric">
            <span>Anomaly Count</span>
            <strong>{result.anomaly_count}</strong>
          </div>
          <div className="metric">
            <span>Anomaly Rate</span>
            <strong>{(result.anomaly_rate * 100).toFixed(1)}%</strong>
          </div>
          <div className="metric">
            <span>Dominant Class</span>
            <strong>{result.dominant_class}</strong>
          </div>
          <div className="metric">
            <span>Session Saved</span>
            <strong>{result.session_saved ? "Yes" : "No"}</strong>
          </div>
          {result.request_truncated && (
            <div className="warning-box">
              The backend truncated the request to its safety limit before prediction finished.
            </div>
          )}
        </div>
      )}
    </div>
  );
}
