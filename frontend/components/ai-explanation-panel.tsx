"use client";

import { useMemo, useState, useTransition } from "react";

import { queryAgent, type AgentQueryResponse, type PredictResponse } from "@/lib/api";

type AIExplanationPanelProps = {
  patientId: string;
  result: PredictResponse | null;
};

export function AIExplanationPanel({ patientId, result }: AIExplanationPanelProps) {
  const defaultQuestion = useMemo(() => {
    if (!result) {
      return "What does this disease or ECG pattern mean clinically, and what should a clinician review next?";
    }
    return `What does this ECG session mean clinically for patient ${patientId}, given a ${result.session_risk} risk level, dominant class ${result.dominant_class}, and anomaly rate ${(result.anomaly_rate * 100).toFixed(1)}%? Include what to review next and what not to assume from the AI output.`;
  }, [patientId, result]);

  const [question, setQuestion] = useState(defaultQuestion);
  const [response, setResponse] = useState<AgentQueryResponse | null>(null);
  const [error, setError] = useState("");
  const [isPending, startTransition] = useTransition();

  function askQuestion() {
    if (!question.trim()) {
      setError("Enter a clinical question for the AI panel.");
      return;
    }

    setError("");
    startTransition(async () => {
      try {
        const anomalousBeats = result?.beats.filter((beat) => beat.is_anomaly) ?? [];
        const answer = await queryAgent({
          patient_id: patientId,
          question,
          session_context: result
            ? {
                session_risk: result.session_risk,
                dominant_class: result.dominant_class,
                anomaly_rate: result.anomaly_rate,
                anomaly_count: result.anomaly_count,
                total_beats: result.total_beats,
                class_counts: result.class_counts,
                review_flags: [
                  anomalousBeats.length ? `${anomalousBeats.length} anomalous beats detected` : "No anomalous beats detected",
                  result.request_truncated ? "Request was truncated to backend safety limit" : "Full request processed",
                ],
              }
            : undefined,
        });
        setResponse(answer);
      } catch (agentError) {
        setError(agentError instanceof Error ? agentError.message : "Explanation failed.");
      }
    });
  }

  return (
    <div className="panel section-card">
      <div className="header-row">
        <div>
          <h3>AI Clinical Assistant</h3>
          <div className="subtle">Ask disease questions directly, or ask for an interpretation grounded in the current ECG session.</div>
        </div>
        <button type="button" className="action-button" onClick={askQuestion}>
          {isPending ? "Thinking..." : "Ask AI"}
        </button>
      </div>

      <div className="question-box top-gap-small">
        <label className="field">
          <span>Clinical Question</span>
          <textarea
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            rows={5}
            placeholder="Ask about the anomaly type, disease meaning, review priorities, or patient-safe explanation."
          />
        </label>
      </div>

      <div className="chip-row">
        <button type="button" className="secondary-chip" onClick={() => setQuestion(`What type of arrhythmia or anomaly is most likely present for patient ${patientId}, and what does the dominant ECG class suggest clinically?`)}>
          Type of anomaly
        </button>
        <button type="button" className="secondary-chip" onClick={() => setQuestion(`What should the clinician review next for patient ${patientId} based on this ECG session and risk level?`)}>
          What to do next
        </button>
        <button type="button" className="secondary-chip" onClick={() => setQuestion(`What should not be assumed from this AI-generated ECG result for patient ${patientId}, and what still needs clinician confirmation?`)}>
          What not to assume
        </button>
        <button type="button" className="secondary-chip" onClick={() => setQuestion("Explain this disease pattern in simple terms for the patient or family, while keeping the answer medically accurate and non-diagnostic.")}>
          Explain disease simply
        </button>
      </div>

      {error && <div className="error-box">{error}</div>}

      {!response && !error && (
        <div className="chart-empty">
          Use this panel to ask your own disease-specific or clinician-focused question. If a prediction exists, the answer will also use the current ECG session context.
        </div>
      )}

      {response && (
        <div className="explanation-box">
          <p>{response.answer}</p>
          <div className="subtle">{response.disclaimer}</div>
        </div>
      )}
    </div>
  );
}
