"use client";

import { useEffect, useMemo, useState } from "react";

import type { PredictBeat } from "@/lib/api";

type WaveformChartProps = {
  beats: number[][];
  results: PredictBeat[];
};

function toSegmentPath(samples: number[], xStart: number, width: number, height: number) {
  if (!samples.length) return "";
  const min = Math.min(...samples);
  const max = Math.max(...samples);
  const range = Math.max(max - min, 1e-6);

  return samples
    .map((value, index) => {
      const x = xStart + (index / Math.max(samples.length - 1, 1)) * width;
      const normalized = (value - min) / range;
      const y = height - normalized * height;
      return `${index === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
}

function riskColor(riskLevel?: string, isAnomaly?: boolean) {
  const value = (riskLevel ?? "").toLowerCase();
  if (value === "high" || value === "critical") return "#b42318";
  if (value === "medium") return "#b45309";
  if (isAnomaly) return "#b45309";
  return "#0f766e";
}

function riskPill(riskLevel?: string, isAnomaly?: boolean) {
  const value = (riskLevel ?? "").toLowerCase();
  if (value === "high" || value === "critical") return "pill pill-high";
  if (value === "medium") return "pill pill-medium";
  if (isAnomaly) return "pill pill-medium";
  return "pill pill-low";
}

export function WaveformChart({ beats, results }: WaveformChartProps) {
  const windowSize = Math.min(18, Math.max(beats.length, 1));
  const [windowStart, setWindowStart] = useState(0);
  const [selectedOffset, setSelectedOffset] = useState(0);

  useEffect(() => {
    setWindowStart(0);
    setSelectedOffset(0);
  }, [beats]);

  useEffect(() => {
    if (beats.length <= windowSize) {
      return;
    }

    const interval = window.setInterval(() => {
      setWindowStart((current) => {
        const lastStart = Math.max(beats.length - windowSize, 0);
        return current >= lastStart ? 0 : current + 1;
      });
    }, 500);

    return () => window.clearInterval(interval);
  }, [beats, windowSize]);

  const liveWindow = useMemo(
    () => beats.slice(windowStart, windowStart + windowSize),
    [beats, windowSize, windowStart],
  );
  const liveResults = useMemo(
    () => results.slice(windowStart, windowStart + liveWindow.length),
    [results, windowStart, liveWindow.length],
  );


  useEffect(() => {
    setSelectedOffset((current) => Math.min(current, Math.max(liveWindow.length - 1, 0)));
  }, [liveWindow.length]);

  if (!liveWindow.length) {
    return <div className="chart-empty">Upload a beat array to preview the live waveform strip.</div>;
  }

  const anomalyCount = liveResults.filter((beat) => beat?.is_anomaly).length;
  const averageConfidence =
    liveResults.length > 0
      ? liveResults.reduce((sum, beat) => sum + (beat?.cnn_confidence ?? 0), 0) / liveResults.length
      : 0;
  const beatWidth = 900 / Math.max(liveWindow.length, 1);
  const selectedBeat = liveResults[selectedOffset];
  const selectedAbsoluteIndex = windowStart + selectedOffset;

  return (
    <div className="stack">
      <div className="wave-card live-strip-card">
        <div className="wave-label-row">
          <strong>Live Input Beat Strip</strong>
          <span className="pill pill-low">
            Beats {windowStart + 1}-{windowStart + liveWindow.length} of {beats.length}
          </span>
        </div>

        <svg viewBox="0 0 900 150" className="wave-svg wave-svg-large" aria-label="Live waveform preview">
          {liveWindow.map((beat, index) => {
            const result = liveResults[index];
            const selected = index === selectedOffset;
            return (
              <g key={`${windowStart + index}-${result?.beat_index ?? index}`}>
                <rect
                  x={index * beatWidth}
                  y={0}
                  width={beatWidth}
                  height={150}
                  fill="transparent"
                  style={{ cursor: result ? "pointer" : "default" }}
                  onClick={() => setSelectedOffset(index)}
                />
                <path
                  d={toSegmentPath(beat, index * beatWidth, beatWidth, 135)}
                  fill="none"
                  stroke={riskColor(result?.risk_level, result?.is_anomaly)}
                  strokeWidth={selected ? "3.6" : "2.2"}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  style={{ opacity: selected ? 1 : 0.88, pointerEvents: "none" }}
                />
              </g>
            );
          })}
        </svg>

        <div className="metric-list compact-metrics top-gap-small">
          <div className="metric"><span>Displayed Beats</span><strong>{liveWindow.length}</strong></div>
          <div className="metric"><span>Anomalies In View</span><strong>{anomalyCount}</strong></div>
          <div className="metric"><span>Avg Confidence</span><strong>{(averageConfidence * 100).toFixed(1)}%</strong></div>
        </div>

        <div className="subtle top-gap-small">
          Low risk beats render in teal, medium in amber, and high or critical beats in red while playback advances every 0.5 seconds.
        </div>
      </div>

      {selectedBeat ? (
        <div className="panel section-card">
          <div className="header-row">
            <div>
              <h3>Beat Inspector</h3>
              <div className="subtle">Click any beat in the live strip or list to inspect model outputs for that beat.</div>
            </div>
            <span className={riskPill(selectedBeat.risk_level, selectedBeat.is_anomaly)}>{selectedBeat.risk_level}</span>
          </div>
          <div className="metric-list compact-metrics top-gap-small">
            <div className="metric"><span>Beat</span><strong>#{selectedBeat.beat_index}</strong></div>
            <div className="metric"><span>Window Slot</span><strong>{selectedAbsoluteIndex + 1}</strong></div>
            <div className="metric"><span>Timestamp</span><strong>{selectedBeat.timestamp_sec.toFixed(2)}s</strong></div>
            <div className="metric"><span>Class</span><strong>{selectedBeat.cnn_short_name}</strong></div>
            <div className="metric"><span>Confidence</span><strong>{(selectedBeat.cnn_confidence * 100).toFixed(1)}%</strong></div>
            <div className="metric"><span>Transformer Score</span><strong>{selectedBeat.transformer_score.toFixed(4)}</strong></div>
            <div className="metric"><span>LSTM Error</span><strong>{selectedBeat.lstm_error.toFixed(4)}</strong></div>
            <div className="metric"><span>Anomaly</span><strong>{selectedBeat.is_anomaly ? "Yes" : "No"}</strong></div>
            <div className="metric"><span>Transformer Flag</span><strong>{selectedBeat.transformer_anomaly ? "Flagged" : "Clear"}</strong></div>
            <div className="metric"><span>LSTM Flag</span><strong>{selectedBeat.lstm_anomaly ? "Flagged" : "Clear"}</strong></div>
          </div>
        </div>
      ) : null}

      {liveResults.length > 0 ? (
        <div className="live-beat-list">
          {liveResults.slice(0, 10).map((result, index) => {
            const displayedBeatIndex = result?.beat_index ?? windowStart + index;
            const selected = index === selectedOffset;
            return (
              <button
                key={`${displayedBeatIndex}-${index}`}
                type="button"
                className="history-item"
                style={{ textAlign: "left", borderColor: selected ? "rgba(15, 118, 110, 0.35)" : undefined }}
                onClick={() => setSelectedOffset(index)}
              >
                <div className="header-row">
                  <strong>Live Beat #{displayedBeatIndex}</strong>
                  <span className={riskPill(result?.risk_level, result?.is_anomaly)}>
                    {result?.risk_level ?? (result?.is_anomaly ? "Medium" : "Low")}
                  </span>
                </div>
                <div className="subtle">
                  {result?.cnn_short_name ?? "Normal"} � t={(result?.timestamp_sec ?? 0).toFixed(2)}s � confidence {((result?.cnn_confidence ?? 0) * 100).toFixed(1)}%
                </div>
              </button>
            );
          })}
        </div>
      ) : (
        <div className="chart-empty">Prediction labels will appear here after inference results are returned.</div>
      )}
    </div>
  );
}















