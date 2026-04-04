import type { PredictResponse } from "@/lib/api";

type DistributionChartProps = {
  result: PredictResponse | null;
};

export function DistributionChart({ result }: DistributionChartProps) {
  if (!result) {
    return <div className="chart-empty">Run a prediction to see anomaly/class distribution.</div>;
  }

  const entries = Object.entries(result.class_counts)
    .map(([label, count]) => ({ label, count }))
    .sort((a, b) => b.count - a.count);

  const maxCount = Math.max(...entries.map((entry) => entry.count), 1);

  return (
    <div className="distribution-list">
      {entries.map((entry) => (
        <div className="distribution-row" key={entry.label}>
          <div className="distribution-meta">
            <strong>{entry.label}</strong>
            <span>{entry.count}</span>
          </div>
          <div className="distribution-track">
            <div
              className="distribution-bar"
              style={{ width: `${(entry.count / maxCount) * 100}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}
