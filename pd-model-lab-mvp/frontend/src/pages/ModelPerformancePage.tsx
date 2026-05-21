import type { RefObject } from "react";
import type { DashboardResponse, ModelInfo } from "../types";

type ModelPerformancePageProps = {
  refProp?: RefObject<HTMLElement>;
  selectedModel: ModelInfo | null;
  rows: DashboardResponse["comparison"];
};

function formatMetric(value?: number | null, digits = 3) {
  return typeof value === "number" ? value.toFixed(digits) : "--";
}

export function ModelPerformancePage({
  refProp,
  selectedModel,
  rows,
}: ModelPerformancePageProps) {
  return (
    <section ref={refProp} className="panel" id="dashboard">
      <div className="section-heading">
        <div>
          <h2>Model Performance</h2>
          <p>Cross-validation metrics from saved dataset artifacts.</p>
        </div>
        <div className="dataset-chip">{selectedModel?.dataset_id}</div>
      </div>
      <PerformanceTable rows={rows} />
    </section>
  );
}

function PerformanceTable({ rows }: { rows: DashboardResponse["comparison"] }) {
  const classical = rows.filter((row) => row.model_family === "classical");
  const hybrid = rows.filter((row) => row.model_family === "quantum");
  const metrics = [
    ["Accuracy", "mean_accuracy"],
    ["AUC-ROC", "mean_roc_auc"],
    ["F1", "mean_f1"],
  ] as const;

  return (
    <div className="performance-layout">
      <div className="performance-table">
        <h3>Classical Models</h3>
        <MetricTable rows={classical} metrics={metrics} />
      </div>
      <div className="performance-table hybrid">
        <h3>Hybrid Quantum Models</h3>
        <MetricTable rows={hybrid} metrics={metrics} />
      </div>
    </div>
  );
}

function MetricTable({
  rows,
  metrics,
}: {
  rows: DashboardResponse["comparison"];
  metrics: readonly (readonly ["Accuracy" | "AUC-ROC" | "F1", "mean_accuracy" | "mean_roc_auc" | "mean_f1"])[];
}) {
  return (
    <table>
      <thead>
        <tr>
          <th>Metric</th>
          {rows.map((row) => (
            <th key={row.model_key}>{row.model_name}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {metrics.map(([label, key]) => (
          <tr key={label}>
            <th>{label}</th>
            {rows.map((row) => (
              <td key={`${row.model_key}-${label}`}>
                {formatMetric(row[key])}
                {row.status === "repaired" && <span className="mini-badge">repaired</span>}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
