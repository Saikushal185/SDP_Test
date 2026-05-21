import { Activity, Cpu, Gauge, Layers, LineChart } from "lucide-react";
import {
  Bar,
  BarChart,
  Cell,
  Pie,
  PieChart,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ConfusionMatrix, FeatureImportance, GroupImpact, ModelInfo } from "../types";
import { ConfusionMatrixPage } from "./ConfusionMatrixPage";
import { FeatureImportancePage } from "./FeatureImportancePage";
import { GroupImpactPage } from "./GroupImpactPage";

type GroupChartPoint = {
  threshold: number;
  "Healthy Control": number;
  "Parkinson's (PD)": number;
};

type ModelInsightsPageProps = {
  selectedModel: ModelInfo | null;
  featureCount: number | null;
  features: FeatureImportance[];
  confusionMatrix: ConfusionMatrix | null;
  groupChartData: GroupChartPoint[];
  threshold: number;
  setThreshold: (threshold: number) => void;
  thresholdRows: GroupImpact["series"];
};

export function ModelInsightsPage({
  selectedModel,
  featureCount,
  features,
  confusionMatrix,
  groupChartData,
  threshold,
  setThreshold,
  thresholdRows,
}: ModelInsightsPageProps) {
  const metrics = selectedModel?.metrics;
  const metricProfile = [
    { metric: "Accuracy", value: boundedMetric(metrics?.mean_accuracy) },
    { metric: "Recall", value: boundedMetric(metrics?.mean_recall) },
    { metric: "F1", value: boundedMetric(metrics?.mean_f1) },
    { metric: "ROC-AUC", value: boundedMetric(metrics?.mean_roc_auc) },
  ];
  const outcomeData = getOutcomeData(confusionMatrix);
  const classRecallData = getClassRecallData(confusionMatrix);

  return (
    <div className="page-stack model-insights-page">
      <header className="page-heading">
        <div>
          <h1>Model Insights</h1>
          <p>Selected-model validation metrics, feature drivers, confusion matrix, and threshold behavior.</p>
        </div>
        <div className="page-note">
          <Cpu size={17} />
          <span>{selectedModel?.dataset_id ?? "No dataset selected"}</span>
        </div>
      </header>

      <section className="insight-summary-grid" aria-label="Selected model details">
        <div className="insight-summary-tile">
          <Layers size={20} />
          <span>Selected Model</span>
          <strong>{selectedModel?.model_name ?? "--"}</strong>
        </div>
        <div className="insight-summary-tile">
          <Cpu size={20} />
          <span>Model Family</span>
          <strong>{selectedModel?.display_family ?? "--"}</strong>
        </div>
        <div className="insight-summary-tile">
          <Layers size={20} />
          <span>Feature Count</span>
          <strong>{metrics?.selected_feature_count ?? featureCount ?? "--"}</strong>
        </div>
        <div className="insight-summary-tile">
          <Layers size={20} />
          <span>Feature Selection</span>
          <strong>{metrics?.feature_selection_method === "mutual_information" ? "Mutual Information" : "--"}</strong>
        </div>
        <div className="insight-summary-tile">
          <Cpu size={20} />
          <span>Feature Mode</span>
          <strong>{metrics?.feature_mode ?? "--"}</strong>
        </div>
        <div className="insight-summary-tile">
          <Cpu size={20} />
          <span>Fit Strategy</span>
          <strong>{metrics?.fit_strategy ?? "--"}</strong>
        </div>
      </section>

      <section className="selected-metric-grid" aria-label="Selected model metrics">
        <MetricCard icon={<Gauge size={20} />} label="Accuracy" value={formatMetric(metrics?.mean_accuracy)} />
        <MetricCard icon={<Activity size={20} />} label="Recall" value={formatMetric(metrics?.mean_recall)} />
        <MetricCard icon={<Layers size={20} />} label="F1 Score" value={formatMetric(metrics?.mean_f1)} />
        <MetricCard icon={<LineChart size={20} />} label="ROC-AUC" value={formatMetric(metrics?.mean_roc_auc)} />
      </section>

      <section className="insight-visual-grid" aria-label="Additional selected model visualizations">
        <article className="panel insight-chart-card">
          <div className="section-heading compact">
            <div>
              <h2>Metric Shape</h2>
              <p>Balanced view of validation scores for this model.</p>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={230}>
            <RadarChart data={metricProfile} outerRadius={76}>
              <PolarGrid stroke="#dbe4ea" />
              <PolarAngleAxis dataKey="metric" tick={{ fontSize: 12 }} />
              <PolarRadiusAxis domain={[0, 1]} tick={{ fontSize: 10 }} />
              <Tooltip formatter={(value: number) => value.toFixed(3)} />
              <Radar dataKey="value" stroke="#0f8b8d" fill="#139a9a" fillOpacity={0.28} />
            </RadarChart>
          </ResponsiveContainer>
        </article>

        <article className="panel insight-chart-card">
          <div className="section-heading compact">
            <div>
              <h2>Outcome Mix</h2>
              <p>Correct and incorrect predictions from 10-fold validation.</p>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={230}>
            <PieChart>
              <Pie data={outcomeData} dataKey="value" nameKey="name" innerRadius={54} outerRadius={82} paddingAngle={2}>
                {outcomeData.map((entry) => (
                  <Cell key={entry.name} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip formatter={(value: number) => value} />
            </PieChart>
          </ResponsiveContainer>
          <div className="outcome-legend">
            {outcomeData.map((entry) => (
              <span key={entry.name}>
                <i style={{ background: entry.color }} />
                {entry.name}
              </span>
            ))}
          </div>
        </article>

        <article className="panel insight-chart-card class-recall-card">
          <div className="section-heading compact">
            <div>
              <h2>Class Recall</h2>
              <p>How well each actual class is recognized.</p>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={230}>
            <BarChart data={classRecallData} margin={{ left: 8, right: 22, bottom: 8 }}>
              <XAxis dataKey="name" tick={{ fontSize: 11 }} />
              <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
              <Tooltip formatter={(value: number) => value.toFixed(3)} />
              <Bar dataKey="value" radius={[5, 5, 0, 0]}>
                {classRecallData.map((entry) => (
                  <Cell key={entry.name} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </article>
      </section>

      <div className="selected-insight-grid">
        <FeatureImportancePage features={features} />
        <ConfusionMatrixPage matrix={confusionMatrix} />
      </div>
      <GroupImpactPage
        chartData={groupChartData}
        threshold={threshold}
        setThreshold={setThreshold}
        thresholdRows={thresholdRows}
      />
    </div>
  );
}

function formatMetric(value?: number | null) {
  return typeof value === "number" ? value.toFixed(3) : "--";
}

function boundedMetric(value?: number | null) {
  if (typeof value !== "number" || Number.isNaN(value)) return 0;
  return Math.max(0, Math.min(1, value));
}

function getConfusionCount(matrix: ConfusionMatrix | null, actual: number, predicted: number) {
  return matrix?.cells.find((cell) => cell.actual === actual && cell.predicted === predicted)?.count ?? 0;
}

function getOutcomeData(matrix: ConfusionMatrix | null) {
  const trueHealthy = getConfusionCount(matrix, 0, 0);
  const falsePd = getConfusionCount(matrix, 0, 1);
  const falseHealthy = getConfusionCount(matrix, 1, 0);
  const truePd = getConfusionCount(matrix, 1, 1);
  return [
    { name: "True Healthy", value: trueHealthy, color: "#0f8b8d" },
    { name: "True PD", value: truePd, color: "#2f735c" },
    { name: "False PD", value: falsePd, color: "#d98a21" },
    { name: "False Healthy", value: falseHealthy, color: "#984f46" },
  ];
}

function ratio(numerator: number, denominator: number) {
  return denominator > 0 ? numerator / denominator : 0;
}

function getClassRecallData(matrix: ConfusionMatrix | null) {
  const trueHealthy = getConfusionCount(matrix, 0, 0);
  const falsePd = getConfusionCount(matrix, 0, 1);
  const falseHealthy = getConfusionCount(matrix, 1, 0);
  const truePd = getConfusionCount(matrix, 1, 1);
  return [
    { name: "Healthy", value: ratio(trueHealthy, trueHealthy + falsePd), color: "#0f8b8d" },
    { name: "PD", value: ratio(truePd, truePd + falseHealthy), color: "#d98a21" },
  ];
}

function MetricCard({
  icon,
  label,
  value,
}: {
  icon: JSX.Element;
  label: string;
  value: string;
}) {
  return (
    <article className="selected-metric-card">
      {icon}
      <span>{label}</span>
      <strong>{value}</strong>
    </article>
  );
}
