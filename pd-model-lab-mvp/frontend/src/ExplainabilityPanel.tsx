import type { RefObject } from "react";
import { BarChart3, Lightbulb } from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { GroupedExplanation, ModelInfo, PredictionRow } from "./types";

function formatPrecisePercent(value?: number | null) {
  return typeof value === "number" ? `${(value * 100).toFixed(1)}%` : "--";
}

function formatContribution(value?: number | null) {
  if (typeof value !== "number") return "--";
  const points = Math.abs(value) * 100;
  const digits = points >= 10 ? 1 : 2;
  return `${value >= 0 ? "+" : "-"}${points.toFixed(digits)} pp`;
}

function riskLabel(probability: number) {
  if (probability >= 0.67) return "High Risk";
  if (probability >= 0.33) return "Medium Risk";
  return "Low Risk";
}

function confidenceLabel(confidence: number) {
  if (confidence >= 0.85) return "High confidence";
  if (confidence >= 0.65) return "Moderate confidence";
  return "Low confidence";
}

function predictionHeadline(label: string) {
  return label === "Parkinson's (PD)" ? "Parkinson's Positive" : "Healthy";
}

function driverSentence(driver: GroupedExplanation) {
  if (driver.value >= 0) {
    return `${driver.name} moved this sample toward a Parkinson's-positive prediction.`;
  }
  return `${driver.name} moved this sample away from a Parkinson's-positive prediction.`;
}

function explanationMethodLabel(method?: string) {
  if (method === "native") return "Native SHAP";
  if (method === "kernel-grouped") return "Grouped Kernel SHAP";
  return "SHAP";
}

export function ExplainabilityPanel({
  refProp,
  prediction,
  selectedModel,
}: {
  refProp: RefObject<HTMLElement>;
  prediction: PredictionRow | null;
  selectedModel: ModelInfo | null;
}) {
  const groups = prediction?.explanation?.groups ?? [];
  const increased = groups.filter((item) => item.value > 0).slice(0, 3);
  const decreased = groups.filter((item) => item.value < 0).slice(0, 3);
  const risk = prediction ? riskLabel(prediction.probability) : null;
  const baseValue =
    typeof prediction?.explanation?.base_value === "number"
      ? prediction.explanation.base_value
      : null;
  const groupedShift = prediction && baseValue !== null ? prediction.probability - baseValue : null;
  const predictionModelName =
    prediction && selectedModel?.model_key === prediction.model_key
      ? selectedModel.model_name
      : prediction?.model_key.split("_").pop();

  return (
    <section ref={refProp} className="panel explainability-panel" id="explainability">
      <div className="section-heading">
        <div>
          <h2>Grouped Explainability</h2>
          <p>Plain-language SHAP drivers for the latest prediction result.</p>
        </div>
        <div className="dataset-chip">
          {prediction ? explanationMethodLabel(prediction.explanation?.method) : "Awaiting prediction"}
        </div>
      </div>

      {!prediction || !prediction.explanation ? (
        <div className="explainability-empty">
          <div className="empty-icon">
            <Lightbulb size={26} />
          </div>
          <div>
            <h3>No explanation available yet</h3>
            <p>Run a prediction from a sample row or CSV to show grouped voice-signal drivers here.</p>
          </div>
        </div>
      ) : (
        <div className="explainability-grid">
          <div className="explainability-chart-card">
            <div className="explainability-title">
              <div className="chart-icon">
                <BarChart3 size={22} />
              </div>
              <div>
                <span>Grouped explanation</span>
                <h3>What most influenced the score</h3>
                <p>Red bars raised the displayed probability. Green bars lowered it from the model baseline.</p>
              </div>
            </div>

            <div className="grouped-chart" style={{ height: Math.max(groups.length * 58 + 28, 230) }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={groups} layout="vertical" margin={{ top: 8, right: 26, left: 126, bottom: 8 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e9e5" />
                  <XAxis
                    type="number"
                    tick={{ fontSize: 11 }}
                    tickFormatter={(value: number) => `${(value * 100).toFixed(0)} pp`}
                  />
                  <YAxis
                    dataKey="name"
                    type="category"
                    width={120}
                    tick={{ fontSize: 11, fontWeight: 700 }}
                  />
                  <Tooltip
                    formatter={(value: number, _name, payload) => {
                      const count = payload?.payload?.featureCount ?? 0;
                      return [
                        formatContribution(value),
                        `${count} related feature${count === 1 ? "" : "s"}`,
                      ];
                    }}
                  />
                  <Bar dataKey="value" radius={[0, 7, 7, 0]} barSize={30}>
                    {groups.map((entry) => (
                      <Cell key={entry.name} fill={entry.value >= 0 ? "#984f46" : "#2f735c"} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="explainability-side">
            <div className="risk-summary-card">
              <span className={`risk-pill ${risk === "High Risk" ? "risk-high" : risk === "Medium Risk" ? "risk-medium" : "risk-low"}`}>
                {risk}
              </span>
              <h3>{predictionHeadline(prediction.predicted_label)}</h3>
              <p>{confidenceLabel(prediction.confidence)} using {predictionModelName ?? prediction.model_key}</p>
              <div className="risk-data">
                <div>
                  <span>Probability</span>
                  <strong>{formatPrecisePercent(prediction.probability)}</strong>
                </div>
                <div>
                  <span>Confidence</span>
                  <strong>{formatPrecisePercent(prediction.confidence)}</strong>
                </div>
                {baseValue !== null && (
                  <div>
                    <span>Baseline</span>
                    <strong>{formatPrecisePercent(baseValue)}</strong>
                  </div>
                )}
                {groupedShift !== null && (
                  <div>
                    <span>Grouped Shift</span>
                    <strong>{formatContribution(groupedShift)}</strong>
                  </div>
                )}
              </div>
            </div>

            <ReasonList title="Main reasons the score increased" tone="increase" items={increased} />
            <ReasonList title="Main reasons the score decreased" tone="decrease" items={decreased} />
          </div>
        </div>
      )}

      <div className="explainability-warning">
        These explanations are percentage-point movements from the model baseline for this sample. They do not identify a medical cause or replace a clinical diagnosis.
      </div>
    </section>
  );
}

function ReasonList({
  title,
  tone,
  items,
}: {
  title: string;
  tone: "increase" | "decrease";
  items: GroupedExplanation[];
}) {
  return (
    <div className={`reason-panel ${tone}`}>
      <h3>{title}</h3>
      {items.length ? (
        <div className="reason-list">
          {items.map((item) => (
            <div className="reason-card" key={item.name}>
              <div className="reason-card-header">
                <strong>{item.name}</strong>
                <span>{formatContribution(item.value)}</span>
              </div>
              <small>{item.featureCount} grouped features</small>
              <p>{driverSentence(item)}</p>
            </div>
          ))}
        </div>
      ) : (
        <p className="reason-empty">No grouped drivers moved the score in this direction.</p>
      )}
    </div>
  );
}
