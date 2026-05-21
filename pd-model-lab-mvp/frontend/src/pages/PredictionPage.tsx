import { Activity, ArrowRight, CheckCircle2, Gauge, UploadCloud } from "lucide-react";
import type { PageId } from "../App";
import type { ModelInfo, PredictionRow } from "../types";
import { RecentPredictionsPage } from "./RecentPredictionsPage";

type PredictionPageProps = {
  latestPrediction: PredictionRow | null;
  predictions: PredictionRow[];
  selectedModel: ModelInfo | null;
  onNavigate: (page: PageId) => void;
};

function formatPercent(value?: number | null) {
  return typeof value === "number" ? `${(value * 100).toFixed(1)}%` : "--";
}

function resultTone(label?: string) {
  return label === "Parkinson's (PD)" ? "pd" : "healthy";
}

export function PredictionPage({
  latestPrediction,
  predictions,
  selectedModel,
  onNavigate,
}: PredictionPageProps) {
  if (!latestPrediction) {
    return (
      <div className="page-stack prediction-page">
        <header className="page-heading">
          <div>
            <h1>Prediction Result</h1>
            <p>Run an analysis to receive a classification result, probability, and confidence score.</p>
          </div>
        </header>

        <section className="panel prediction-empty">
          <UploadCloud size={34} />
          <h2>No prediction has been run for this model yet</h2>
          <p>Start with a sample row or upload a CSV file on the Input page.</p>
          <button className="primary-button" onClick={() => onNavigate("input")}>
            <UploadCloud size={17} />
            Go to Input
          </button>
        </section>
      </div>
    );
  }

  const tone = resultTone(latestPrediction.predicted_label);

  return (
    <div className="page-stack prediction-page">
      <header className="page-heading">
        <div>
          <h1>Prediction Result</h1>
          <p>Latest classification result for the selected model.</p>
        </div>
        <button className="secondary-button" onClick={() => onNavigate("input")}>
          <UploadCloud size={16} />
          New Input
        </button>
      </header>

      <div className="prediction-result-grid">
        <section className={`panel result-summary-card ${tone}`}>
          <div className="result-icon">
            {tone === "pd" ? <Activity size={26} /> : <CheckCircle2 size={26} />}
          </div>
          <span>Classification</span>
          <h2>{latestPrediction.predicted_label}</h2>
          <p>
            Source {latestPrediction.source}, row {latestPrediction.row_index + 1}
          </p>
        </section>

        <section className="panel result-metrics-card">
          <div className="result-metric">
            <span>Probability</span>
            <strong>{formatPercent(latestPrediction.probability)}</strong>
          </div>
          <div className="result-metric">
            <span>Confidence</span>
            <strong>{formatPercent(latestPrediction.confidence)}</strong>
          </div>
          <div className="result-metric">
            <span>Model</span>
            <strong>{selectedModel?.model_name ?? latestPrediction.model_key}</strong>
          </div>
        </section>

        <section className="panel result-next-card">
          <Gauge size={22} />
          <h2>Review the reasoning</h2>
          <p>The explainability page shows grouped SHAP contributions and a local chatbot for this result.</p>
          <button className="link-action" onClick={() => onNavigate("explainability")}>
            Open Explainability
            <ArrowRight size={15} />
          </button>
        </section>
      </div>

      <RecentPredictionsPage predictions={predictions} />
    </div>
  );
}
