import { Activity, ArrowRight, BarChart3, HeartPulse, Lightbulb, UploadCloud } from "lucide-react";
import type { PageId } from "../App";
import type { DashboardResponse, ModelInfo, PredictionRow } from "../types";

type HomePageProps = {
  dashboard: DashboardResponse | null;
  models: ModelInfo[];
  selectedModel: ModelInfo | null;
  latestPrediction: PredictionRow | null;
  onNavigate: (page: PageId) => void;
};

function formatPercent(value?: number | null) {
  return typeof value === "number" ? `${Math.round(value * 100)}%` : "--";
}

export function HomePage({
  dashboard,
  models,
  selectedModel,
  latestPrediction,
  onNavigate,
}: HomePageProps) {
  const readyModels = models.filter((model) => model.inference_ready).length;
  const datasetCount = dashboard?.datasets.length ?? 0;
  const featureCount =
    dashboard?.datasets.find((dataset) => dataset.dataset_id === selectedModel?.dataset_id)?.feature_count ??
    dashboard?.datasets[0]?.feature_count ??
    0;

  return (
    <div className="page-stack home-page">
      <section className="home-hero">
        <div className="home-hero-copy">
          <h1>Parkinson's Speech Model Lab</h1>
          <p>
            A research MVP for analyzing speech-derived features, receiving Parkinson's risk predictions, and
            understanding the grouped SHAP reasoning behind each result.
          </p>
          <div className="hero-actions">
            <button className="primary-button" onClick={() => onNavigate("input")}>
              <UploadCloud size={17} />
              Start Analysis
            </button>
            <button className="secondary-button" onClick={() => onNavigate("model-insights")}>
              <BarChart3 size={17} />
              View Models
            </button>
          </div>
        </div>
        <div className="home-status-panel">
          <HeartPulse size={28} />
          <span>Current Model</span>
          <strong>{selectedModel?.model_name ?? "No model selected"}</strong>
          <small>{selectedModel?.display_family ?? "Choose a model to begin"}</small>
          <div className="home-status-metric">
            <span>Latest PD Probability</span>
            <strong>{latestPrediction ? formatPercent(latestPrediction.probability) : "--"}</strong>
          </div>
        </div>
      </section>

      <section className="workflow-grid" aria-label="Primary workflow">
        <WorkflowCard
          icon={<UploadCloud size={22} />}
          title="Input Patient Speech Data"
          body="Upload a speech-feature CSV or edit a known sample row."
          action="Go to Input"
          onClick={() => onNavigate("input")}
        />
        <WorkflowCard
          icon={<Activity size={22} />}
          title="Receive Predictions"
          body="Review classification, probability, confidence, source row, and model."
          action="Open Prediction"
          onClick={() => onNavigate("prediction")}
        />
        <WorkflowCard
          icon={<Lightbulb size={22} />}
          title="Understand The Reasoning"
          body="Use grouped SHAP explanations and the local XAI chatbot."
          action="Open Explainability"
          onClick={() => onNavigate("explainability")}
        />
      </section>

      <section className="home-metrics panel">
        <MetricTile label="Datasets" value={String(datasetCount)} />
        <MetricTile label="Inference-ready models" value={String(readyModels)} />
        <MetricTile label="Speech features" value={featureCount ? String(featureCount) : "--"} />
        <MetricTile label="Selected family" value={selectedModel?.display_family ?? "--"} />
      </section>
    </div>
  );
}

function WorkflowCard({
  icon,
  title,
  body,
  action,
  onClick,
}: {
  icon: JSX.Element;
  title: string;
  body: string;
  action: string;
  onClick: () => void;
}) {
  return (
    <article className="workflow-card">
      <div className="workflow-card-icon">{icon}</div>
      <h2>{title}</h2>
      <p>{body}</p>
      <button className="link-action" onClick={onClick}>
        {action}
        <ArrowRight size={15} />
      </button>
    </article>
  );
}

function MetricTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric-tile">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
