import { Cpu, Layers } from "lucide-react";
import type { DashboardResponse, GroupImpact, ModelInfo } from "../types";
import { GroupImpactPage } from "./GroupImpactPage";
import { ModelPerformancePage } from "./ModelPerformancePage";

type GroupChartPoint = {
  threshold: number;
  "Healthy Control": number;
  "Parkinson's (PD)": number;
};

type ModelInsightsPageProps = {
  selectedModel: ModelInfo | null;
  performanceRows: DashboardResponse["comparison"];
  featureCount: number | null;
  groupChartData: GroupChartPoint[];
  threshold: number;
  setThreshold: (threshold: number) => void;
  thresholdRows: GroupImpact["series"];
};

export function ModelInsightsPage({
  selectedModel,
  performanceRows,
  featureCount,
  groupChartData,
  threshold,
  setThreshold,
  thresholdRows,
}: ModelInsightsPageProps) {
  return (
    <div className="page-stack model-insights-page">
      <header className="page-heading">
        <div>
          <h1>Model Insights</h1>
          <p>Compare classical and hybrid quantum models, evaluation metrics, and threshold behavior.</p>
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
          <strong>{featureCount ?? "--"}</strong>
        </div>
      </section>

      <ModelPerformancePage selectedModel={selectedModel} rows={performanceRows} />
      <GroupImpactPage
        chartData={groupChartData}
        threshold={threshold}
        setThreshold={setThreshold}
        thresholdRows={thresholdRows}
      />
    </div>
  );
}
