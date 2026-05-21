import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  BarChart3,
  CheckCircle2,
  CircleUserRound,
  Home,
  Info,
  Lightbulb,
  Loader2,
  RefreshCw,
  Sparkles,
  UploadCloud,
  Users,
} from "lucide-react";
import {
  fetchConfusionMatrix,
  fetchDashboard,
  fetchFeatureImportance,
  fetchGroupImpact,
  fetchModels,
  fetchSamples,
  runPrediction,
} from "./api";
import type {
  ConfusionMatrix,
  DashboardResponse,
  FeatureImportance,
  GroupImpact,
  ModelInfo,
  PredictionRow,
  SamplePayload,
} from "./types";
import { AboutPage } from "./pages/AboutPage";
import { ExplainabilityPage } from "./pages/ExplainabilityPage";
import { HomePage } from "./pages/HomePage";
import { InputPage } from "./pages/InputPage";
import { ModelInsightsPage } from "./pages/ModelInsightsPage";
import { PredictionPage } from "./pages/PredictionPage";

export type PageId = "home" | "input" | "prediction" | "explainability" | "model-insights" | "about";

const navItems: Array<{ id: PageId; label: string; icon: typeof Home }> = [
  { id: "home", label: "Home", icon: Home },
  { id: "input", label: "Input", icon: UploadCloud },
  { id: "prediction", label: "Prediction", icon: Activity },
  { id: "explainability", label: "Explainability", icon: Lightbulb },
  { id: "model-insights", label: "Model Insights", icon: BarChart3 },
  { id: "about", label: "About", icon: Users },
];

function statusLabel(status?: string) {
  if (status === "repaired") return "Repaired / Inference-Ready";
  if (status === "ready") return "Inference-Ready";
  return "Unavailable";
}

export default function App() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [dashboard, setDashboard] = useState<DashboardResponse | null>(null);
  const [activePage, setActivePage] = useState<PageId>("home");
  const [selectedKey, setSelectedKey] = useState("");
  const [samples, setSamples] = useState<SamplePayload | null>(null);
  const [sampleIndex, setSampleIndex] = useState(0);
  const [edits, setEdits] = useState<Record<string, number | null>>({});
  const [features, setFeatures] = useState<FeatureImportance[]>([]);
  const [impact, setImpact] = useState<GroupImpact | null>(null);
  const [confusionMatrix, setConfusionMatrix] = useState<ConfusionMatrix | null>(null);
  const [threshold, setThreshold] = useState(0.5);
  const [recentPredictions, setRecentPredictions] = useState<PredictionRow[]>([]);
  const [isPredicting, setIsPredicting] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadInitialData() {
      try {
        const [modelRows, dashboardPayload] = await Promise.all([fetchModels(), fetchDashboard()]);
        setModels(modelRows);
        setDashboard(dashboardPayload);
        const preferred =
          modelRows.find((model) => model.model_key === "pd_speech_features_local_top10_mi_XGBoost") ??
          modelRows.find((model) => model.inference_ready) ??
          modelRows[0];
        setSelectedKey(preferred?.model_key ?? "");
      } catch (reason) {
        setError(reason instanceof Error ? reason.message : "Unable to load model metadata");
      } finally {
        setLoading(false);
      }
    }
    loadInitialData();
  }, []);

  const selectedModel = useMemo(
    () => models.find((model) => model.model_key === selectedKey) ?? null,
    [models, selectedKey],
  );

  useEffect(() => {
    if (!selectedModel) return;
    const model = selectedModel;
    setEdits({});
    setSampleIndex(0);
    setFeatures([]);
    setImpact(null);
    setConfusionMatrix(null);
    setError(null);
    async function loadModelData() {
      try {
        const [samplePayload, featureRows, impactPayload, matrixPayload] = await Promise.all([
          fetchSamples(model.dataset_id, 5),
          fetchFeatureImportance(model.model_key),
          fetchGroupImpact(model.model_key),
          fetchConfusionMatrix(model.model_key),
        ]);
        setSamples(samplePayload);
        setFeatures(featureRows);
        setImpact(impactPayload);
        setConfusionMatrix(matrixPayload);
        setError(null);
      } catch (reason) {
        setError(reason instanceof Error ? reason.message : "Unable to load selected model data");
      }
    }
    loadModelData();
  }, [selectedModel]);

  const currentSample = samples?.rows[sampleIndex] ?? null;
  const editableFeatureNames = useMemo(() => {
    const source = samples?.feature_names ?? [];
    const important = features.map((item) => item.feature);
    const merged = Array.from(new Set([...important, ...source]));
    return merged.slice(0, 10);
  }, [samples, features]);

  const groupChartData = useMemo(() => {
    if (!impact) return [];
    const thresholds = Array.from(new Set(impact.series.map((row) => row.threshold)));
    return thresholds.map((point) => {
      const rows = impact.series.filter((row) => row.threshold === point);
      return {
        threshold: point,
        "Healthy Control": rows.find((row) => row.group === "Healthy Control")?.positive_rate ?? 0,
        "Parkinson's (PD)": rows.find((row) => row.group === "Parkinson's (PD)")?.positive_rate ?? 0,
      };
    });
  }, [impact]);

  const thresholdRows = useMemo(() => {
    if (!impact) return [];
    return impact.series.filter((row) => Math.abs(row.threshold - threshold) < 0.001);
  }, [impact, threshold]);

  const displayedPredictions = useMemo(() => {
    if (!selectedModel) return recentPredictions.slice(0, 8);
    return recentPredictions
      .filter((row) => row.model_key === selectedModel.model_key)
      .slice(0, 8);
  }, [recentPredictions, selectedModel]);

  const explainedPrediction = useMemo(() => {
    return displayedPredictions.find((row) => row.explanation?.groups?.length) ?? null;
  }, [displayedPredictions]);

  async function handlePredict() {
    if (!selectedModel) return;
    setIsPredicting(true);
    setError(null);
    try {
      const editedFeatures = currentSample
        ? { ...currentSample.features, ...edits }
        : undefined;
      const predictions = await runPrediction({
        modelKey: selectedModel.model_key,
        sampleRow: sampleIndex,
        editedFeatures,
      });
      setRecentPredictions((existing) => [...predictions, ...existing].slice(0, 8));
      setActivePage("prediction");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Prediction failed");
    } finally {
      setIsPredicting(false);
    }
  }

  function renderPage() {
    const latestPrediction = displayedPredictions[0] ?? null;

    switch (activePage) {
      case "home":
        return (
          <HomePage
            dashboard={dashboard}
            models={models}
            selectedModel={selectedModel}
            latestPrediction={latestPrediction}
            onNavigate={setActivePage}
          />
        );
      case "input":
        return (
          <InputPage
            sampleIndex={sampleIndex}
            setSampleIndex={setSampleIndex}
            samples={samples}
            editableFeatureNames={editableFeatureNames}
            currentSample={currentSample}
            edits={edits}
            setEdits={setEdits}
            isPredicting={isPredicting}
            selectedModel={selectedModel}
            onPredict={handlePredict}
          />
        );
      case "prediction":
        return (
          <PredictionPage
            latestPrediction={latestPrediction}
            predictions={displayedPredictions}
            selectedModel={selectedModel}
            extractedFeatures={null}
            onNavigate={setActivePage}
          />
        );
      case "explainability":
        return (
          <ExplainabilityPage
            prediction={explainedPrediction}
            selectedModel={selectedModel}
            features={features}
            onNavigate={setActivePage}
          />
        );
      case "model-insights":
        return (
          <ModelInsightsPage
            selectedModel={selectedModel}
            featureCount={samples?.feature_names.length ?? null}
            features={features}
            confusionMatrix={confusionMatrix}
            groupChartData={groupChartData}
            threshold={threshold}
            setThreshold={setThreshold}
            thresholdRows={thresholdRows}
          />
        );
      case "about":
        return <AboutPage modelCount={models.length} datasetCount={dashboard?.datasets.length ?? 0} />;
      default:
        return null;
    }
  }

  if (loading) {
    return (
      <main className="loading-screen">
        <Loader2 className="spin" />
        <span>Loading model artifacts...</span>
      </main>
    );
  }

  return (
    <div className="app-shell">
      <aside className="sidebar" aria-label="Main navigation">
        <div className="brand profile-brand" aria-label="Default profile">
          <div className="brand-mark">
            <CircleUserRound size={34} />
          </div>
        </div>

        <nav className="nav-list">
          {navItems.map((item) => (
            <button
              key={item.id}
              className={activePage === item.id ? "active" : undefined}
              onClick={() => setActivePage(item.id)}
            >
              <item.icon size={19} />
              <span>{item.label}</span>
            </button>
          ))}
        </nav>

        <small>v0.1.0</small>
      </aside>

      <main className="workspace">
        <header className="topbar">
          <label className="model-picker">
            <span>Model</span>
            <select value={selectedKey} onChange={(event) => setSelectedKey(event.target.value)}>
              {models.map((model) => (
                <option key={model.model_key} value={model.model_key}>
                  {model.model_key}
                </option>
              ))}
            </select>
          </label>

          <div className="status-grid">
            <StatusItem label="Model Status" value={statusLabel(selectedModel?.status)} icon={<CheckCircle2 size={16} />} />
            <StatusItem label="Backend" value={selectedModel?.display_family ?? "--"} icon={<Sparkles size={16} />} />
            <StatusItem label="Feature Count" value={String(samples?.feature_names.length ?? "--")} icon={<Info size={16} />} />
            <button className="icon-button" onClick={() => window.location.reload()} aria-label="Refresh">
              <RefreshCw size={18} />
            </button>
          </div>
        </header>

        {error && <div className="error-banner">{error}</div>}

        {renderPage()}

        <footer className="disclaimer">
          <Info size={16} />
          Predictions are for research support only and are not intended for clinical decision-making or diagnosis.
        </footer>
      </main>
    </div>
  );
}
function StatusItem({ label, value, icon }: { label: string; value: string; icon: JSX.Element }) {
  return (
    <div className="status-item">
      <span>{label}</span>
      <strong>
        {icon}
        {value}
      </strong>
    </div>
  );
}
