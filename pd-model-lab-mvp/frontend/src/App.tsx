import { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  BarChart3,
  CheckCircle2,
  ChevronRight,
  Download,
  Gauge,
  Info,
  Lightbulb,
  Loader2,
  Play,
  RefreshCw,
  Search,
  Sparkles,
  UploadCloud,
  Users,
  Waves,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  fetchDashboard,
  fetchFeatureImportance,
  fetchGroupImpact,
  fetchModels,
  fetchSamples,
  runPrediction,
} from "./api";
import type {
  DashboardResponse,
  FeatureImportance,
  GroupImpact,
  ModelInfo,
  PredictionRow,
  SamplePayload,
} from "./types";
import { ExplainabilityPanel } from "./ExplainabilityPanel";

const navItems = [
  { id: "dashboard", label: "Dashboard", icon: Gauge },
  { id: "predict", label: "Predict", icon: Activity },
  { id: "explainability", label: "Explainability", icon: Lightbulb },
  { id: "feature-importance", label: "Feature Importance", icon: BarChart3 },
  { id: "group-impact", label: "Group Impact", icon: Users },
];

function formatMetric(value?: number | null, digits = 3) {
  return typeof value === "number" ? value.toFixed(digits) : "--";
}

function formatPercent(value?: number | null) {
  return typeof value === "number" ? `${Math.round(value * 100)}%` : "--";
}

function shortFeatureName(value: string) {
  return value.length > 24 ? `${value.slice(0, 21)}...` : value;
}

function statusLabel(status?: string) {
  if (status === "repaired") return "Repaired / Inference-Ready";
  if (status === "ready") return "Inference-Ready";
  return "Unavailable";
}

export default function App() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [dashboard, setDashboard] = useState<DashboardResponse | null>(null);
  const [selectedKey, setSelectedKey] = useState("");
  const [samples, setSamples] = useState<SamplePayload | null>(null);
  const [sampleIndex, setSampleIndex] = useState(0);
  const [edits, setEdits] = useState<Record<string, number | null>>({});
  const [featureSearch, setFeatureSearch] = useState("");
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [features, setFeatures] = useState<FeatureImportance[]>([]);
  const [impact, setImpact] = useState<GroupImpact | null>(null);
  const [threshold, setThreshold] = useState(0.5);
  const [recentPredictions, setRecentPredictions] = useState<PredictionRow[]>([]);
  const [isPredicting, setIsPredicting] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const sections = {
    dashboard: useRef<HTMLElement>(null),
    predict: useRef<HTMLElement>(null),
    explainability: useRef<HTMLElement>(null),
    "feature-importance": useRef<HTMLElement>(null),
    "group-impact": useRef<HTMLElement>(null),
  };

  useEffect(() => {
    async function loadInitialData() {
      try {
        const [modelRows, dashboardPayload] = await Promise.all([fetchModels(), fetchDashboard()]);
        setModels(modelRows);
        setDashboard(dashboardPayload);
        const preferred =
          modelRows.find((model) => model.model_key === "pd_speech_features_local_VQC") ??
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
    async function loadModelData() {
      try {
        const [samplePayload, featureRows, impactPayload] = await Promise.all([
          fetchSamples(model.dataset_id, 5),
          fetchFeatureImportance(model.model_key),
          fetchGroupImpact(model.model_key),
        ]);
        setSamples(samplePayload);
        setFeatures(featureRows);
        setImpact(impactPayload);
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
    const filtered = featureSearch
      ? merged.filter((name) => name.toLowerCase().includes(featureSearch.toLowerCase()))
      : merged;
    return filtered.slice(0, 12);
  }, [samples, features, featureSearch]);

  const performanceRows = useMemo(() => {
    if (!dashboard || !selectedModel) return [];
    return dashboard.comparison.filter((row) => row.dataset_id === selectedModel.dataset_id);
  }, [dashboard, selectedModel]);

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
        editedFeatures: uploadedFile ? undefined : editedFeatures,
        file: uploadedFile,
      });
      setRecentPredictions((existing) => [...predictions, ...existing].slice(0, 8));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Prediction failed");
    } finally {
      setIsPredicting(false);
    }
  }

  function scrollTo(id: keyof typeof sections) {
    sections[id].current?.scrollIntoView({ behavior: "smooth", block: "start" });
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
        <div className="brand">
          <div className="brand-mark">
            <Waves size={27} />
          </div>
          <div>
            <strong>PD Speech</strong>
            <span>Model Lab</span>
          </div>
        </div>

        <nav className="nav-list">
          {navItems.map((item) => (
            <button key={item.id} onClick={() => scrollTo(item.id as keyof typeof sections)}>
              <item.icon size={19} />
              <span>{item.label}</span>
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div className="researcher-dot">R</div>
          <span>Researcher</span>
          <ChevronRight size={16} />
        </div>
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

        <section ref={sections.predict} className="panel prediction-panel" id="predict">
          <WorkflowStep number="1" title="Upload CSV">
            <label className="dropzone">
              <UploadCloud size={30} />
              <strong>{uploadedFile ? uploadedFile.name : "Drag & drop CSV file here"}</strong>
              <span>{uploadedFile ? `${Math.ceil(uploadedFile.size / 1024)} KB selected` : "Browse Files"}</span>
              <input
                type="file"
                accept=".csv,text/csv"
                onChange={(event) => setUploadedFile(event.target.files?.[0] ?? null)}
              />
            </label>
            {uploadedFile && (
              <button className="link-button" onClick={() => setUploadedFile(null)}>
                Use sample row instead
              </button>
            )}
          </WorkflowStep>

          <WorkflowStep number="2" title="Sample Row">
            <div className="row-tools">
              <label>
                Row
                <select value={sampleIndex} onChange={(event) => setSampleIndex(Number(event.target.value))}>
                  {samples?.rows.map((row, index) => (
                    <option key={row.row_index} value={index}>
                      {row.row_index + 1}
                    </option>
                  ))}
                </select>
              </label>
              <label className="feature-search">
                <Search size={14} />
                <input
                  value={featureSearch}
                  placeholder="Search features"
                  onChange={(event) => setFeatureSearch(event.target.value)}
                />
              </label>
            </div>
            <div className="feature-editor">
              {editableFeatureNames.map((feature) => (
                <label key={feature}>
                  <span>{feature}</span>
                  <input
                    type="number"
                    value={String(edits[feature] ?? currentSample?.features[feature] ?? "")}
                    onChange={(event) =>
                      setEdits((existing) => ({
                        ...existing,
                        [feature]: event.target.value === "" ? null : Number(event.target.value),
                      }))
                    }
                  />
                </label>
              ))}
            </div>
          </WorkflowStep>

          <WorkflowStep number="3" title="Predict">
            <p className="muted">Run prediction on the uploaded CSV or selected edited sample row.</p>
            <div className="predict-actions">
              <button className="primary-button" onClick={handlePredict} disabled={isPredicting || !selectedModel}>
                {isPredicting ? <Loader2 className="spin" size={17} /> : <Play size={17} />}
                Run Prediction
              </button>
              <button
                className="secondary-button"
                onClick={() => {
                  setUploadedFile(null);
                  setEdits({});
                }}
              >
                Clear
              </button>
            </div>
            <div className="model-note">
              <CheckCircle2 size={16} />
              <span>{selectedModel?.model_key ?? "No model selected"}</span>
            </div>
          </WorkflowStep>
        </section>

        <ExplainabilityPanel
          refProp={sections.explainability}
          prediction={explainedPrediction}
          selectedModel={selectedModel}
        />

        <section ref={sections.dashboard} className="panel" id="dashboard">
          <div className="section-heading">
            <div>
              <h2>Model Performance</h2>
              <p>Cross-validation metrics from saved dataset artifacts.</p>
            </div>
            <div className="dataset-chip">{selectedModel?.dataset_id}</div>
          </div>
          <PerformanceTable rows={performanceRows} />
        </section>

        <div className="analysis-grid">
          <section ref={sections["feature-importance"]} className="panel chart-panel" id="feature-importance">
            <div className="section-heading compact">
              <div>
                <h2>Feature Importance</h2>
                <p>Top drivers for the selected model.</p>
              </div>
            </div>
            <ResponsiveContainer width="100%" height={205}>
              <BarChart data={features} layout="vertical" margin={{ left: 18, right: 24 }}>
                <CartesianGrid stroke="#e8edf1" horizontal={false} />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis
                  dataKey="feature"
                  type="category"
                  width={155}
                  tick={{ fontSize: 11 }}
                  tickFormatter={shortFeatureName}
                />
                <Tooltip formatter={(value: number) => value.toFixed(4)} />
                <Bar dataKey="importance" fill="#139a9a" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </section>

          <section ref={sections["group-impact"]} className="panel chart-panel wide" id="group-impact">
            <div className="section-heading compact">
              <div>
                <h2>Group / Class Impact</h2>
                <p>Positive-rate shifts across decision thresholds.</p>
              </div>
              <label className="threshold-control">
                Threshold {threshold.toFixed(2)}
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={threshold}
                  onChange={(event) => setThreshold(Number(event.target.value))}
                />
              </label>
            </div>
            <div className="impact-layout">
              <ResponsiveContainer width="100%" height={205}>
                <LineChart data={groupChartData}>
                  <CartesianGrid stroke="#e8edf1" />
                  <XAxis dataKey="threshold" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} domain={[0, 1]} />
                  <Tooltip formatter={(value: number) => value.toFixed(3)} />
                  <Legend />
                  <Line type="monotone" dataKey="Healthy Control" stroke="#0f8b8d" strokeWidth={2.4} dot={false} />
                  <Line type="monotone" dataKey="Parkinson's (PD)" stroke="#d98a21" strokeWidth={2.4} dot={false} />
                </LineChart>
              </ResponsiveContainer>
              <div className="impact-summary">
                <h3>Threshold Summary</h3>
                {thresholdRows.map((row) => (
                  <div className="summary-row" key={row.group}>
                    <span>{row.group}</span>
                    <strong>{formatPercent(row.positive_rate)}</strong>
                    <small>N={row.n}</small>
                  </div>
                ))}
              </div>
            </div>
          </section>
        </div>

        <section className="panel recent-panel">
          <div className="section-heading compact">
            <div>
              <h2>Recent Predictions</h2>
              <p>Latest local results from CSV or edited sample rows.</p>
            </div>
            <button className="secondary-button">
              <Download size={16} />
              Download CSV
            </button>
          </div>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Source</th>
                  <th>Row</th>
                  <th>PD Probability</th>
                  <th>Predicted Label</th>
                  <th>Model</th>
                </tr>
              </thead>
              <tbody>
                {displayedPredictions.map((row, index) => (
                  <tr key={`${row.model_key}-${index}`}>
                    <td>{row.source}</td>
                    <td>{row.row_index + 1}</td>
                    <td>{row.probability.toFixed(3)}</td>
                    <td>{row.predicted_label}</td>
                    <td>{row.model_key}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

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

function WorkflowStep({
  number,
  title,
  children,
}: {
  number: string;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="workflow-step">
      <h2>
        <span>{number}</span>
        {title}
      </h2>
      {children}
    </div>
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
