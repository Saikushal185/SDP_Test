"use client";

import { useEffect, useState } from "react";
import {
  FaCheckCircle,
  FaDownload,
  FaExclamationTriangle,
  FaFileCsv,
  FaMicroscope,
  FaPlay,
  FaTimesCircle,
} from "react-icons/fa";
import {
  ActionButton,
  EmptyState,
  PageIntro,
  Panel,
  SiteContainer,
  StatusPill,
} from "@/components/site/ui";
import {
  batchEvaluate,
  fetchCrossDatasetSummary,
  fetchDatasetTemplate,
  fetchModelRegistry,
  type BatchEvaluationResult,
  type CrossDatasetSummary,
  type DatasetTemplate,
  type ModelRegistryData,
} from "@/lib/api";
import { modelDisplayName, predictionLabel } from "@/lib/prediction-utils";

function percent(value: number | null | undefined): string {
  return typeof value === "number" ? `${(value * 100).toFixed(1)}%` : "N/A";
}

export default function DatasetTestPage() {
  const [registry, setRegistry] = useState<ModelRegistryData | null>(null);
  const [template, setTemplate] = useState<DatasetTemplate | null>(null);
  const [crossSummary, setCrossSummary] = useState<CrossDatasetSummary | null>(null);
  const [selectedModel, setSelectedModel] = useState("xgboost");
  const [csvText, setCsvText] = useState("");
  const [fileName, setFileName] = useState("");
  const [result, setResult] = useState<BatchEvaluationResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;

    Promise.all([
      fetchModelRegistry(),
      fetchDatasetTemplate(),
      fetchCrossDatasetSummary(),
    ])
      .then(([registryData, templateData, summaryData]) => {
        if (cancelled) {
          return;
        }
        setRegistry(registryData);
        setTemplate(templateData);
        setCrossSummary(summaryData);
        setSelectedModel(registryData.best_model || registryData.models[0]?.key || "xgboost");
      })
      .catch((fetchError) => {
        if (!cancelled) {
          setError(
            fetchError instanceof Error
              ? fetchError.message
              : "Failed to load dataset testing metadata."
          );
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const selectedDisplayName = modelDisplayName(selectedModel, registry?.models ?? []);

  const handleFileUpload = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    setFileName(file.name);
    setCsvText(await file.text());
    setResult(null);
    setError("");
    event.target.value = "";
  };

  const handleEvaluate = async () => {
    if (!csvText.trim()) {
      setError("Choose a CSV file before running strict evaluation.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      setResult(await batchEvaluate(csvText, selectedModel));
    } catch (evaluateError) {
      setError(
        evaluateError instanceof Error
          ? evaluateError.message
          : "Batch evaluation failed."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="pb-20">
      <PageIntro
        eyebrow="Dataset test"
        title="Test another CSV only when its feature schema matches."
        description="This route checks external datasets against the deployed pd_speech_features.csv feature schema. If columns are missing, evaluation stops and explains what is incompatible."
        actions={
          template ? (
            <a
              href={`data:text/csv;charset=utf-8,${encodeURIComponent(template.csv)}`}
              download="pd_speech_features_template.csv"
              className="button-primary inline-flex items-center rounded-full px-5 py-3 text-sm font-semibold"
            >
              <FaDownload className="mr-2 text-xs" />
              Download template
            </a>
          ) : null
        }
        meta={[
          {
            label: "Training dataset",
            value: template?.training_dataset ?? "Loading",
          },
          {
            label: "Required columns",
            value: template ? `${template.required_features.length}` : "Loading",
          },
          {
            label: "Selected model",
            value: selectedDisplayName,
          },
        ]}
      />

      <SiteContainer className="space-y-6 pt-10">
        {error ? (
          <div className="rounded-[24px] border border-[rgba(142,75,67,0.18)] bg-[rgba(142,75,67,0.1)] px-5 py-4 text-sm leading-7 text-[var(--danger)]">
            {error}
          </div>
        ) : null}

        <div className="grid gap-6 xl:grid-cols-[0.9fr_1.1fr]">
          <div className="space-y-6">
            <Panel>
              <p className="eyebrow">Strict evaluator</p>
              <h2 className="mt-3 text-2xl font-semibold text-[var(--text-strong)]">
                Upload a candidate CSV
              </h2>
              <p className="mt-3 text-sm leading-7 text-[var(--text-muted)]">
                The backend validates required feature columns first. Predictions and metrics are returned only after that check passes.
              </p>

              <div className="mt-8">
                <label className="field-label" htmlFor="dataset-model">
                  Model trained on pd_speech_features.csv
                </label>
                <select
                  id="dataset-model"
                  value={selectedModel}
                  onChange={(event) => setSelectedModel(event.target.value)}
                  className="select-shell"
                >
                  {(registry?.models ?? []).map((registeredModel) => (
                    <option key={registeredModel.key} value={registeredModel.key}>
                      {registeredModel.display_name}
                    </option>
                  ))}
                </select>
              </div>

              <label className="mt-7 flex min-h-[220px] cursor-pointer flex-col items-center justify-center rounded-[30px] border border-dashed border-[var(--border-strong)] bg-white/48 px-6 text-center transition hover:bg-white/70">
                <div className="grid h-16 w-16 place-items-center rounded-full bg-[var(--accent-soft)] text-2xl text-[var(--accent-strong)]">
                  <FaFileCsv />
                </div>
                <p className="mt-5 text-lg font-semibold text-[var(--text-strong)]">
                  {fileName || "Choose external CSV"}
                </p>
                <p className="mt-3 max-w-md text-sm leading-7 text-[var(--text-muted)]">
                  Include the required deployed feature columns. A label column such as `class`, `target`, or `status` is optional.
                </p>
                <span className="mt-5 inline-flex items-center gap-2 rounded-full border border-[var(--border-strong)] bg-white/80 px-4 py-2 text-sm font-semibold text-[var(--accent-strong)]">
                  Browse CSV
                </span>
                <input
                  type="file"
                  accept=".csv"
                  className="hidden"
                  onChange={handleFileUpload}
                />
              </label>

              <ActionButton
                onClick={handleEvaluate}
                disabled={loading || !csvText}
                className="mt-7 w-full"
              >
                {loading ? (
                  "Checking schema and evaluating"
                ) : (
                  <span className="flex items-center gap-2">
                    <FaPlay className="text-xs" />
                    Check compatibility and evaluate
                  </span>
                )}
              </ActionButton>
            </Panel>

            <Panel tone="muted">
              <p className="eyebrow">Template preview</p>
              <h2 className="mt-3 text-2xl font-semibold text-[var(--text-strong)]">
                Required header starts like this
              </h2>
              <p className="mt-3 text-sm leading-7 text-[var(--text-muted)]">
                The downloadable template includes all required feature columns plus an optional `class` label.
              </p>
              <pre className="mt-6 max-h-48 overflow-auto rounded-[22px] border border-[var(--border-subtle)] bg-[rgba(17,33,38,0.04)] p-4 text-xs leading-6 text-[var(--text-strong)]">
                {template?.csv.split("\n")[0] ?? "Loading template..."}
              </pre>
            </Panel>
          </div>

          <div className="space-y-6">
            {result ? (
              <Panel tone={result.compatible ? "strong" : "default"}>
                <div className="flex flex-wrap items-start justify-between gap-4">
                  <div>
                    <StatusPill tone={result.compatible ? "positive" : "critical"}>
                      {result.compatible ? "compatible" : "blocked"}
                    </StatusPill>
                    <h2 className={result.compatible ? "mt-5 font-display text-4xl text-[#fff8ea]" : "mt-5 text-3xl font-semibold text-[var(--text-strong)]"}>
                      {result.compatible
                        ? "Strict evaluation completed"
                        : "Schema mismatch found"}
                    </h2>
                    <p className={result.compatible ? "mt-3 text-sm leading-7 text-[rgba(255,248,234,0.74)]" : "mt-3 text-sm leading-7 text-[var(--text-muted)]"}>
                      {result.message}
                    </p>
                  </div>
                  <StatusPill tone="accent">{result.row_count} rows</StatusPill>
                </div>

                <div className="mt-8 grid gap-3 sm:grid-cols-3">
                  <div className={result.compatible ? "console-stat" : "data-chip"}>
                    <span>Present</span>
                    <strong>{result.present_feature_count}</strong>
                  </div>
                  <div className={result.compatible ? "console-stat" : "data-chip"}>
                    <span>Required</span>
                    <strong>{result.required_feature_count}</strong>
                  </div>
                  <div className={result.compatible ? "console-stat" : "data-chip"}>
                    <span>Missing</span>
                    <strong>{result.missing_columns.length}</strong>
                  </div>
                </div>

                {!result.compatible ? (
                  <div className="mt-7 rounded-[24px] border border-[rgba(142,75,67,0.18)] bg-[rgba(142,75,67,0.1)] p-5">
                    <div className="flex items-start gap-3">
                      <FaTimesCircle className="mt-1 shrink-0 text-[var(--danger)]" />
                      <div>
                        <p className="font-semibold text-[var(--danger)]">
                          Missing required columns
                        </p>
                        <p className="mt-2 text-sm leading-7 text-[rgba(142,75,67,0.92)]">
                          {result.missing_columns.slice(0, 18).join(", ")}
                          {result.missing_columns.length > 18 ? " ..." : ""}
                        </p>
                      </div>
                    </div>
                  </div>
                ) : null}

                {result.compatible && result.prediction_summary ? (
                  <div className="mt-8 grid gap-4 lg:grid-cols-2">
                    <div className="rounded-[26px] border border-[rgba(255,248,234,0.1)] bg-[rgba(255,248,234,0.06)] p-5">
                      <p className="text-[0.72rem] font-bold uppercase tracking-[0.2em] text-[rgba(255,248,234,0.45)]">
                        Prediction mix
                      </p>
                      <div className="mt-5 grid grid-cols-2 gap-3">
                        <div>
                          <p className="font-display text-4xl text-[#fff8ea]">
                            {result.prediction_summary.positive}
                          </p>
                          <p className="mt-1 text-sm text-[rgba(255,248,234,0.66)]">
                            positive
                          </p>
                        </div>
                        <div>
                          <p className="font-display text-4xl text-[#fff8ea]">
                            {result.prediction_summary.negative}
                          </p>
                          <p className="mt-1 text-sm text-[rgba(255,248,234,0.66)]">
                            healthy
                          </p>
                        </div>
                      </div>
                    </div>

                    <div className="rounded-[26px] border border-[rgba(255,248,234,0.1)] bg-[rgba(255,248,234,0.06)] p-5">
                      <p className="text-[0.72rem] font-bold uppercase tracking-[0.2em] text-[rgba(255,248,234,0.45)]">
                        Metrics
                      </p>
                      <div className="mt-5 grid grid-cols-2 gap-3 text-sm text-[rgba(255,248,234,0.7)]">
                        <span>Accuracy</span>
                        <strong className="text-right text-[#fff8ea]">
                          {percent(result.metrics?.accuracy)}
                        </strong>
                        <span>Recall</span>
                        <strong className="text-right text-[#fff8ea]">
                          {percent(result.metrics?.recall)}
                        </strong>
                        <span>F1</span>
                        <strong className="text-right text-[#fff8ea]">
                          {percent(result.metrics?.f1)}
                        </strong>
                        <span>AUC</span>
                        <strong className="text-right text-[#fff8ea]">
                          {result.metrics?.auc?.toFixed(3) ?? "N/A"}
                        </strong>
                      </div>
                    </div>
                  </div>
                ) : null}
              </Panel>
            ) : (
              <EmptyState
                icon={<FaMicroscope />}
                title="No dataset checked yet"
                description="Upload an external CSV and run the strict evaluator to see compatibility, missing columns, predictions, and metrics."
              />
            )}

            {result?.compatible && result.predictions.length > 0 ? (
              <Panel>
                <p className="eyebrow">Prediction rows</p>
                <h2 className="mt-3 text-2xl font-semibold text-[var(--text-strong)]">
                  First evaluated records
                </h2>
                <div className="table-shell mt-7 max-h-[420px] overflow-auto">
                  <table>
                    <thead>
                      <tr>
                        <th className="text-left">Row</th>
                        <th className="text-left">Prediction</th>
                        <th className="text-center">Probability</th>
                        <th className="text-center">Confidence</th>
                        <th className="text-center">Correct</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.predictions.slice(0, 50).map((row) => (
                        <tr key={row.row_index}>
                          <td>{row.row_index + 1}</td>
                          <td>{predictionLabel(row.prediction)}</td>
                          <td className="text-center">
                            {(row.probability * 100).toFixed(1)}%
                          </td>
                          <td className="text-center">
                            {(row.confidence * 100).toFixed(1)}%
                          </td>
                          <td className="text-center">
                            {typeof row.correct === "boolean" ? (
                              row.correct ? (
                                <FaCheckCircle className="mx-auto text-[var(--success)]" />
                              ) : (
                                <FaTimesCircle className="mx-auto text-[var(--danger)]" />
                              )
                            ) : (
                              "N/A"
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Panel>
            ) : null}
          </div>
        </div>

        <Panel>
          <div className="flex items-start gap-3">
            <FaExclamationTriangle className="mt-1 shrink-0 text-[var(--warning)]" />
            <div>
              <p className="eyebrow">Existing artifact context</p>
              <h2 className="mt-3 text-2xl font-semibold text-[var(--text-strong)]">
                External study artifacts are not automatically direct-test datasets
              </h2>
              <p className="mt-3 text-sm leading-7 text-[var(--text-muted)]">
                The rows below show whether local artifact schemas contain the deployed required features. Incompatible schemas are still useful context, but strict model testing stays blocked.
              </p>
            </div>
          </div>

          <div className="mt-7 grid gap-4 md:grid-cols-2">
            {(crossSummary?.datasets ?? []).map((dataset) => (
              <div
                key={dataset.dataset_id}
                className="rounded-[24px] border border-[var(--border-subtle)] bg-white/58 p-5"
              >
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <p className="font-semibold text-[var(--text-strong)]">
                      {dataset.dataset_id.replaceAll("_", " ")}
                    </p>
                    <p className="mt-2 text-sm leading-7 text-[var(--text-muted)]">
                      {dataset.note}
                    </p>
                  </div>
                  <StatusPill tone={dataset.strict_compatible ? "positive" : "caution"}>
                    {dataset.strict_compatible ? "compatible" : "blocked"}
                  </StatusPill>
                </div>
                <div className="mt-5 grid grid-cols-3 gap-3">
                  <div className="data-chip">
                    <span>features</span>
                    <strong>{dataset.feature_count}</strong>
                  </div>
                  <div className="data-chip">
                    <span>overlap</span>
                    <strong>{dataset.required_overlap}</strong>
                  </div>
                  <div className="data-chip">
                    <span>missing</span>
                    <strong>{dataset.missing_required_count}</strong>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Panel>
      </SiteContainer>
    </div>
  );
}
